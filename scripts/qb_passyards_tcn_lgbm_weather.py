# qb_passyards_tcn_lgbm_weather_results.py
# 6-game Pass-TCN -> LightGBM for QB passing yards, with weather/env context
# Saves artifacts under results/qb_pass_weather/

import os
import math
import json
import logging
from typing import List

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


# =========================
# Config
# =========================
PARQUET_PATH = os.environ.get(
    "PARQUET_PATH",
    # Point this to your weather-joined, patched parquet
    r"C:\Users\Alex\Desktop\sports-betting-v2\data\processed\player_games_with_weather_3h_patched.parquet"
)

# ---- NEW: results folder ----
RESULTS_ROOT = "results"
RUN_NAME = "qb_pass_weather"
ARTIFACT_DIR = os.path.join(RESULTS_ROOT, RUN_NAME)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Window length (PRIOR games only)
L = 6

# Pull prior games from previous seasons to fill early weeks?
CROSS_SEASON_WINDOWS = True  # True recommended; set False to reset at season boundary

# Encoder (TCN) hyperparams
EMBED_DIM = 64
TCN_DROPOUT = 0.10
TCN_LR = 1e-3
TCN_BATCH_SIZE = 256
TCN_EPOCHS = 25
TCN_EARLY_STOP = 5

# Year splits
TCN_TRAIN_YEARS = [2019, 2020, 2021]
TCN_VAL_YEARS   = [2022]

LGB_TRAIN_YEARS = [2019, 2020, 2021, 2022]
LGB_VAL_YEARS   = [2023]
LGB_TEST_YEARS  = [2024]

TARGET_COL = "pass_yards"


# =========================
# Logging
# =========================
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("qb_passyards_weather_results")
fh = logging.FileHandler(os.path.join(ARTIFACT_DIR, "run.log"), mode="w", encoding="utf-8")
fh.setLevel(logging.INFO)
fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
log.addHandler(fh)
log.info("Artifacts dir: %s", os.path.abspath(ARTIFACT_DIR))


# =========================
# Utils
# =========================
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)


def derive_opponent(row):
    if pd.isna(row.get("posteam")) or pd.isna(row.get("home_team")) or pd.isna(row.get("away_team")):
        return np.nan
    return row["away_team"] if row["posteam"] == row["home_team"] else row["home_team"]


def have_cols(df: pd.DataFrame, cols: List[str]) -> List[str]:
    present = [c for c in cols if c in df.columns]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        log.info("Dropping %d missing TCN cols (showing up to 12): %s", len(missing), missing[:12])
    return present


def mask_years(meta_df: pd.DataFrame, years: List[int]) -> np.ndarray:
    return meta_df["season"].isin(years).values


def as_category(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype("category")
    return df


# =========================
# Load & prep
# =========================
log.info("Loading parquet: %s", PARQUET_PATH)
df = pd.read_parquet(PARQUET_PATH)
log.info("Loaded %d rows, %d columns", len(df), df.shape[1])

for col in ["season", "week"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df["opponent"] = df.apply(derive_opponent, axis=1)
df["home_indicator"] = (df["posteam"] == df["home_team"]).astype("int8")

# Filter to QBs
if "has_qb" in df.columns:
    qb_df = df[df["has_qb"] == 1].copy()
else:
    qb_df = df[df["position"].astype(str).str.upper() == "QB"].copy()

# Identity key for grouping
if "qb_player" in qb_df.columns and qb_df["qb_player"].notna().any():
    qb_df["qb_key"] = qb_df["qb_player"].fillna("").astype(str)
else:
    qb_df["qb_key"] = qb_df["player_id"].astype(str)

# Sort
sort_cols = [c for c in ["qb_key", "season", "week", "game_date"] if c in qb_df.columns]
qb_df = qb_df.sort_values(sort_cols).reset_index(drop=True)

log.info("QB subset: %d rows, %d unique QBs", len(qb_df), qb_df["qb_key"].nunique())
if TARGET_COL not in qb_df.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not present in data.")


# =========================
# Pass-TCN feature whitelist (history-only)
# =========================
PASS_TCN_CANDIDATES = [
    # Volume
    "qb_dropbacks", "qb_pass_att", "qb_completions",
    # Efficiency / air
    "qb_ypa", "qb_ay_per_att", "qb_epa_per_db",
    # Pressure / mobility
    "qb_sacks", "qb_sack_rate_db", "qb_scrambles", "qb_scramble_rate_db",
    # Depth & formation
    "qb_depth_1_9_share", "qb_depth_10_19_share", "qb_depth_20p_share", "qb_depth_blos_share",
    "qb_shotgun_share_db", "qb_undercenter_share_db",
    # Rates / situational
    "qb_pass_rate_h1_all", "qb_pass_rate_h2_all", "qb_pass_rate_neutral_h1", "qb_pass_rate_neutral_h2",
    "qb_ed_neutral_pass_rate", "qb_ed_pass_rate_all", "qb_neutral_db_per_game",
    "qb_rz_att", "qb_i10_att",
    # Outcome history (prior games only)
    "pass_yards",
]
PASS_TCN_COLS = have_cols(qb_df, PASS_TCN_CANDIDATES)
log.info("Using %d Pass-TCN channels: %s", len(PASS_TCN_COLS), PASS_TCN_COLS)


# =========================
# Window builders (carry env into meta)
# =========================
META_BASE_COLS = [
    "season", "week", "posteam", "home_team", "away_team",
    "opponent", "home_indicator", "qb_key", "game_id", "position",
    # env features carried into meta
    "roof", "surface", "is_dome",
    "temp_effective", "wind_effective",
    "precip_3h_mm", "precip_available",
]
META_BASE_COLS = [c for c in META_BASE_COLS if c in qb_df.columns]


def build_windows_crossseason(df_qb: pd.DataFrame, feature_cols: List[str], target_col: str, L: int):
    rows, ys, metas, hist_lengths = [], [], [], []
    for qb_key, g in df_qb.groupby("qb_key", sort=False):
        g = g.reset_index(drop=True)
        feats = g[feature_cols].astype(float).fillna(0.0).values
        labels = g[target_col].astype(float).fillna(0.0).values

        for i in range(len(g)):
            start = max(0, i - L)
            window = feats[start:i, :]
            hist_len = window.shape[0]
            if hist_len < L:
                pad = np.zeros((L - hist_len, feats.shape[1]), dtype=np.float32)
                window = np.vstack([pad, window])
            window_cl = window.T.astype(np.float32)
            metas.append(g.iloc[i][META_BASE_COLS].copy())
            rows.append(window_cl)
            ys.append(float(labels[i]))
            hist_lengths.append(hist_len)
    X_seq = np.stack(rows, axis=0)  # (N, C, L)
    y = np.array(ys, dtype=np.float32)
    meta = pd.DataFrame(metas).reset_index(drop=True)
    hist_len = np.array(hist_lengths, dtype=np.int16)
    return X_seq, y, meta, hist_len


def build_windows_withinseason(df_qb: pd.DataFrame, feature_cols: List[str], target_col: str, L: int):
    rows, ys, metas, hist_lengths = [], [], [], []
    for (qb_key, season), g in df_qb.groupby(["qb_key", "season"], sort=False):
        g = g.reset_index(drop=True)
        feats = g[feature_cols].astype(float).fillna(0.0).values
        labels = g[target_col].astype(float).fillna(0.0).values

        for i in range(len(g)):
            start = max(0, i - L)
            window = feats[start:i, :]
            hist_len = window.shape[0]
            if hist_len < L:
                pad = np.zeros((L - hist_len, feats.shape[1]), dtype=np.float32)
                window = np.vstack([pad, window])
            window_cl = window.T.astype(np.float32)
            metas.append(g.iloc[i][META_BASE_COLS].copy())
            rows.append(window_cl)
            ys.append(float(labels[i]))
            hist_lengths.append(hist_len)
    X_seq = np.stack(rows, axis=0)  # (N, C, L)
    y = np.array(ys, dtype=np.float32)
    meta = pd.DataFrame(metas).reset_index(drop=True)
    hist_len = np.array(hist_lengths, dtype=np.int16)
    return X_seq, y, meta, hist_len


build_windows_fn = build_windows_crossseason if CROSS_SEASON_WINDOWS else build_windows_withinseason
log.info("Building %d-game windows (%s-season).", L, "cross" if CROSS_SEASON_WINDOWS else "within")

X_seq_raw, y_all, meta, hist_len = build_windows_fn(qb_df, PASS_TCN_COLS, TARGET_COL, L=L)
N, C, Lwin = X_seq_raw.shape
log.info("Window tensor: N=%d, C=%d, L=%d", N, C, Lwin)
hist_counts = pd.Series(hist_len).value_counts().sort_index()
log.info("History length distribution (0..%d): %s", L, dict(hist_counts))

if "position" not in meta.columns:
    meta["position"] = "QB"


# =========================
# Splits
# =========================
mask_tcn_train = mask_years(meta, TCN_TRAIN_YEARS)
mask_tcn_val   = mask_years(meta, TCN_VAL_YEARS)

mask_lgb_train = mask_years(meta, LGB_TRAIN_YEARS)
mask_lgb_val   = mask_years(meta, LGB_VAL_YEARS)
mask_lgb_test  = mask_years(meta, LGB_TEST_YEARS)

log.info("TCN train rows: %d | TCN val rows: %d", mask_tcn_train.sum(), mask_tcn_val.sum())
log.info(
    "LGB train rows: %d | LGB val rows: %d | LGB test rows: %d",
    mask_lgb_train.sum(), mask_lgb_val.sum(), mask_lgb_test.sum()
)


# =========================
# Scale TCN inputs (fit on TCN train only)
# =========================
X_flat = X_seq_raw.transpose(0, 2, 1).reshape(N * Lwin, C)  # (N*L, C)
scaler = StandardScaler()
scaler.fit(X_flat[mask_tcn_train.repeat(Lwin)])
X_scaled = scaler.transform(X_flat).reshape(N, Lwin, C).transpose(0, 2, 1)  # (N, C, L)


# =========================
# Torch Dataset / Model
# =========================
class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
    def forward(self, x):
        out = self.conv1(x); out = out[:, :, :x.size(2)]; out = self.relu1(out); out = self.drop1(out)
        out = self.conv2(out); out = out[:, :, :x.size(2)]; out = self.relu2(out); out = self.drop2(out)
        res = x if self.downsample is None else self.downsample(x); res = res[:, :, :x.size(2)]
        return out + res


class PassTCN(nn.Module):
    def __init__(self, in_ch, embed_dim=64, dropout=0.1):
        super().__init__()
        hidden = 128
        self.block1 = TemporalBlock(in_ch, hidden, kernel_size=3, dilation=1, dropout=dropout)
        self.block2 = TemporalBlock(hidden, hidden, kernel_size=3, dilation=2, dropout=dropout)
        self.block3 = TemporalBlock(hidden, hidden, kernel_size=3, dilation=4, dropout=dropout)
        self.proj = nn.Sequential(nn.Conv1d(hidden, embed_dim, kernel_size=1), nn.ReLU())
        self.head = nn.Linear(embed_dim, 1)
    def forward(self, x):  # x: (B, C, L)
        h = self.block1(x); h = self.block2(h); h = self.block3(h)
        z = self.proj(h)          # (B, E, L)
        z_last = z[:, :, -1]      # causal last step
        yhat = self.head(z_last).squeeze(-1)
        return yhat, z_last


# =========================
# Train Pass-TCN (2019-2021), validate 2022
# =========================
X_tcn_train = X_scaled[mask_tcn_train]
y_tcn_train = y_all[mask_tcn_train]
X_tcn_val   = X_scaled[mask_tcn_val]
y_tcn_val   = y_all[mask_tcn_val]

log.info("Training Pass-TCN | in_ch=%d, embed_dim=%d, device=%s", len(PASS_TCN_COLS), EMBED_DIM, DEVICE)
train_loader = DataLoader(SeqDataset(X_tcn_train, y_tcn_train), batch_size=TCN_BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(SeqDataset(X_tcn_val,   y_tcn_val),   batch_size=TCN_BATCH_SIZE, shuffle=False)

model = PassTCN(in_ch=len(PASS_TCN_COLS), embed_dim=EMBED_DIM, dropout=TCN_DROPOUT).to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=TCN_LR, weight_decay=1e-4)
loss_fn = nn.MSELoss()

best_val, best_state, no_improve = float("inf"), None, 0
for epoch in range(1, TCN_EPOCHS + 1):
    model.train()
    running = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad()
        yhat, _ = model(xb)
        loss = loss_fn(yhat, yb)
        loss.backward()
        opt.step()
        running += loss.item() * xb.size(0)
    train_mse = running / len(X_tcn_train)

    model.eval()
    running = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            yhat, _ = model(xb)
            running += loss_fn(yhat, yb).item() * xb.size(0)
    val_mse = running / len(X_tcn_val)
    val_rmse = math.sqrt(val_mse)

    log.info("Epoch %02d | TCN train MSE=%.2f | val MSE=%.2f (val RMSE=%.2f yards)",
             epoch, train_mse, val_mse, val_rmse)

    if val_mse + 1e-6 < best_val:
        best_val = val_mse
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= TCN_EARLY_STOP:
            log.info("TCN early stopping.")
            break

if best_state is not None:
    model.load_state_dict(best_state)
log.info("Best TCN val MSE: %.3f (RMSE=%.2f yards)", best_val, math.sqrt(best_val))

# Freeze encoder
for p in model.parameters():
    p.requires_grad_(False)
model.eval()

# Save encoder + scaler
torch.save(model.state_dict(), os.path.join(ARTIFACT_DIR, "pass_tcn.pt"))
with open(os.path.join(ARTIFACT_DIR, "scaler.json"), "w") as f:
    json.dump(
        {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "cols": PASS_TCN_COLS,
            "L": L,
            "cross_season": CROSS_SEASON_WINDOWS,
        },
        f,
    )
log.info("Saved TCN and scaler.")


# =========================
# Generate embeddings for ALL rows
# =========================
log.info("Generating embeddings for all rows (2019-2024).")
BATCH = 1024
embeddings = np.zeros((X_scaled.shape[0], EMBED_DIM), dtype=np.float32)
with torch.no_grad():
    for start in range(0, X_scaled.shape[0], BATCH):
        end = min(start + BATCH, X_scaled.shape[0])
        xb = torch.from_numpy(X_scaled[start:end]).float().to(DEVICE)
        _, z = model(xb)
        embeddings[start:end, :] = z.cpu().numpy()

E_df = pd.DataFrame(embeddings, columns=[f"passE_{i}" for i in range(EMBED_DIM)])


# =========================
# LightGBM data (with env features)
# =========================
meta = meta.copy()
meta["history_length"] = hist_len.astype(np.int16)

# Context columns for LGBM, including env
CTX_COLS = [
    "position", "posteam", "opponent",
    "home_indicator", "season", "week", "history_length",
    "roof", "surface", "is_dome",
    "temp_effective", "wind_effective",
    "precip_3h_mm", "precip_available",
]
CTX_COLS = [c for c in CTX_COLS if c in meta.columns]

ctx = meta[CTX_COLS].copy()

# Categorical features (includes roof/surface if present)
ctx = as_category(ctx, ["position", "posteam", "opponent", "roof", "surface"])

X_full = pd.concat([E_df, ctx], axis=1)
y_full = pd.Series(y_all, name=TARGET_COL)

mask_lgb_train = mask_years(meta, LGB_TRAIN_YEARS)
mask_lgb_val   = mask_years(meta, LGB_VAL_YEARS)
mask_lgb_test  = mask_years(meta, LGB_TEST_YEARS)

X_train = X_full[mask_lgb_train].reset_index(drop=True)
y_train = y_full[mask_lgb_train].reset_index(drop=True)

X_val   = X_full[mask_lgb_val].reset_index(drop=True)
y_val   = y_full[mask_lgb_val].reset_index(drop=True)

X_test  = X_full[mask_lgb_test].reset_index(drop=True)
y_test  = y_full[mask_lgb_test].reset_index(drop=True)

log.info("LGB datasets | train=%d | val=%d | test=%d", len(X_train), len(X_val), len(X_test))

cat_feats = [c for c in ["position", "posteam", "opponent", "roof", "surface"] if c in X_train.columns]
lgb_train = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_feats, free_raw_data=False)
lgb_val   = lgb.Dataset(X_val,   label=y_val,   categorical_feature=cat_feats,
                        reference=lgb_train, free_raw_data=False)

params = dict(
    objective="mae",
    metric=["l1", "l2"],
    learning_rate=0.05,
    num_leaves=127,
    min_data_in_leaf=100,
    feature_fraction=0.9,
    bagging_fraction=0.8,
    bagging_freq=1,
    verbosity=-1,
    seed=SEED,
)

log.info("Training LightGBM head for passing yards (with env features).")

callbacks = [
    lgb.early_stopping(stopping_rounds=200, verbose=True),
    lgb.log_evaluation(period=50),
]

gbm = lgb.train(
    params=params,
    train_set=lgb_train,
    valid_sets=[lgb_train, lgb_val],
    valid_names=["train", "valid"],
    num_boost_round=5000,
    callbacks=callbacks,
)


def evaluate(name: str, X: pd.DataFrame, y: pd.Series):
    pred = gbm.predict(X, num_iteration=gbm.best_iteration)
    mae = mean_absolute_error(y, pred)
    rmse = math.sqrt(mean_squared_error(y, pred))
    log.info("[%s] MAE=%.3f | RMSE=%.3f", name, mae, rmse)
    return pred, float(mae), float(rmse)


val_pred,  mae23, rmse23 = evaluate("Val 2023", X_val,  y_val)
test_pred, mae24, rmse24 = evaluate("Test 2024", X_test, y_test)

# Save artifacts & metrics
gbm.save_model(os.path.join(ARTIFACT_DIR, "lgbm_qb_pass_weather.txt"))
with open(os.path.join(ARTIFACT_DIR, "metrics.json"), "w") as f:
    json.dump(
        {
            "val_2023": {"mae": mae23, "rmse": rmse23},
            "test_2024": {"mae": mae24, "rmse": rmse24},
        },
        f,
        indent=2,
    )

with open(os.path.join(ARTIFACT_DIR, "results.txt"), "w") as f:
    f.write(f"Val 2023: MAE={mae23:.3f}, RMSE={rmse23:.3f}\n")
    f.write(f"Test 2024: MAE={mae24:.3f}, RMSE={rmse24:.3f}\n")

# Save row-level predictions for inspection
val_rows  = meta[mask_lgb_val].reset_index(drop=True)
test_rows = meta[mask_lgb_test].reset_index(drop=True)
pd.DataFrame({
    "season":  val_rows["season"],
    "week":    val_rows["week"],
    "qb_key":  val_rows["qb_key"],
    "game_id": val_rows.get("game_id", np.nan),
    "actual_pass_yards": y_val,
    "pred_pass_yards":   val_pred,
    "abs_err":           np.abs(val_pred - y_val),
}).to_csv(os.path.join(ARTIFACT_DIR, "preds_val_2023.csv"), index=False)
pd.DataFrame({
    "season":  test_rows["season"],
    "week":    test_rows["week"],
    "qb_key":  test_rows["qb_key"],
    "game_id": test_rows.get("game_id", np.nan),
    "actual_pass_yards": y_test,
    "pred_pass_yards":   test_pred,
    "abs_err":           np.abs(test_pred - y_test),
}).to_csv(os.path.join(ARTIFACT_DIR, "preds_test_2024.csv"), index=False)

log.info("Saved LightGBM model, metrics, results.txt, and predictions CSVs. Done.")

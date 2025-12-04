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
# CONFIGURATION
# =========================
BASE_DIR = r'/Users/alexcory/Documents/sports-betting-v2/data/processed'
PARQUET_FILE = 'rb_modeling_with_defense_with_roles_v5.parquet'
PARQUET_PATH = os.path.join(BASE_DIR, PARQUET_FILE)

ARTIFACT_DIR = "artifacts_volume_v5"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

TARGET_COL = "rb_carries" 

# Hyperparams
SEED = 42
# Check for Apple Silicon (MPS)
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

L = 6 
CROSS_SEASON_WINDOWS = True 
EMBED_DIM = 64
TCN_DROPOUT = 0.1
TCN_LR = 1e-3
TCN_BATCH_SIZE = 256
TCN_EPOCHS = 35
TCN_EARLY_STOP = 5

# Splits
TCN_TRAIN_YEARS = [2019, 2020, 2021]
TCN_VAL_YEARS   = [2022]
LGB_TRAIN_YEARS = [2019, 2020, 2021, 2022]
LGB_VAL_YEARS   = [2023]
LGB_TEST_YEARS  = [2024]

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("vol_v5")
fh = logging.FileHandler(os.path.join(ARTIFACT_DIR, "run.log"), mode="w", encoding="utf-8")
fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
log.addHandler(fh)

# Utils
def set_seed(seed):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
set_seed(SEED)

def as_category(df, cols):
    for c in cols:
        if c in df.columns: df[c] = df[c].astype("category")
    return df

# =========================
# LOAD & PREP
# =========================
log.info(f"Loading data from: {PARQUET_PATH}")
if not os.path.exists(PARQUET_PATH):
    raise FileNotFoundError(f"File not found: {PARQUET_PATH}")

df = pd.read_parquet(PARQUET_PATH)

if "player_id" in df.columns: df["player_key"] = df["player_id"].astype(str)
df = df.sort_values(["player_key", "season", "week"]).reset_index(drop=True)

# Engineer Recency
log.info("Engineering Recency Features...")
df['last_week_carries'] = df.groupby('player_key')['rb_carries'].shift(1).fillna(0)
df['last_week_snap_share'] = df.groupby('player_key')['rb_ed_carry_share_all'].shift(1).fillna(0)

# =========================
# FEATURE DEFINITIONS
# =========================
VOL_TCN_CANDIDATES = [
    "rb_carries", "rb_ed_carry_share_all", "rb_rz_carries", 
    "rb_rush_yards", "rb_stuffed_rate",
    "starter_qb_scramble_rate_db", 
    "implied_game_script", "team_spread", "is_home"
]
VOL_TCN_COLS = [c for c in VOL_TCN_CANDIDATES if c in df.columns]

META_COLS = [
    "season", "week", "posteam", "opponent", "player_key", "full_name",
    "implied_game_script", "team_spread", "is_home", "wind_effective",
    "opponent_rush_epa_allowed", "opponent_rush_yards_per_game_allowed",
    "depth_rank", "is_projected_starter", "is_active",
    "last_week_carries", "last_week_snap_share"
]
META_COLS = [c for c in META_COLS if c in df.columns]

log.info(f"TCN Features: {len(VOL_TCN_COLS)}")
log.info(f"Meta Features: {len(META_COLS)}")

# =========================
# BUILD WINDOWS
# =========================
def build_windows(df_main, feature_cols, target_col, L):
    rows, ys, metas = [], [], []
    for _, g in df_main.groupby("player_key", sort=False):
        g = g.reset_index(drop=True)
        feats = g[feature_cols].astype(float).fillna(0.0).values
        labels = g[target_col].astype(float).fillna(0.0).values
        
        for i in range(len(g)):
            start = max(0, i - L)
            window = feats[start:i, :]
            if window.shape[0] < L:
                pad = np.zeros((L - window.shape[0], feats.shape[1]), dtype=np.float32)
                window = np.vstack([pad, window])
            
            metas.append(g.iloc[i][META_COLS].copy())
            rows.append(window.T.astype(np.float32))
            ys.append(float(labels[i]))
            
    return np.stack(rows), np.array(ys, dtype=np.float32), pd.DataFrame(metas).reset_index(drop=True)

log.info("Building windows...")
X_seq_raw, y_all, meta = build_windows(df, VOL_TCN_COLS, TARGET_COL, L)
N, C, Lwin = X_seq_raw.shape

# =========================
# SPLITS & SCALING
# =========================
mask_tcn_tr = meta["season"].isin(TCN_TRAIN_YEARS).values
mask_tcn_vl = meta["season"].isin(TCN_VAL_YEARS).values
mask_lgb_tr = meta["season"].isin(LGB_TRAIN_YEARS).values
mask_lgb_vl = meta["season"].isin(LGB_VAL_YEARS).values
mask_lgb_te = meta["season"].isin(LGB_TEST_YEARS).values

X_flat = X_seq_raw.transpose(0, 2, 1).reshape(N * Lwin, C)
scaler = StandardScaler()
scaler.fit(X_flat[mask_tcn_tr.repeat(Lwin)]) 
X_scaled = scaler.transform(X_flat).reshape(N, Lwin, C).transpose(0, 2, 1)

# =========================
# TCN MODEL (FIXED)
# =========================
class SeqDataset(Dataset):
    def __init__(self, X, y): self.X=torch.from_numpy(X).float(); self.y=torch.from_numpy(y).float()
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.y[i]

class RushTCN(nn.Module):
    def __init__(self, in_ch, embed_dim):
        super().__init__()
        # Encoder
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 64, 3, padding=2, dilation=1), nn.ReLU(), nn.Dropout(TCN_DROPOUT),
            nn.Conv1d(64, 64, 3, padding=4, dilation=2), nn.ReLU(), nn.Dropout(TCN_DROPOUT),
            nn.Conv1d(64, embed_dim, 1), nn.AdaptiveMaxPool1d(1)
        )
        # Prediction Head (FIX: Added this so loss calculation works)
        self.head = nn.Linear(embed_dim, 1)

    def forward(self, x):
        # 1. Get Embedding (B, Embed_Dim)
        z = self.net(x).squeeze(-1) 
        # 2. Get Prediction (B)
        pred = self.head(z).squeeze(-1)
        # Return BOTH
        return pred, z

log.info(f"Training TCN on {DEVICE}...")
dl_tr = DataLoader(SeqDataset(X_scaled[mask_tcn_tr], y_all[mask_tcn_tr]), batch_size=TCN_BATCH_SIZE, shuffle=True)
dl_vl = DataLoader(SeqDataset(X_scaled[mask_tcn_vl], y_all[mask_tcn_vl]), batch_size=TCN_BATCH_SIZE, shuffle=False)

model = RushTCN(len(VOL_TCN_COLS), EMBED_DIM).to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=TCN_LR)
crit = nn.MSELoss()

best_v, no_imp = float("inf"), 0

for ep in range(1, TCN_EPOCHS + 1):
    model.train()
    running_loss = 0.0
    for xb, yb in dl_tr:
        opt.zero_grad()
        # FIX: Unpack tuple (pred, embed)
        pred, _ = model(xb.to(DEVICE))
        loss = crit(pred, yb.to(DEVICE))
        loss.backward()
        opt.step()
        running_loss += loss.item() * xb.size(0)
    
    # Validation
    model.eval()
    run_v = 0.0
    with torch.no_grad():
        for xb, yb in dl_vl:
            pred, _ = model(xb.to(DEVICE))
            run_v += crit(pred, yb.to(DEVICE)).item() * xb.size(0)
    
    mse_v = run_v / len(mask_tcn_vl.nonzero()[0])
    
    if ep % 5 == 0:
        log.info(f"Epoch {ep} | Val RMSE: {math.sqrt(mse_v):.2f}")

# Embed Generation (FIXED LOOP)
model.eval()
embeds = np.zeros((N, EMBED_DIM), dtype=np.float32)
with torch.no_grad():
    for i in range(0, N, 1024):
        sl = slice(i, min(i+1024, N))
        # FIX: Unpack tuple, discard prediction, keep embedding (z)
        _, z = model(torch.from_numpy(X_scaled[sl]).float().to(DEVICE))
        embeds[sl] = z.cpu().numpy()

# =========================
# LIGHTGBM HEAD
# =========================
# Prepare Features
ctx_cols = [c for c in META_COLS if c not in ["season", "week", "player_key", "full_name", "game_id"]]
ctx = as_category(meta[ctx_cols].copy(), ["posteam", "opponent"])

X_lgb = pd.concat([pd.DataFrame(embeds, columns=[f"E{i}" for i in range(EMBED_DIM)]), ctx], axis=1)
y_lgb = pd.Series(y_all, name=TARGET_COL)

X_tr, y_tr = X_lgb[mask_lgb_tr], y_lgb[mask_lgb_tr]
X_vl, y_vl = X_lgb[mask_lgb_vl], y_lgb[mask_lgb_vl]
X_te, y_te = X_lgb[mask_lgb_te], y_lgb[mask_lgb_te]

# Exclude Week 18
val_clean = X_vl.index.isin(meta[meta["week"] != 18].index)
test_clean = X_te.index.isin(meta[meta["week"] != 18].index)

ds_tr = lgb.Dataset(X_tr, y_tr, categorical_feature=["posteam", "opponent"])
ds_vl = lgb.Dataset(X_vl[val_clean], y_vl[val_clean], categorical_feature=["posteam", "opponent"], reference=ds_tr)

params = {
    'objective': 'mae',
    'metric': 'mae',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'feature_fraction': 0.8,
    'seed': 42,
    'verbosity': -1
}

log.info("Training LightGBM Volume Model...")
gbm = lgb.train(
    params, 
    ds_tr, 
    valid_sets=[ds_tr, ds_vl], 
    num_boost_round=3000, 
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)]
)

# =========================
# EVALUATION
# =========================
preds = gbm.predict(X_te[test_clean])
actuals = y_te[test_clean]
mae = mean_absolute_error(actuals, preds)

log.info("------------------------------------------------")
log.info(f"FINAL TEST MAE (2024, No Wk18): {mae:.3f} carries")
log.info("------------------------------------------------")

# STARTER MAE CHECK
out_df = meta.loc[X_te[test_clean].index].copy()
out_df["pred_carries"] = preds
out_df["actual_carries"] = actuals.values

# Starter = Actual > 8 OR Pred > 8
starters = out_df[(out_df["actual_carries"] > 8) | (out_df["pred_carries"] > 8)]
mae_start = mean_absolute_error(starters["actual_carries"], starters["pred_carries"])
log.info(f"STARTER ONLY MAE: {mae_start:.3f} carries")

out_csv = os.path.join(ARTIFACT_DIR, "pred_volume_v5.csv")
out_df.to_csv(out_csv, index=False)
log.info(f"Predictions saved to {out_csv}")
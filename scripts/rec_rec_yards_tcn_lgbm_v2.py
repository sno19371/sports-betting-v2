# rec_rec_yards_tcn_lgbm_v2.py
# V2: Deeper Residual TCN -> LightGBM for receiving yards, with weather.

import os
import logging
import json
import math
from typing import List, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# Logging
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("rec_yards_v2")


# ============================================================
# Config
# ============================================================
PARQUET_PATH = os.environ.get(
    "REC_PARQUET_PATH",
    r"C:\Users\Alex\Desktop\sports-betting-v2\data\processed\player_games_with_weather_3h_patched.parquet"
)

ARTIFACT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "artifacts_rec_yards_v2"
)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

WINDOW_LEN = 6
EMBED_DIM = 64      # embedding dim & hidden channels
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================
# Residual TCN blocks
# ============================================================
class ResidualTCNBlock(nn.Module):
    """
    Residual 1D Conv block with dilation:
      Conv(dilation) -> ReLU -> Dropout -> Conv(dilation) -> ReLU
      + residual connection (with 1x1 if channels differ)
    """

    def __init__(self, in_ch: int, out_ch: int, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_ch,
            out_ch,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.conv2 = nn.Conv1d(
            out_ch,
            out_ch,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.match_res = None
        if in_ch != out_ch:
            self.match_res = nn.Conv1d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.act(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.act(out)

        if self.match_res is not None:
            residual = self.match_res(residual)

        return out + residual


class RecTCN(nn.Module):
    """
    Deeper TCN encoder + linear head.
    Input:  (B, C_in, L)
    Output:
      y_hat: (B,)          - scalar rec_yards prediction
      emb:   (B, EMBED_DIM) - embedding for LightGBM
    """

    def __init__(self, in_ch: int, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.proj = nn.Conv1d(in_ch, embed_dim, kernel_size=1)  # project channels -> embed_dim

        # 3 residual blocks with increasing dilation
        self.block1 = ResidualTCNBlock(embed_dim, embed_dim, dilation=1, dropout=0.1)
        self.block2 = ResidualTCNBlock(embed_dim, embed_dim, dilation=2, dropout=0.1)
        self.block3 = ResidualTCNBlock(embed_dim, embed_dim, dilation=4, dropout=0.1)

        self.head = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, C_in, L)
        """
        h = self.proj(x)        # (B, EMBED_DIM, L)
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)

        emb = h.mean(dim=-1)    # global avg pool -> (B, EMBED_DIM)
        y_hat = self.head(emb).squeeze(-1)  # (B,)
        return y_hat, emb


# ============================================================
# Helper: build receiver windows
# ============================================================
def build_rec_windows(
    df: pd.DataFrame,
    feat_cols: List[str],
    window_len: int = WINDOW_LEN,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    For each receiver game row, build a 6-game history window of receiver features.
    Uses previous games only; pads with zeros if < window_len history.

    Returns:
      X_seq: (N, C, L)
      y:     (N,)
      meta:  DataFrame of rows aligned with X_seq/y
    """
    df = df.sort_values(["player_id", "season", "week", "game_date"]).reset_index(drop=True)

    windows = []
    targets = []
    meta_rows = []

    for pid, g in df.groupby("player_id", sort=False):
        g = g.reset_index(drop=False)  # keep original index
        feats = g[feat_cols].to_numpy(dtype=float)  # (n_games, C)
        n = feats.shape[0]

        for t in range(n):
            # previous games only: [t-window_len, ..., t-1]
            start = max(0, t - window_len)
            prev = feats[start:t]
            hist_len = prev.shape[0]

            if hist_len < window_len:
                pad = np.zeros((window_len - hist_len, feats.shape[1]), dtype=float)
                prev = np.concatenate([pad, prev], axis=0)

            window = prev.T  # (C, L)
            windows.append(window)
            targets.append(float(g.loc[t, "rec_yards"]))
            meta_rows.append(g.loc[t])

    X_seq = np.stack(windows, axis=0)   # (N, C, L)
    y = np.array(targets, dtype=float)
    meta = pd.DataFrame(meta_rows)
    return X_seq, y, meta


# ============================================================
# Main
# ============================================================
def main():
    log.info(f"Loading receiver dataset: {PARQUET_PATH}")
    df = pd.read_parquet(PARQUET_PATH)
    log.info("Loaded %d rows, %d columns", len(df), len(df.columns))

    # 1) Filter to wide receivers & tight ends
    if "position" not in df.columns:
        raise ValueError("Expected 'position' column.")
    if "rec_yards" not in df.columns:
        raise ValueError("Expected 'rec_yards' target column.")

    rec = df[df["position"].isin(["WR", "TE"])].copy()
    rec = rec[~rec["rec_yards"].isna()].copy()
    log.info("Receiver subset: %d rows, %d unique players", len(rec), rec["player_id"].nunique())

    # 2) TCN feature columns (receiving history)
    TCN_COLS = [
        "rec_targets",
        "rec_receptions",
        "rec_rec_yards",
        "rec_rec_tds",
        "rec_yac",
        "rec_air_yards",
        "rec_adot",
        "rec_target_share",
        "rec_wopr",
        "rec_racr",
        "rec_fd_receptions",
        "rec_explosive15_receptions",
    ]

    missing = [c for c in TCN_COLS if c not in rec.columns]
    if missing:
        raise ValueError(f"Missing receiver TCN feature cols: {missing}")

    log.info("Using %d receiver TCN channels: %s", len(TCN_COLS), TCN_COLS)

    # 3) Build windows
    X_seq, y, meta = build_rec_windows(rec, TCN_COLS, WINDOW_LEN)
    N, C, L = X_seq.shape
    log.info("TCN window tensor: (N=%d, C=%d, L=%d)", N, C, L)

    # 4) Channel-wise normalization (fit on TCN train subset)
    if "season" not in meta.columns:
        raise ValueError("meta missing 'season' column.")

    tcn_train_mask = meta["season"] <= 2022
    tcn_val_mask   = meta["season"] == 2023

    # compute mean/std on training windows only
    train_data = X_seq[tcn_train_mask]   # (N_train, C, L)
    chan_mean = train_data.mean(axis=(0, 2))   # (C,)
    chan_std  = train_data.std(axis=(0, 2)) + 1e-6

    # apply normalization to all windows
    X_seq_norm = X_seq.copy()
    for c_idx in range(C):
        X_seq_norm[:, c_idx, :] = (X_seq_norm[:, c_idx, :] - chan_mean[c_idx]) / chan_std[c_idx]

    # save scaler for reference
    np.save(os.path.join(ARTIFACT_DIR, "tcn_chan_mean.npy"), chan_mean)
    np.save(os.path.join(ARTIFACT_DIR, "tcn_chan_std.npy"), chan_std)

    # 5) Build TCN train/val tensors
    X_train = torch.tensor(X_seq_norm[tcn_train_mask], dtype=torch.float32)
    y_train = torch.tensor(y[tcn_train_mask], dtype=torch.float32)
    X_val   = torch.tensor(X_seq_norm[tcn_val_mask],   dtype=torch.float32)
    y_val   = torch.tensor(y[tcn_val_mask],   dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RecTCN(in_ch=C, embed_dim=EMBED_DIM).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    # mini-batch loader
    BATCH_SIZE = 64
    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    # 6) Train TCN
    EPOCHS = 30
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad()
            y_hat, _ = model(xb)
            loss = loss_fn(y_hat, yb)
            loss.backward()
            opt.step()

            total_loss += loss.item()
            n_batches += 1

        avg_train_mse = total_loss / max(n_batches, 1)

        model.eval()
        with torch.no_grad():
            y_hat_val, _ = model(X_val.to(device))
            val_mse = loss_fn(y_hat_val, y_val.to(device)).item()
            val_rmse = math.sqrt(val_mse)

        log.info(
            "Epoch %02d | TCN train MSE=%.2f | val RMSE=%.2f yards",
            epoch, avg_train_mse, val_rmse
        )

    # 7) Get embeddings for all rows (using normalized windows)
    model.eval()
    with torch.no_grad():
        _, embeds = model(torch.tensor(X_seq_norm, dtype=torch.float32).to(device))
    embeds = embeds.cpu().numpy()  # (N, EMBED_DIM)

    # 8) LightGBM features: embeddings + weather
    WEATHER_COLS = ["temp_effective", "wind_effective", "precip_3h_mm", "is_dome"]

    rec_indexed = rec.reset_index(drop=True)
    if len(rec_indexed) != len(meta):
        log.warning(
            "meta rows (%d) != rec rows (%d); assuming same order and length.",
            len(meta), len(rec_indexed)
        )

    weather_mat = []
    for c in WEATHER_COLS:
        if c not in rec_indexed.columns:
            log.warning("Weather col %s missing in receiver table; filling with 0.", c)
            rec_indexed[c] = 0.0
        weather_mat.append(rec_indexed[c].to_numpy(dtype=np.float32))
    weather_mat = np.stack(weather_mat, axis=1)  # (N, 4)

    X_lgb = np.hstack([embeds, weather_mat])     # (N, EMBED_DIM+4)

    # 9) Season-based split for LightGBM
    lgb_train_mask = meta["season"] <= 2022
    lgb_val_mask   = meta["season"] == 2023
    lgb_test_mask  = meta["season"] == 2024

    X_train_lgb = X_lgb[lgb_train_mask]
    y_train_lgb = y[lgb_train_mask]

    X_val_lgb = X_lgb[lgb_val_mask]
    y_val_lgb = y[lgb_val_mask]

    X_test_lgb = X_lgb[lgb_test_mask]
    y_test_lgb = y[lgb_test_mask]

    lgb_train = lgb.Dataset(X_train_lgb, label=y_train_lgb)
    lgb_val   = lgb.Dataset(X_val_lgb, label=y_val_lgb, reference=lgb_train)

    params = {
        "objective": "regression",
        "metric": ["l1", "l2"],
        "learning_rate": 0.03,
        "num_leaves": 64,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_data_in_leaf": 20,
        "verbosity": -1,
        "seed": SEED,
    }

    log.info("Training LightGBM head for receiving yards (v2)…")
    callbacks = [
        early_stopping(stopping_rounds=200),
        log_evaluation(period=50),
    ]
    gbm = lgb.train(
        params,
        lgb_train,
        num_boost_round=1000,
        valid_sets=[lgb_train, lgb_val],
        valid_names=["train", "valid"],
        callbacks=callbacks,
    )

    # 10) Evaluate
    val_pred = gbm.predict(X_val_lgb, num_iteration=gbm.best_iteration)
    test_pred = gbm.predict(X_test_lgb, num_iteration=gbm.best_iteration)

    val_mae = mean_absolute_error(y_val_lgb, val_pred)
    val_rmse = math.sqrt(mean_squared_error(y_val_lgb, val_pred))

    test_mae = mean_absolute_error(y_test_lgb, test_pred)
    test_rmse = math.sqrt(mean_squared_error(y_test_lgb, test_pred))

    log.info("[Val 2023] MAE=%.3f | RMSE=%.3f", val_mae, val_rmse)
    log.info("[Test 2024] MAE=%.3f | RMSE=%.3f", test_mae, test_rmse)

    metrics = {
        "val_2023": {"mae": float(val_mae), "rmse": float(val_rmse)},
        "test_2024": {"mae": float(test_mae), "rmse": float(test_rmse)},
    }
    with open(os.path.join(ARTIFACT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Save preds for inspection
    meta_val = meta[lgb_val_mask].copy()
    meta_val["y_true"] = y_val_lgb
    meta_val["y_pred"] = val_pred
    meta_val.to_csv(os.path.join(ARTIFACT_DIR, "preds_val_2023.csv"), index=False)

    meta_test = meta[lgb_test_mask].copy()
    meta_test["y_true"] = y_test_lgb
    meta_test["y_pred"] = test_pred
    meta_test.to_csv(os.path.join(ARTIFACT_DIR, "preds_test_2024.csv"), index=False)

    with open(os.path.join(ARTIFACT_DIR, "results.txt"), "w") as f:
        f.write(f"[Val 2023] MAE={val_mae:.3f} | RMSE={val_rmse:.3f}\n")
        f.write(f"[Test 2024] MAE={test_mae:.3f} | RMSE={test_rmse:.3f}\n")

    log.info("Saved metrics, results.txt, scaler, and prediction CSVs. Done.")


if __name__ == "__main__":
    main()

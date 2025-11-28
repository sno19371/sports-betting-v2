# rb_rushyards_tcn_lgbm_mvp.py
# MVP: 6-game Rush-TCN -> LightGBM for RB rushing yards, with weather.

import os
import logging
import json
import math
from typing import List, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

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
log = logging.getLogger("rb_rush_mvp")


# ============================================================
# Config
# ============================================================
PARQUET_PATH = os.environ.get(
    "RB_PARQUET_PATH",
    r"C:\Users\Alex\Desktop\sports-betting-v2\data\processed\player_games_with_weather_3h_patched.parquet"
)

ARTIFACT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "artifacts_rb_rush_mvp"
)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

WINDOW_LEN = 6
EMBED_DIM = 64
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================
# TCN model: encoder + regression head
# ============================================================
class RushTCN(nn.Module):
    """
    Very simple depthwise TCN encoder + linear head.
    Input:  (B, C, L)
    Output: y_hat: (B,), emb: (B, EMBED_DIM)
    """

    def __init__(self, in_ch: int, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, in_ch, kernel_size=2, padding=1, groups=in_ch)
        self.conv2 = nn.Conv1d(in_ch, in_ch, kernel_size=2, padding=1, groups=in_ch)
        self.pointwise = nn.Conv1d(in_ch, embed_dim, kernel_size=1)
        self.act = nn.ReLU()
        self.head = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, C, L)
        returns:
          y_hat: (B,)
          emb:   (B, embed_dim)
        """
        h = self.act(self.conv1(x))
        h = self.act(self.conv2(h))
        h = self.pointwise(h)          # (B, embed_dim, L)
        emb = h.mean(dim=-1)           # global avg pool -> (B, embed_dim)
        y_hat = self.head(emb).squeeze(-1)  # (B,)
        return y_hat, emb


# ============================================================
# Helper: build RB windows
# ============================================================
def build_rb_windows(
    df: pd.DataFrame,
    feat_cols: List[str],
    window_len: int = WINDOW_LEN,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    For each RB game row, build a 6-game history window of RB features.
    Uses previous games only; pads with zeros if < 6 history.
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
        g = g.reset_index(drop=False)  # preserve original row mapping
        feats = g[feat_cols].to_numpy(dtype=float)  # (n_games, C)
        n = feats.shape[0]

        for t in range(n):
            # previous games only: [t-window_len, ..., t-1]
            start = max(0, t - window_len)
            prev = feats[start:t]      # (hist_len, C)
            hist_len = prev.shape[0]

            if hist_len < window_len:
                pad = np.zeros((window_len - hist_len, feats.shape[1]), dtype=float)
                prev = np.concatenate([pad, prev], axis=0)

            window = prev.T            # (C, L)
            windows.append(window)
            targets.append(float(g.loc[t, "rb_rush_yards"]))
            meta_rows.append(g.loc[t])

    X_seq = np.stack(windows, axis=0)   # (N, C, L)
    y = np.array(targets, dtype=float)
    meta = pd.DataFrame(meta_rows)
    return X_seq, y, meta


# ============================================================
# Main
# ============================================================
def main():
    log.info(f"Loading RB dataset: {PARQUET_PATH}")
    df = pd.read_parquet(PARQUET_PATH)
    log.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # 1) Filter RB rows
    if "position" not in df.columns:
        raise ValueError("Expected 'position' column to filter RBs.")
    if "rb_rush_yards" not in df.columns:
        raise ValueError("Expected 'rb_rush_yards' target column.")

    rb = df[df["position"] == "RB"].copy()
    rb = rb[~rb["rb_rush_yards"].isna()].copy()
    log.info("RB subset: %d rows, %d unique RBs", len(rb), rb["player_id"].nunique())

    # 2) TCN feature columns
    TCN_COLS = [
        "rb_carries",
        "rb_rush_yards",
        "rb_rush_tds",
        "rb_ypc",
        "rb_epa_sum",
        "rb_explosive10",
        "rb_explosive15",
        "rb_explosive20",
        "rb_rush_1st_downs",
        "rb_i10_rush_yards",
        "rb_rz_rush_yards",
    ]
    missing = [c for c in TCN_COLS if c not in rb.columns]
    if missing:
        raise ValueError(f"Missing TCN feature cols in RB dataframe: {missing}")
    log.info("Using %d TCN channels: %s", len(TCN_COLS), TCN_COLS)

    # 3) Build windows
    X_seq, y, meta = build_rb_windows(rb, TCN_COLS, WINDOW_LEN)
    N, C, L = X_seq.shape
    log.info("TCN window tensor: (N=%d, C=%d, L=%d)", N, C, L)

    # 4) Season-based splits
    if "season" not in meta.columns:
        raise ValueError("meta missing 'season' column for splits.")

    # TCN: train on <=2022, val on 2023
    tcn_train_mask = meta["season"] <= 2022
    tcn_val_mask = meta["season"] == 2023

    # LGB: train <=2022, val=2023, test=2024
    lgb_train_mask = meta["season"] <= 2022
    lgb_val_mask = meta["season"] == 2023
    lgb_test_mask = meta["season"] == 2024

    log.info(
        "TCN train=%d, val=%d",
        tcn_train_mask.sum(),
        tcn_val_mask.sum(),
    )
    log.info(
        "LGB train=%d, val=%d, test=%d",
        lgb_train_mask.sum(),
        lgb_val_mask.sum(),
        lgb_test_mask.sum(),
    )

    # 5) Prepare tensors
    X_train = torch.tensor(X_seq[tcn_train_mask], dtype=torch.float32)
    y_train = torch.tensor(y[tcn_train_mask], dtype=torch.float32)
    X_val = torch.tensor(X_seq[tcn_val_mask], dtype=torch.float32)
    y_val = torch.tensor(y[tcn_val_mask], dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RushTCN(in_ch=C, embed_dim=EMBED_DIM).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # 6) Train TCN
    EPOCHS = 25
    for epoch in range(1, EPOCHS + 1):
        model.train()
        opt.zero_grad()
        y_hat_train, _ = model(X_train.to(device))
        loss = loss_fn(y_hat_train, y_train.to(device))
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            y_hat_val, _ = model(X_val.to(device))
            val_mse = loss_fn(y_hat_val, y_val.to(device)).item()
            val_rmse = math.sqrt(val_mse)

        log.info(
            "Epoch %02d | TCN train MSE=%.2f | val RMSE=%.2f yards",
            epoch, loss.item(), val_rmse
        )

    # 7) Get embeddings for all rows
    model.eval()
    with torch.no_grad():
        _, embeds = model(torch.tensor(X_seq, dtype=torch.float32).to(device))
    embeds = embeds.cpu().numpy()        # (N, EMBED_DIM)

    # 8) LightGBM features: embeddings + weather
    WEATHER_COLS = ["temp_effective", "wind_effective", "precip_3h_mm", "is_dome"]
    for c in WEATHER_COLS:
        if c not in meta.columns and c not in rb.columns:
            # meta is per-window; weather columns are in rb (aligned by index)
            # We'll pull them from rb using meta's original index
            pass

    # meta rows correspond to rb rows; use rb for weather
    rb_indexed = rb.reset_index(drop=True)
    # sanity: meta and rb_indexed should align on row counts
    if len(rb_indexed) != len(meta):
        log.warning("meta rows (%d) != rb rows (%d); using meta indices where possible.", len(meta), len(rb_indexed))

    weather_mat = []
    for c in WEATHER_COLS:
        if c not in rb_indexed.columns:
            log.warning("Weather col %s missing in RB table; filling with 0.", c)
            rb_indexed[c] = 0.0
        weather_mat.append(rb_indexed[c].to_numpy(dtype=np.float32))
    weather_mat = np.stack(weather_mat, axis=1)  # (N, 4)

    X_lgb = np.hstack([embeds, weather_mat])     # (N, EMBED_DIM+4)

    # 9) Split for LightGBM
    X_train_lgb = X_lgb[lgb_train_mask]
    y_train_lgb = y[lgb_train_mask]

    X_val_lgb = X_lgb[lgb_val_mask]
    y_val_lgb = y[lgb_val_mask]

    X_test_lgb = X_lgb[lgb_test_mask]
    y_test_lgb = y[lgb_test_mask]

    lgb_train = lgb.Dataset(X_train_lgb, label=y_train_lgb)
    lgb_val = lgb.Dataset(X_val_lgb, label=y_val_lgb, reference=lgb_train)

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

    log.info("Training LightGBM head for RB rushing yards…")
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

    log.info("Saved metrics, results.txt, and prediction CSVs. Done.")


if __name__ == "__main__":
    main()

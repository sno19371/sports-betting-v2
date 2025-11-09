# add_ints_fumbles_total_tds.py
import argparse
import pandas as pd
from pathlib import Path

# Columns we may use from PBP (we'll handle missing ones gracefully)
PBP_COLS = [
    "game_id",
    "interception",              # 0/1 flag (preferred if present)
    "interception_player_id",    # fallback presence -> interception happened
    "passer_player_id",          # who THREW the pick (what we attribute to)
    "fumble",                    # 0/1 flag (preferred if present)
    "fumbled_1_player_id",       # credited primary fumbler
]

def read_pbp(path: str) -> pd.DataFrame:
    path_lower = path.lower()
    if path_lower.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        # CSV fallback
        df = pd.read_csv(path, low_memory=False)
    # keep only available columns
    keep = [c for c in PBP_COLS if c in df.columns]
    return df[keep].copy()

def as_str(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype("string")
    return df

def ensure_int_cols(df: pd.DataFrame, cols):
    for c in cols:
        if c not in df.columns:
            df[c] = 0
    df[cols] = df[cols].fillna(0).astype("int64")
    return df

def build_ints_by_passer(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Attribute interceptions to the passer who threw them:
      group by (game_id, passer_player_id) on intercepted plays.
    """
    if "passer_player_id" not in pbp.columns:
        return pd.DataFrame(columns=["game_id", "player_id", "ints"])

    # Robust interception flag:
    if "interception" in pbp.columns:
        inter_flag = (pbp["interception"] == 1)
    elif "interception_player_id" in pbp.columns:
        inter_flag = pbp["interception_player_id"].notna()
    else:
        return pd.DataFrame(columns=["game_id", "player_id", "ints"])

    ints = (
        pbp.loc[inter_flag & pbp["passer_player_id"].notna(), ["game_id", "passer_player_id"]]
           .rename(columns={"passer_player_id": "player_id"})
           .assign(ints=1)
           .groupby(["game_id", "player_id"], as_index=False)["ints"].sum()
    )
    return ints

def build_fumbles_by_fumbler(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Attribute fumbles to the primary fumbler (fumbled_1_player_id).
    """
    if "fumbled_1_player_id" not in pbp.columns:
        return pd.DataFrame(columns=["game_id", "player_id", "fumbles"])

    if "fumble" in pbp.columns:
        fum_flag = (pbp["fumble"] == 1)
        df = pbp.loc[fum_flag & pbp["fumbled_1_player_id"].notna(), ["game_id", "fumbled_1_player_id"]]
    else:
        df = pbp.loc[pbp["fumbled_1_player_id"].notna(), ["game_id", "fumbled_1_player_id"]]

    fumbles = (
        df.rename(columns={"fumbled_1_player_id": "player_id"})
          .assign(fumbles=1)
          .groupby(["game_id", "player_id"], as_index=False)["fumbles"].sum()
    )
    return fumbles

def main():
    ap = argparse.ArgumentParser(description="Add total_tds, ints (by passer), fumbles (by fumbler), total_turnovers to full games parquet.")
    ap.add_argument("--full", required=True, help="Path to your existing full games parquet")
    ap.add_argument("--pbp",  required=True, help="Path to play-by-play (.parquet or .csv)")
    ap.add_argument("--out",  required=True, help="Output parquet path")
    args = ap.parse_args()

    # Read inputs
    full = pd.read_parquet(args.full).copy()
    pbp  = read_pbp(args.pbp)

    # Normalize IDs for safe merges
    full = as_str(full, ["player_id", "game_id"])
    pbp  = as_str(pbp,  ["game_id", "passer_player_id", "fumbled_1_player_id"])

    # Build per-game, per-player ints (passer) and fumbles (fumbler)
    ints_df    = build_ints_by_passer(pbp)
    fumbles_df = build_fumbles_by_fumbler(pbp)

    # Merge turnovers into a single frame
    turn = ints_df.merge(fumbles_df, on=["game_id", "player_id"], how="outer")
    turn = ensure_int_cols(turn, ["ints", "fumbles"])
    turn["total_turnovers"] = (turn["ints"] + turn["fumbles"]).astype("int64")

    # Merge back to your full table
    out_df = full.merge(turn, on=["game_id", "player_id"], how="left")
    out_df = ensure_int_cols(out_df, ["ints", "fumbles", "total_turnovers"])

    # Ensure TD component columns exist
    for c in ("pass_tds", "rush_tds", "rec_tds"):
        if c not in out_df.columns:
            out_df[c] = 0
        out_df[c] = out_df[c].fillna(0)

    # total_tds = pass_tds + rush_tds + rec_tds
    out_df["total_tds"] = (
        out_df["pass_tds"].astype("int64")
        + out_df["rush_tds"].astype("int64")
        + out_df["rec_tds"].astype("int64")
    )

    # Write
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.out, index=False)
    print(f"Wrote {args.out} with shape {out_df.shape}")

if __name__ == "__main__":
    main()

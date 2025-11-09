# add_basic_box_from_pbp.py
import argparse
import pandas as pd
from pathlib import Path

KEEP_COLS = [
    # keys
    "game_id","posteam",
    # passing
    "passer_player_id","passing_yards","pass_touchdown",
    # rushing
    "rusher_player_id","rushing_yards","rush_touchdown",
    # receiving
    "receiver_player_id","receiving_yards",
    # TD scorer to make receiving TDs exact
    "td_player_id",
]

def read_parquet_any(path: str, columns=None) -> pd.DataFrame:
    p = Path(path)
    if p.is_dir():
        return pd.read_parquet(p.as_posix(), columns=columns)  # directory dataset
    return pd.read_parquet(path, columns=columns)

def safe_int(x: pd.Series) -> pd.Series:
    # turn booleans/NAs into integer 0/1
    return x.fillna(0).astype("int64")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", required=True, help="path to full_games.parquet")
    ap.add_argument("--pbp",  required=True, help="path to pbp parquet (file or directory)")
    ap.add_argument("--out",  required=True, help="output parquet path")
    args = ap.parse_args()

    # 1) Load full games
    full = pd.read_parquet(args.full)
    # normalize key dtypes
    full["game_id"] = full["game_id"].astype("string")
    full["posteam"]  = full["posteam"].astype("string")
    full["player_id"] = full["player_id"].astype("string")

    # 2) Load a slim PBP with only needed columns
    pbp = read_parquet_any(args.pbp, columns=KEEP_COLS).copy()
    # keep only plays that actually have a team on offense
    pbp = pbp[pbp["posteam"].notna()].copy()
    for c in ["game_id","posteam","passer_player_id","rusher_player_id",
              "receiver_player_id","td_player_id"]:
        if c in pbp.columns:
            pbp[c] = pbp[c].astype("string")

    # 3) Build per-role aggregates
    # Passing (by passer)
    p_pass = (
        pbp[pbp["passer_player_id"].notna()]
        .assign(pass_touchdown=safe_int(pbp.loc[pbp["passer_player_id"].notna(), "pass_touchdown"]))
        .groupby(["game_id","posteam","passer_player_id"], as_index=False)
        .agg(pass_yards=("passing_yards","sum"),
             pass_tds=("pass_touchdown","sum"))
        .rename(columns={"passer_player_id":"player_id"})
    )

    # Rushing (by rusher)
    p_rush = (
        pbp[pbp["rusher_player_id"].notna()]
        .assign(rush_touchdown=safe_int(pbp.loc[pbp["rusher_player_id"].notna(), "rush_touchdown"]))
        .groupby(["game_id","posteam","rusher_player_id"], as_index=False)
        .agg(rush_yards=("rushing_yards","sum"),
             rush_tds=("rush_touchdown","sum"))
        .rename(columns={"rusher_player_id":"player_id"})
    )

    # Receiving (by receiver) — TDs via scorer match for exactness
    # rec_tds counts a TD only when td_player_id == receiver_player_id
    recv = pbp[pbp["receiver_player_id"].notna()].copy()
    recv["rec_td_flag"] = (recv["td_player_id"] == recv["receiver_player_id"]).fillna(False).astype("int64")
    p_recv = (
        recv.groupby(["game_id","posteam","receiver_player_id"], as_index=False)
            .agg(rec_yards=("receiving_yards","sum"),
                 rec_tds=("rec_td_flag","sum"))
            .rename(columns={"receiver_player_id":"player_id"})
    )

    # 4) Combine role aggregates into one wide block
    agg = p_pass.merge(p_rush, on=["game_id","posteam","player_id"], how="outer")
    agg = agg.merge(p_recv, on=["game_id","posteam","player_id"], how="outer")

    # Fill NaNs with zeros for new stats
    for c in ["pass_yards","pass_tds","rush_yards","rush_tds","rec_yards","rec_tds"]:
        if c in agg.columns:
            agg[c] = agg[c].fillna(0).astype("int64")

    # 5) Merge onto full_games
    out = full.merge(agg, on=["game_id","posteam","player_id"], how="left")

    # 6) Ensure the six columns exist & are 0 when missing
    for c in ["pass_yards","pass_tds","rush_yards","rush_tds","rec_yards","rec_tds"]:
        if c not in out.columns:
            out[c] = 0
        else:
            out[c] = out[c].fillna(0).astype("int64")

    # Optional: place them after `position`
    cols = list(out.columns)
    for c in ["pass_yards","pass_tds","rush_yards","rush_tds","rec_yards","rec_tds"]:
        if c in cols:
            cols.remove(c)
    try:
        idx = cols.index("position") + 1
    except ValueError:
        idx = len(cols)
    cols = cols[:idx] + ["pass_yards","pass_tds","rush_yards","rush_tds","rec_yards","rec_tds"] + cols[idx:]
    out = out[cols]

    out.to_parquet(args.out, index=False)
    print(f"Wrote {args.out}  rows={len(out):,}  cols={out.shape[1]}")
    
if __name__ == "__main__":
    main()

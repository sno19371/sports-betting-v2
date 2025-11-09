import argparse
import pandas as pd
from pathlib import Path

PBP_COLS = [
    "game_id",
    "pass_touchdown", "rush_touchdown",
    "interception", "fumble",
    "passer_player_id", "receiver_player_id", "rusher_player_id",
    "fumbled_1_player_id",
]

def read_pbp(path: str) -> pd.DataFrame:
    path = str(path)
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path, columns=PBP_COLS)
    # CSV fallback
    return pd.read_csv(path, usecols=PBP_COLS, low_memory=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", required=True, help="full_games.parquet (input)")
    ap.add_argument("--pbp",  required=True, help="play-by-play file (.parquet or .csv)")
    ap.add_argument("--out",  required=True, help="output parquet path")
    args = ap.parse_args()

    full = pd.read_parquet(args.full)
    pbp  = read_pbp(args.pbp)

    # Make sure IDs are comparable strings
    full["player_id"] = full["player_id"].astype("string")
    for col in ["passer_player_id","receiver_player_id","rusher_player_id","fumbled_1_player_id"]:
        pbp[col] = pbp[col].astype("string")

    # ---- TDs by role ----
    pass_tds = (
        pbp.loc[pbp["pass_touchdown"] == 1, ["game_id","passer_player_id"]]
           .rename(columns={"passer_player_id": "player_id"})
           .assign(pass_tds=1)
           .groupby(["game_id","player_id"], as_index=False)["pass_tds"].sum()
    )

    rec_tds = (
        pbp.loc[pbp["pass_touchdown"] == 1, ["game_id","receiver_player_id"]]
           .rename(columns={"receiver_player_id": "player_id"})
           .assign(rec_tds=1)
           .groupby(["game_id","player_id"], as_index=False)["rec_tds"].sum()
    )

    rush_tds = (
        pbp.loc[pbp["rush_touchdown"] == 1, ["game_id","rusher_player_id"]]
           .rename(columns={"rusher_player_id": "player_id"})
           .assign(rush_tds=1)
           .groupby(["game_id","player_id"], as_index=False)["rush_tds"].sum()
    )

    # ---- Turnovers: INTs against passer; fumbles by fumbled_1 player ----
    ints = (
        pbp.loc[pbp["interception"] == 1, ["game_id","passer_player_id"]]
           .rename(columns={"passer_player_id": "player_id"})
           .assign(interceptions=1)
           .groupby(["game_id","player_id"], as_index=False)["interceptions"].sum()
    )

    fumbles = (
        pbp.loc[(pbp["fumble"] == 1) & pbp["fumbled_1_player_id"].notna(),
                ["game_id","fumbled_1_player_id"]]
           .rename(columns={"fumbled_1_player_id": "player_id"})
           .assign(fumbles=1)
           .groupby(["game_id","player_id"], as_index=False)["fumbles"].sum()
    )

    # ---- Combine the per-player per-game aggregates ----
    agg = pass_tds.merge(rec_tds,  on=["game_id","player_id"], how="outer") \
                  .merge(rush_tds, on=["game_id","player_id"], how="outer") \
                  .merge(ints,     on=["game_id","player_id"], how="outer") \
                  .merge(fumbles,  on=["game_id","player_id"], how="outer")

    # Fill missing counts with 0 and cast to int64
    for c in ["pass_tds","rec_tds","rush_tds","interceptions","fumbles"]:
        if c not in agg.columns:
            agg[c] = 0
    agg[["pass_tds","rec_tds","rush_tds","interceptions","fumbles"]] = \
        agg[["pass_tds","rec_tds","rush_tds","interceptions","fumbles"]].fillna(0).astype("int64")

    # Derived totals
    agg["total_tds"] = (agg["pass_tds"] + agg["rush_tds"] + agg["rec_tds"]).astype("int64")
    agg["total_turnovers"] = (agg["interceptions"] + agg["fumbles"]).astype("int64")

    # ---- Merge back into full_games ----
    out_df = full.merge(agg, on=["game_id","player_id"], how="left")
    for c in ["pass_tds","rec_tds","rush_tds","total_tds","interceptions","fumbles","total_turnovers"]:
        if c not in out_df.columns:
            out_df[c] = 0
    out_df[["pass_tds","rec_tds","rush_tds","total_tds","interceptions","fumbles","total_turnovers"]] = \
        out_df[["pass_tds","rec_tds","rush_tds","total_tds","interceptions","fumbles","total_turnovers"]].fillna(0).astype("int64")

    # Optional: put the new columns near your other outcome targets (append at end is also fine)
    # Here we just leave them where they land after merge.

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.out, index=False)
    print(f"Wrote {args.out} with shape {out_df.shape}")

if __name__ == "__main__":
    main()

# build_full_games.py
import argparse
import pandas as pd


def main(joined_path: str, roster_path: str, out_path: str):
    joined = pd.read_parquet(joined_path)
    roster = pd.read_parquet(roster_path)

    # Keep only what we need from roster and ensure comparable dtypes
    keep_cols = ["gsis_id", "full_name", "position"]
    extra_keys = []
    if "season" in roster.columns and "season" in joined.columns:
        keep_cols.append("season")
        extra_keys = ["season"]

    roster_small = roster[keep_cols].dropna(subset=["gsis_id"]).copy()
    roster_small["gsis_id"] = roster_small["gsis_id"].astype(str)

    # If season is present, prefer season-appropriate row; otherwise just drop duplicates
    if "season" in roster_small.columns:
        # If multiple rows per (season, gsis_id), keep the first occurrence
        roster_small = roster_small.drop_duplicates(subset=["season", "gsis_id"], keep="first")
    else:
        roster_small = roster_small.drop_duplicates(subset=["gsis_id"], keep="first")

    # Ensure joined id is string to match
    if "player_id" not in joined.columns:
        raise ValueError("joined_games is missing 'player_id' column required for join on gsis_id")
    joined["player_id"] = joined["player_id"].astype(str)

    # Join on gsis_id (and season if available)
    left_keys = extra_keys + ["player_id"]
    right_keys = extra_keys + ["gsis_id"]
    merged = joined.merge(
        roster_small.rename(columns={"gsis_id": "player_id"}),
        on=left_keys,
        how="left",
    )

    # Insert full_name and position right after 'player'
    cols = merged.columns.tolist()
    if "player" not in cols:
        raise ValueError("joined_games is missing 'player' column for placement")
    insert_after = cols.index("player") + 1

    new_cols_order = (
        cols[:insert_after]
        + [c for c in ["full_name", "position"] if c in merged.columns]
        + [c for c in cols if c not in ["full_name", "position"]][insert_after - 0:]
    )

    merged = merged[new_cols_order]

    merged.to_parquet(out_path, index=False)
    print(f"Wrote {out_path} with {len(merged):,} rows and {merged.shape[1]} columns")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--joined", required=True, help="Path to joined_games.parquet")
    ap.add_argument("--roster", required=True, help="Path to rosters_2019_2023_dedup.parquet")
    ap.add_argument("--out", required=True, help="Path to write full_games.parquet")
    args = ap.parse_args()
    main(args.joined, args.roster, args.out)
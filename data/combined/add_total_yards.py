# add_total_yards.py
import argparse
import pandas as pd
from pathlib import Path

YARD_COLS = ["pass_yards", "rush_yards", "rec_yards"]

def main():
    ap = argparse.ArgumentParser(description="Add total_yards to a full games parquet.")
    ap.add_argument("--inp", required=True, help="Input parquet (e.g., full_games_with_tds_ints_fumbles.parquet)")
    ap.add_argument("--out", required=True, help="Output parquet path")
    args = ap.parse_args()

    df = pd.read_parquet(args.inp)

    # Ensure the 3 yard columns exist, are numeric, and NaNs -> 0
    for c in YARD_COLS:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # total_yards = pass_yards + rush_yards + rec_yards
    df["total_yards"] = (
        df["pass_yards"].astype("int64")
      + df["rush_yards"].astype("int64")
      + df["rec_yards"].astype("int64")
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"Wrote {args.out} with shape {df.shape}")

if __name__ == "__main__":
    main()

import argparse
import logging

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("patch_weather")


def main():
    ap = argparse.ArgumentParser(description="Patch dome temps and precip flags in player_games_with_weather_3h.")
    ap.add_argument("--in", dest="inp", required=True, help="Input parquet (player_games_with_weather_3h.parquet)")
    ap.add_argument("--out", dest="out", required=True, help="Output parquet path (patched)")
    args = ap.parse_args()

    log.info("Loading input parquet: %s", args.inp)
    df = pd.read_parquet(args.inp)

    # Sanity check for required columns
    required_cols = ["is_dome", "temp_effective", "precip_3h_mm"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input parquet is missing required columns: {missing}")

    # 1) If is_dome is true, set temp_effective = 70°F
    log.info("Forcing temp_effective = 70°F for dome games (is_dome == 1).")
    dome_mask = df["is_dome"].astype(bool)
    df.loc[dome_mask, "temp_effective"] = 70.0

    # 2) Create precip_available flag BEFORE filling nulls
    log.info("Creating precip_available flag (1 = had precip value, 0 = was null).")
    df["precip_available"] = df["precip_3h_mm"].notna().astype("int8")

    # 3) Fill precip_3h_mm nulls with 0.0
    log.info("Filling NaN precip_3h_mm with 0.0.")
    df["precip_3h_mm"] = (
        pd.to_numeric(df["precip_3h_mm"], errors="coerce")
          .fillna(0.0)
          .astype("float32")
    )

    log.info("Writing patched parquet to: %s", args.out)
    df.to_parquet(args.out, index=False)
    log.info("Done.")


if __name__ == "__main__":
    main()

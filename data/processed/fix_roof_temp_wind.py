import argparse
import pandas as pd
import numpy as np


def normalize_roof(roof_value, is_dome_value):
    """
    Normalize roof to one of: 'dome', 'closed', 'open', 'outdoors'.
    Uses both the existing roof string and is_dome flag as hints.
    """
    val = "" if pd.isna(roof_value) else str(roof_value).strip().lower()

    # Common synonyms / messy values
    if val in {"dome", "domed", "indoor", "indoors"}:
        return "dome"
    if val in {"closed", "closed roof", "retractable", "retractable roof"}:
        return "closed"
    if val in {"open", "open roof", "partial open"}:
        return "open"
    if val in {"outdoor", "outdoors", "outside"}:
        return "outdoors"

    # Fall back to is_dome flag if available
    if is_dome_value in (1, True, "1", "true", "True"):
        return "dome"

    # Ultimate fallback – assume outdoors
    return "outdoors"


def main(input_path, output_path):
    print(f"Loading {input_path} ...")
    df = pd.read_parquet(input_path)

    # Get is_dome column if present (may not exist in some versions)
    is_dome_series = df["is_dome"] if "is_dome" in df.columns else pd.Series([None] * len(df))

    # Normalize roof
    print("Normalizing roof values ...")
    df["roof"] = [
        normalize_roof(r, d) for r, d in zip(df.get("roof"), is_dome_series)
    ]

    # Indoor vs outdoor masks
    indoor_mask = df["roof"].isin(["dome", "closed"])
    outdoor_mask = df["roof"].isin(["open", "outdoors"])

    # For dome/closed: override temp & wind
    print("Setting temp/wind for dome/closed ...")
    df.loc[indoor_mask, "temp"] = 70
    df.loc[indoor_mask, "wind"] = 0

    # For open/outdoors:
    # Keep existing temp/wind. If they are null but temp_clip/wind_clip exist,
    # optionally fill from those.
    if "temp_clip" in df.columns:
        needs_temp_fill = outdoor_mask & df["temp"].isna()
        df.loc[needs_temp_fill, "temp"] = df.loc[needs_temp_fill, "temp_clip"]

    if "wind_clip" in df.columns:
        needs_wind_fill = outdoor_mask & df["wind"].isna()
        df.loc[needs_wind_fill, "wind"] = df.loc[needs_wind_fill, "wind_clip"]

    # Drop unwanted columns
    cols_to_drop = [
        "temp_clip",
        "wind_clip",
        "is_dome",
        "is_turf",
        "game_stadium",  # always null per your note
    ]
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]

    print(f"Dropping columns: {cols_to_drop}")
    df = df.drop(columns=cols_to_drop)

    # Write out new parquet
    print(f"Writing cleaned parquet to {output_path} ...")
    df.to_parquet(output_path, index=False)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fix roof/temp/wind and drop helper columns "
                    "for player_games_with_conditionswith_roof parquet."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="player_games_with_conditionswith_roof.parquet",
        help="Path to input parquet (with current roof/temp/wind columns).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="player_games_with_conditions_with_roof_fixed.parquet",
        help="Path to output parquet with cleaned roof/temp/wind.",
    )

    args = parser.parse_args()
    main(args.input, args.output)

import pandas as pd

# ---- File paths (edit these if your files live elsewhere) ----
FULL_GAMES_PATH = "full_games_with_tds_ints_fumbles_yards.parquet"
SCHEDULE_PATH = "games_schedule.parquet"
OUTPUT_PATH = "full_games_with_tds_ints_fumbles_yards_with_weather.parquet"


def main():
    # Load parquet files
    print("Loading Parquet files...")
    full_games = pd.read_parquet(FULL_GAMES_PATH)
    schedule = pd.read_parquet(SCHEDULE_PATH)

    # Keep only the columns we care about from games_schedule
    schedule_subset = schedule[[
        "game_id",
        "roof",
        "surface",
        "temp",
        "wind",
        "stadium",
    ]]

    # Left join: all rows from full_games, matching weather/stadium by game_id
    print("Merging data on game_id...")
    merged = full_games.merge(schedule_subset, on="game_id", how="left")

    # Normalize roof text just in case (e.g., 'Closed', 'DOME', etc.)
    merged["roof"] = merged["roof"].str.lower()

    # If roof is closed, dome, or open AND temp/wind is null, set temp=70, wind=0
    mask_roof_indoor_like = merged["roof"].isin(["closed", "dome", "open"])

    merged.loc[mask_roof_indoor_like & merged["temp"].isna(), "temp"] = 70
    merged.loc[mask_roof_indoor_like & merged["wind"].isna(), "wind"] = 0

    # Save out to a new parquet file
    print("Saving merged table with weather to Parquet...")
    merged.to_parquet(OUTPUT_PATH, index=False)

    print("Done!")
    print(f"Output file: {OUTPUT_PATH}")
    print(f"Rows: {merged.shape[0]}, Columns: {merged.shape[1]}")


if __name__ == "__main__":
    main()

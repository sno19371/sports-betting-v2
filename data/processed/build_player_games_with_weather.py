import pandas as pd
import numpy as np


# ---------- helpers ----------

def normalize_roof(roof_value):
    """Map raw roof strings into: dome, closed, open, outdoors."""
    if pd.isna(roof_value):
        return None
    val = str(roof_value).strip().lower()

    # Dome / indoor
    if val in {
        "dome", "domed", "indoor", "indoors",
        "fixed-dome", "fixed dome"
    }:
        return "dome"

    # Closed / retractable but closed
    if val in {
        "closed", "closed roof", "retractable (closed)",
        "retractable_closed", "retractable-closed"
    }:
        return "closed"

    # Open / retractable but open
    if val in {
        "open", "open roof", "retractable (open)",
        "retractable_open", "retractable-open", "partial open"
    }:
        return "open"

    # Outdoors / outside
    if val in {
        "outdoor", "outdoors", "outside"
    }:
        return "outdoors"

    # If it's some weird value, treat as outdoors by default
    return "outdoors"


def first_non_null(series):
    """Return first non-null value in a Series, else NaN."""
    for v in series:
        if pd.notna(v):
            return v
    return np.nan


# ---------- main pipeline ----------

def main():
    # --- paths: change these if needed ---
    pbp_path = "pbp_combined.parquet"
    player_games_in = "player_games_with_conditions_with_roof.parquet"
    player_games_out = "player_games_with_conditions_weather.parquet"

    # --- 1. Load pbp_combined with only needed columns ---
    print(f"Loading PBP from {pbp_path} ...")
    pbp = pd.read_parquet(
        pbp_path,
        columns=["game_id", "roof", "temp", "wind"]
    )

    # --- 2. Normalize roof ---
    print("Normalizing PBP roof values ...")
    pbp["roof_norm"] = pbp["roof"].apply(normalize_roof)

    # --- 3. Build per-game weather: first non-null temp/wind per game ---
    print("Building per-game weather from PBP ...")
    game_weather = (
        pbp
        .groupby("game_id", as_index=False)
        .agg(
            roof=("roof_norm", first_non_null),
            temp=("temp", first_non_null),
            wind=("wind", first_non_null),
        )
    )

    # --- 4. Apply 70/0 override for dome/closed ---
    indoor_mask = game_weather["roof"].isin(["dome", "closed"])
    outdoor_mask = game_weather["roof"].isin(["open", "outdoors"])

    print("Applying indoor overrides (temp=70, wind=0 for dome/closed) ...")
    game_weather.loc[indoor_mask, "temp"] = 70
    game_weather.loc[indoor_mask, "wind"] = 0

    # Note: for open/outdoors we just keep the first non-null temp/wind we found.
    # If a game has no weather info at all in PBP, temp/wind will stay NaN.

    # --- 5. Load player_games parquet ---
    print(f"Loading player games from {player_games_in} ...")
    pg = pd.read_parquet(player_games_in)

    # --- 6. Drop old roof/weather helper columns ---
    cols_to_drop = [
        "roof",
        "temp",
        "wind",
        "temp_clip",
        "wind_clip",
        "is_dome",
        "is_turf",
        "game_stadium",
    ]
    cols_to_drop = [c for c in cols_to_drop if c in pg.columns]
    if cols_to_drop:
        print(f"Dropping old columns from player_games: {cols_to_drop}")
        pg = pg.drop(columns=cols_to_drop)

    # --- 7. Join per-game weather onto player_games ---
    print("Joining per-game weather onto player games ...")
    pg = pg.merge(game_weather, on="game_id", how="left")

    # At this point pg has roof/temp/wind columns that are:
    #  - normalized roof
    #  - 70/0 for dome/closed
    #  - first non-null temp/wind from pbp_combined for open/outdoors

    # --- 8. Save to new parquet ---
    print(f"Writing updated player games to {player_games_out} ...")
    pg.to_parquet(player_games_out, index=False)
    print("Done.")


if __name__ == "__main__":
    main()

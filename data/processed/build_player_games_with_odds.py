#!/usr/bin/env python3
"""
build_player_games_with_odds.py

Combined pipeline that creates player_games_with_odds_flexed_fixed.parquet from:
  - full_games_with_tds_ints_fumbles_yards.parquet (player-per-game data)
  - pbp_combined.parquet (play-by-play for kickoff times)
  - locations.csv (stadium lat/lon, roof, surface)
  - nfl_odds_historical.csv (betting lines)
  - openmeteo_cache_3h.json (optional cache for weather API)

Pipeline steps:
  1. Join OpenMeteo weather data (temp, wind, 3h precip) to player games
  2. Patch dome temperatures (set to 70°F) and create precip_available flag
  3. Merge odds with fuzzy date matching (+/- 1 day for flexed games)

Usage:
  python build_player_games_with_odds.py

Or with custom paths:
  python build_player_games_with_odds.py \
    --players full_games_with_tds_ints_fumbles_yards.parquet \
    --pbp pbp_combined.parquet \
    --locations locations.csv \
    --odds nfl_odds_historical.csv \
    --cache openmeteo_cache_3h.json \
    --out player_games_with_odds_flexed_fixed.parquet
"""

import argparse
import json
import logging
import os
import time
from collections import Counter
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
from dateutil import parser as dtparse

# ---------- Logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("build_player_games")

# ---------- Constants ----------
HOURLY_FIELDS = ["temperature_2m", "wind_speed_10m", "precipitation"]

TEAM_NORM = {
    "JAC": "JAX",
    "WSH": "WAS",
    "STL": "LAR",
    "LA": "LAR",
    "SD": "LAC",
    "OAK": "LV",
}

TEAM_FULL_TO_ABBR = {
    'Arizona Cardinals': 'ARI', 'Atlanta Falcons': 'ATL', 'Baltimore Ravens': 'BAL',
    'Buffalo Bills': 'BUF', 'Carolina Panthers': 'CAR', 'Chicago Bears': 'CHI',
    'Cincinnati Bengals': 'CIN', 'Cleveland Browns': 'CLE', 'Dallas Cowboys': 'DAL',
    'Denver Broncos': 'DEN', 'Detroit Lions': 'DET', 'Green Bay Packers': 'GB',
    'Houston Texans': 'HOU', 'Indianapolis Colts': 'IND', 'Jacksonville Jaguars': 'JAX',
    'Kansas City Chiefs': 'KC', 'Las Vegas Raiders': 'LV', 'Oakland Raiders': 'LV',
    'Los Angeles Chargers': 'LAC', 'San Diego Chargers': 'LAC',
    'Los Angeles Rams': 'LAR', 'St. Louis Rams': 'LAR',
    'Miami Dolphins': 'MIA', 'Minnesota Vikings': 'MIN', 'New England Patriots': 'NE',
    'New Orleans Saints': 'NO', 'New York Giants': 'NYG', 'New York Jets': 'NYJ',
    'Philadelphia Eagles': 'PHI', 'Pittsburgh Steelers': 'PIT', 'San Francisco 49ers': 'SF',
    'Seattle Seahawks': 'SEA', 'Tampa Bay Buccaneers': 'TB', 'Tennessee Titans': 'TEN',
    'Washington Commanders': 'WAS', 'Washington Football Team': 'WAS', 'Washington Redskins': 'WAS'
}


# ============================================================================
# STEP 1: Weather from OpenMeteo
# ============================================================================

def norm_team(x: str) -> str:
    """Normalize team abbreviations to canonical form."""
    s = (str(x) if x is not None else "").strip().upper()
    return TEAM_NORM.get(s, s)


def roof_from_csv(csv_roof: str) -> str:
    """Map locations.csv roof values to: dome / closed / open / outdoors."""
    s = (str(csv_roof) if csv_roof is not None else "").strip().lower()
    if s in {"yes", "dome", "indoor"}:
        return "dome"
    if s == "retractable":
        return "closed"
    if s in {"no", "outdoor", "outdoors"}:
        return "outdoors"
    return "outdoors"


def surface_from_csv(s: str) -> str:
    """Normalize surface type."""
    s = (str(s) if s is not None else "").strip().lower()
    if "turf" in s or "artificial" in s:
        return "turf"
    if "grass" in s or "hybrid" in s or "natural" in s:
        return "grass"
    return "unknown"


def parse_kick_local(game_date: str, time_of_day: str) -> datetime:
    """Build a naive 'local' datetime from game_date + time_of_day."""
    try:
        d = dtparse.parse(str(game_date)).date()
    except Exception:
        d = pd.to_datetime(str(game_date), errors="coerce").date()
    if pd.isna(d):
        raise ValueError(f"Unparseable game_date: {game_date}")

    if time_of_day and str(time_of_day).strip():
        try:
            t = dtparse.parse(str(time_of_day)).time()
        except Exception:
            tok = str(time_of_day).split(",")[-1].strip()
            try:
                t = dtparse.parse(tok).time()
            except Exception:
                t = datetime.strptime("13:00:00", "%H:%M:%S").time()
    else:
        t = datetime.strptime("13:00:00", "%H:%M:%S").time()

    return datetime.combine(d, t)


def round_to_nearest_hour(dt: datetime) -> datetime:
    """Round naive datetime to nearest hour (ties up)."""
    if dt.minute >= 30:
        dt = dt + timedelta(hours=1)
    return dt.replace(minute=0, second=0, microsecond=0)


def fetch_hourly_one_day(lat: float, lon: float, date_str: str, session: requests.Session) -> dict:
    """Call Open-Meteo ERA5 for one day's hourly data at given lat/lon."""
    base = "https://archive-api.open-meteo.com/v1/era5"
    params = {
        "latitude": f"{lat:.6f}",
        "longitude": f"{lon:.6f}",
        "start_date": date_str,
        "end_date": date_str,
        "hourly": ",".join(HOURLY_FIELDS),
        "timezone": "auto",
        "timeformat": "iso8601",
    }
    for attempt in range(4):
        try:
            r = session.get(base, params=params, timeout=30)
            if r.status_code == 429:
                time.sleep(1.0 + attempt)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            log.warning("Open-Meteo fetch failed (%s). Retry %d", e, attempt + 1)
            time.sleep(1.0 + attempt)
    raise RuntimeError(f"Failed to fetch Open-Meteo for {lat},{lon} {date_str}")


def extract_hour_and_3h(json_obj: dict, local_hour: datetime) -> dict:
    """Extract temp, wind, and 3-hour precipitation from OpenMeteo response."""
    hourly = json_obj.get("hourly", {})
    times = hourly.get("time", [])
    if not times:
        return {"temp_c": np.nan, "wind_ms": np.nan, "precip_mm": np.nan, "precip_3h_mm": np.nan}

    time_index = pd.to_datetime(times)
    t_local = pd.to_datetime(local_hour)

    diffs = np.abs((time_index - t_local).total_seconds())
    i0 = int(diffs.argmin())

    def arr(field):
        vals = hourly.get(field, [])
        if not vals:
            return np.full(len(time_index), np.nan, dtype=float)
        return np.array(pd.to_numeric(vals, errors="coerce"), dtype=float)

    temp_arr = arr("temperature_2m")
    wind_arr = arr("wind_speed_10m")
    precip_arr = arr("precipitation")

    temp_c = temp_arr[i0] if not np.isnan(temp_arr[i0]) else np.nan
    wind_ms = wind_arr[i0] if not np.isnan(wind_arr[i0]) else np.nan
    precip_mm = precip_arr[i0] if not np.isnan(precip_arr[i0]) else np.nan

    t_end = t_local + timedelta(hours=3)
    mask = (time_index >= t_local) & (time_index < t_end)
    precip_3h = float(np.nansum(precip_arr[mask])) if mask.any() else np.nan

    return {
        "temp_c": temp_c,
        "wind_ms": wind_ms,
        "precip_mm": precip_mm,
        "precip_3h_mm": precip_3h,
    }


def pick_mode(series: pd.Series):
    """Return the most common non-null value."""
    vals = [v for v in series.dropna().tolist() if str(v).strip() != ""]
    if not vals:
        return None
    return Counter(vals).most_common(1)[0][0]


def step1_join_weather(
    players_path: str,
    pbp_path: str,
    locations_path: str,
    cache_path: str | None = None,
) -> pd.DataFrame:
    """
    Step 1: Join OpenMeteo weather data to player-per-game table.
    
    Returns DataFrame with weather columns added.
    """
    log.info("=" * 60)
    log.info("STEP 1: Joining OpenMeteo weather data")
    log.info("=" * 60)

    # Load locations (stadium metadata)
    log.info("Loading locations: %s", locations_path)
    loc = pd.read_csv(locations_path)
    loc["homeTeamAbbr"] = loc["homeTeamAbbr"].map(norm_team)
    loc = loc.rename(columns={"homeTeamAbbr": "home_team", "long": "lon"})
    loc["lat"] = pd.to_numeric(loc["lat"], errors="coerce")
    loc["lon"] = pd.to_numeric(loc["lon"], errors="coerce")
    loc["roof"] = loc["roof"].apply(roof_from_csv)
    loc["surface"] = loc["surface"].apply(surface_from_csv)
    log.info("Locations loaded for %d unique home teams.", loc["home_team"].nunique())

    # Load PBP for kickoff times
    log.info("Loading PBP: %s", pbp_path)
    pbp = pd.read_parquet(pbp_path)
    need_cols = ["season", "week", "home_team", "away_team", "game_date", "time_of_day"]
    for c in need_cols:
        if c not in pbp.columns:
            raise ValueError(f"PBP missing required column: {c}")

    games = pbp[need_cols].copy()
    games["home_team"] = games["home_team"].map(norm_team)
    games["away_team"] = games["away_team"].map(norm_team)

    games = (
        games.groupby(["season", "week", "home_team", "away_team", "game_date"], dropna=False)
        .agg({"time_of_day": pick_mode})
        .reset_index()
    )

    # Attach stadium lat/lon + roof/surface
    games = games.merge(
        loc[["home_team", "lat", "lon", "roof", "surface"]],
        on="home_team",
        how="left",
    )

    missing_geo = games["lat"].isna().sum()
    if missing_geo:
        log.warning("WARNING: %d games missing lat/lon after location join.", missing_geo)

    # Compute kickoff local time (rounded to hour)
    games["kick_local"] = games.apply(
        lambda r: round_to_nearest_hour(parse_kick_local(r["game_date"], r["time_of_day"])),
        axis=1,
    )
    games["game_date_str"] = pd.to_datetime(games["game_date"]).dt.strftime("%Y-%m-%d")

    # Fetch OpenMeteo weather
    cache = {}
    if cache_path and os.path.exists(cache_path):
        try:
            cache = json.load(open(cache_path, "r"))
        except Exception:
            cache = {}

    session = requests.Session()
    weather_rows = []

    keys = games[["home_team", "game_date_str", "lat", "lon"]].drop_duplicates()
    log.info("Unique (home_team, game_date) keys to fetch: %d", len(keys))

    for _, row in keys.iterrows():
        ht = row["home_team"]
        gd = row["game_date_str"]
        lat = float(row["lat"]) if not pd.isna(row["lat"]) else None
        lon = float(row["lon"]) if not pd.isna(row["lon"]) else None

        if lat is None or lon is None:
            weather_rows.append({"home_team": ht, "game_date_str": gd, "wx_json": None})
            continue

        cache_key = f"{ht}:{gd}:{lat:.4f},{lon:.4f}"
        if cache_key in cache:
            wx_json = cache[cache_key]
        else:
            wx_json = fetch_hourly_one_day(lat, lon, gd, session)
            cache[cache_key] = wx_json
            time.sleep(0.15)  # rate limit

        weather_rows.append({"home_team": ht, "game_date_str": gd, "wx_json": wx_json})

    if cache_path:
        try:
            json.dump(cache, open(cache_path, "w"))
        except Exception as e:
            log.warning("Could not write cache: %s", e)

    wx_df = pd.DataFrame(weather_rows)
    games = games.merge(wx_df, on=["home_team", "game_date_str"], how="left")

    def pick_hour_features(row):
        if not isinstance(row["wx_json"], dict) or row["wx_json"] is None:
            return {"temp_c": np.nan, "wind_ms": np.nan, "precip_mm": np.nan, "precip_3h_mm": np.nan}
        return extract_hour_and_3h(row["wx_json"], row["kick_local"])

    hour_df = games.apply(pick_hour_features, axis=1).apply(pd.Series)
    games = pd.concat([games, hour_df], axis=1)

    # Build model-ready features
    games["is_dome"] = (games["roof"] == "dome").astype("int8")

    # temp: C -> F
    temp_c = pd.to_numeric(games["temp_c"], errors="coerce")
    temp_f = temp_c * 9.0 / 5.0 + 32.0
    temp_eff = temp_f.astype("float32")
    temp_eff = temp_eff.where(~games["is_dome"].eq(1) | temp_eff.notna(), 70.0)
    temp_eff = temp_eff.fillna(70.0).astype("float32")
    games["temp_effective"] = temp_eff

    # wind: m/s -> mph
    wind_ms = pd.to_numeric(games["wind_ms"], errors="coerce")
    wind_mph = wind_ms * 2.236936
    wind_eff = np.where(games["is_dome"].eq(1), 0.0, wind_mph)
    wind_eff = np.nan_to_num(wind_eff, nan=0.0).astype("float32")
    games["wind_effective"] = wind_eff

    games["precip_3h_mm"] = pd.to_numeric(games["precip_3h_mm"], errors="coerce").astype("float32")

    # Keep only what we need per game
    env_cols = [
        "season", "week", "home_team", "away_team", "game_date",
        "roof", "surface", "is_dome", "temp_effective", "wind_effective", "precip_3h_mm",
    ]
    games_env = games[env_cols].drop_duplicates()
    log.info("Built env rows for %d games.", len(games_env))

    # Join onto player-per-game table
    log.info("Loading players table: %s", players_path)
    players = pd.read_parquet(players_path)
    for c in ["season", "week", "home_team", "away_team", "game_date"]:
        if c not in players.columns:
            raise ValueError(f"Players parquet missing required column '{c}'")

    players["home_team"] = players["home_team"].map(norm_team)
    players["away_team"] = players["away_team"].map(norm_team)

    merged = players.merge(
        games_env,
        on=["season", "week", "home_team", "away_team", "game_date"],
        how="left",
        suffixes=("", "_env"),
    )

    total = len(merged)
    have_temp = merged["temp_effective"].notna().sum()
    log.info(
        "Weather join coverage: %.2f%% rows with env features (%d / %d)",
        100.0 * have_temp / total if total else 0.0,
        have_temp,
        total,
    )

    return merged


# ============================================================================
# STEP 2: Patch dome temperatures and precip flags
# ============================================================================

def step2_patch_weather(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 2: Patch dome temperatures and create precip_available flag.
    
    - If is_dome == 1, set temp_effective = 70°F
    - Create precip_available flag (1 = had precip value, 0 = was null)
    - Fill NaN precip_3h_mm with 0.0
    """
    log.info("=" * 60)
    log.info("STEP 2: Patching dome temperatures and precip flags")
    log.info("=" * 60)

    required_cols = ["is_dome", "temp_effective", "precip_3h_mm"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    # Force temp_effective = 70°F for dome games
    log.info("Forcing temp_effective = 70°F for dome games (is_dome == 1).")
    dome_mask = df["is_dome"].astype(bool)
    df.loc[dome_mask, "temp_effective"] = 70.0

    # Create precip_available flag BEFORE filling nulls
    log.info("Creating precip_available flag (1 = had precip value, 0 = was null).")
    df["precip_available"] = df["precip_3h_mm"].notna().astype("int8")

    # Fill precip_3h_mm nulls with 0.0
    log.info("Filling NaN precip_3h_mm with 0.0.")
    df["precip_3h_mm"] = (
        pd.to_numeric(df["precip_3h_mm"], errors="coerce")
        .fillna(0.0)
        .astype("float32")
    )

    return df


# ============================================================================
# STEP 3: Merge odds with fuzzy date matching
# ============================================================================

def step3_merge_odds(df: pd.DataFrame, odds_csv_path: str) -> pd.DataFrame:
    """
    Step 3: Merge betting odds with fuzzy date matching.
    
    Uses exact date match first, then tries +/- 1 day for flexed games.
    """
    log.info("=" * 60)
    log.info("STEP 3: Merging odds with fuzzy date matching")
    log.info("=" * 60)

    # Process odds data
    log.info("Loading Odds CSV from: %s", odds_csv_path)
    df_odds = pd.read_csv(odds_csv_path)

    # Clean headers
    df_odds.columns = df_odds.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('?', '')

    # Fix date
    log.info("Parsing Odds dates...")
    df_odds['join_date'] = pd.to_datetime(df_odds['date'], format='mixed').dt.date

    # Map teams
    df_odds['join_home'] = df_odds['home_team'].map(TEAM_FULL_TO_ABBR)
    df_odds['join_away'] = df_odds['away_team'].map(TEAM_FULL_TO_ABBR)

    # Filter columns
    cols_to_keep = ['join_date', 'join_home', 'join_away', 'home_line_close', 'away_line_close', 'total_score_open']
    df_odds = df_odds[cols_to_keep].copy()

    # Process player data join keys
    df['join_date'] = pd.to_datetime(df['game_date'], errors='coerce').dt.date
    df['join_home'] = df['home_team']
    df['join_away'] = df['away_team']

    log.info("Player Rows: %d | Odds Rows: %d", len(df), len(df_odds))
    log.info("Starting Fuzzy Merge (Exact Match -> Then +/- 1 Day)...")

    # STEP 3a: Exact Date Match
    merged_df = pd.merge(
        df,
        df_odds,
        how='left',
        on=['join_date', 'join_home', 'join_away']
    )

    exact_match_count = merged_df['home_line_close'].notna().sum()
    log.info("   Exact Matches: %d", exact_match_count)

    # STEP 3b: Try Match with Date + 1 Day
    df_odds_plus = df_odds.copy()
    df_odds_plus['join_date'] = df_odds_plus['join_date'] + timedelta(days=1)

    merged_df = pd.merge(
        merged_df,
        df_odds_plus,
        how='left',
        on=['join_date', 'join_home', 'join_away'],
        suffixes=('', '_plus')
    )

    # Fill in missing values
    cols_to_fill = ['home_line_close', 'away_line_close', 'total_score_open']
    for col in cols_to_fill:
        merged_df[col] = merged_df[col].fillna(merged_df[f"{col}_plus"])

    # STEP 3c: Try Match with Date - 1 Day
    df_odds_minus = df_odds.copy()
    df_odds_minus['join_date'] = df_odds_minus['join_date'] - timedelta(days=1)

    merged_df = pd.merge(
        merged_df,
        df_odds_minus,
        how='left',
        on=['join_date', 'join_home', 'join_away'],
        suffixes=('', '_minus')
    )

    for col in cols_to_fill:
        merged_df[col] = merged_df[col].fillna(merged_df[f"{col}_minus"])

    # Cleanup
    cols_to_drop = [c for c in merged_df.columns if c.endswith('_plus') or c.endswith('_minus')]
    cols_to_drop.extend(['join_date', 'join_home', 'join_away'])
    merged_df.drop(columns=cols_to_drop, inplace=True)

    # Validation
    final_matched = merged_df['home_line_close'].notna().sum()
    log.info("✅ Final Matched Rows: %d / %d", final_matched, len(merged_df))

    return merged_df


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Build player_games_with_odds_flexed_fixed.parquet from raw data."
    )
    ap.add_argument(
        "--players",
        default="full_games_with_tds_ints_fumbles_yards.parquet",
        help="Path to player-per-game parquet"
    )
    ap.add_argument(
        "--pbp",
        default="pbp_combined.parquet",
        help="Path to PBP parquet (for kickoff times)"
    )
    ap.add_argument(
        "--locations",
        default="locations.csv",
        help="Path to locations.csv (stadium metadata)"
    )
    ap.add_argument(
        "--odds",
        default="nfl_odds_historical.csv",
        help="Path to odds CSV"
    )
    ap.add_argument(
        "--cache",
        default="openmeteo_cache_3h.json",
        help="Path to OpenMeteo cache JSON"
    )
    ap.add_argument(
        "--out",
        default="player_games_with_odds_flexed_fixed.parquet",
        help="Output parquet path"
    )
    args = ap.parse_args()

    log.info("=" * 60)
    log.info("BUILD PLAYER GAMES WITH ODDS PIPELINE")
    log.info("=" * 60)
    log.info("Inputs:")
    log.info("  Players:   %s", args.players)
    log.info("  PBP:       %s", args.pbp)
    log.info("  Locations: %s", args.locations)
    log.info("  Odds:      %s", args.odds)
    log.info("  Cache:     %s", args.cache)
    log.info("Output: %s", args.out)
    log.info("")

    # Step 1: Join weather
    df = step1_join_weather(
        players_path=args.players,
        pbp_path=args.pbp,
        locations_path=args.locations,
        cache_path=args.cache,
    )

    # Step 2: Patch weather
    df = step2_patch_weather(df)

    # Step 3: Merge odds
    df = step3_merge_odds(df, args.odds)

    # Save final output
    log.info("=" * 60)
    log.info("SAVING OUTPUT")
    log.info("=" * 60)
    log.info("Writing to: %s", args.out)
    df.to_parquet(args.out, index=False)
    log.info("✅ Done! %d rows written.", len(df))


if __name__ == "__main__":
    main()


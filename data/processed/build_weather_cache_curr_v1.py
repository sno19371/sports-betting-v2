#!/usr/bin/env python3
"""
build_weather_cache_curr.py

Fetches weather from Open-Meteo for all games in games_schedule.parquet
and saves to JSON cache. No player table required.

Usage:
  python build_weather_cache_curr.py

Or with custom paths:
  python build_weather_cache_curr.py \
    --schedule games_schedule.parquet \
    --locations locations.csv \
    --cache openmeteo_cache_3h_curr.json
"""

import argparse
import json
import logging
import os
import time
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import requests
from dateutil import parser as dtparse

# ---------- Logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("build_weather_cache")

# ---------- Config ----------
HOURLY_FIELDS = ["temperature_2m", "wind_speed_10m", "precipitation"]

TEAM_NORM = {
    "JAC": "JAX",
    "WSH": "WAS",
    "STL": "LAR",
    "LA": "LAR",
    "SD": "LAC",
    "OAK": "LV",
}


def norm_team(x):
    """Normalize team abbreviations."""
    s = (str(x) if x is not None else "").strip().upper()
    return TEAM_NORM.get(s, s)


def roof_from_csv(csv_roof):
    """Map locations.csv roof values to: dome / closed / open / outdoors."""
    s = (str(csv_roof) if csv_roof is not None else "").strip().lower()
    if s in {"yes", "dome", "indoor"}:
        return "dome"
    if s == "retractable":
        return "closed"
    if s in {"no", "outdoor", "outdoors"}:
        return "outdoors"
    return "outdoors"


def parse_gametime(gametime_str):
    """
    Parse gametime string (e.g., "1:00PM", "4:25PM") to a time object.
    Returns 1 PM if parsing fails.
    """
    if not gametime_str or pd.isna(gametime_str):
        return datetime.strptime("13:00:00", "%H:%M:%S").time()
    
    try:
        # Try common formats
        s = str(gametime_str).strip().upper()
        for fmt in ["%I:%M%p", "%I:%M %p", "%H:%M", "%H:%M:%S"]:
            try:
                return datetime.strptime(s, fmt).time()
            except ValueError:
                continue
        # Fallback: use dateutil
        return dtparse.parse(str(gametime_str)).time()
    except Exception:
        return datetime.strptime("13:00:00", "%H:%M:%S").time()


def round_to_nearest_hour(dt):
    """Round datetime to nearest hour."""
    if dt.minute >= 30:
        dt = dt + timedelta(hours=1)
    return dt.replace(minute=0, second=0, microsecond=0)


def fetch_hourly_one_day(lat, lon, date_str, session):
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
                log.warning("Rate limited, waiting...")
                time.sleep(2.0 + attempt)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            log.warning("Fetch failed (%s). Retry %d/4", e, attempt + 1)
            time.sleep(1.0 + attempt)
    raise RuntimeError(f"Failed to fetch Open-Meteo for {lat},{lon} {date_str}")


def main():
    ap = argparse.ArgumentParser(
        description="Build OpenMeteo weather cache from games_schedule.parquet"
    )
    ap.add_argument(
        "--schedule",
        default="games_schedule.parquet",
        help="Path to games_schedule.parquet"
    )
    ap.add_argument(
        "--locations",
        default="locations.csv",
        help="Path to locations.csv (homeTeamAbbr, roof, surface, lat, long)"
    )
    ap.add_argument(
        "--cache",
        default="openmeteo_cache_3h_curr.json",
        help="Output JSON cache path"
    )
    args = ap.parse_args()

    # ---------- Load Schedule ----------
    log.info("Loading schedule: %s", args.schedule)
    schedule = pd.read_parquet(args.schedule)
    
    # Check required columns
    required = ["game_id", "gameday", "home_team"]
    missing = [c for c in required if c not in schedule.columns]
    if missing:
        raise ValueError(f"Schedule missing required columns: {missing}")
    
    log.info("Schedule has %d total games", len(schedule))
    
    # Filter to seasons 2019-2025 only
    if "season" in schedule.columns:
        schedule = schedule[schedule["season"].between(2019, 2025)]
        log.info("Filtered to seasons 2019-2025: %d games", len(schedule))
    else:
        # Fall back to filtering by gameday year
        schedule["gameday"] = pd.to_datetime(schedule["gameday"])
        schedule = schedule[schedule["gameday"].dt.year.between(2019, 2025)]
        log.info("Filtered to years 2019-2025: %d games", len(schedule))
    
    # Check for gametime column
    gametime_col = None
    for col in ["gametime", "game_time", "kickoff", "time"]:
        if col in schedule.columns:
            gametime_col = col
            break
    
    if gametime_col:
        log.info("Found gametime column: %s", gametime_col)
    else:
        log.info("No gametime column found, will default to 1 PM")

    # ---------- Load Locations ----------
    log.info("Loading locations: %s", args.locations)
    loc = pd.read_csv(args.locations)
    loc["homeTeamAbbr"] = loc["homeTeamAbbr"].map(norm_team)
    loc = loc.rename(columns={"homeTeamAbbr": "home_team", "long": "lon"})
    loc["lat"] = pd.to_numeric(loc["lat"], errors="coerce")
    loc["lon"] = pd.to_numeric(loc["lon"], errors="coerce")
    
    log.info("Locations loaded for %d teams", loc["home_team"].nunique())

    # ---------- Build Games List ----------
    games = schedule[["game_id", "gameday", "home_team"]].copy()
    if gametime_col:
        games["gametime"] = schedule[gametime_col]
    else:
        games["gametime"] = None
    
    games["home_team"] = games["home_team"].map(norm_team)
    games["gameday"] = pd.to_datetime(games["gameday"])
    games["game_date_str"] = games["gameday"].dt.strftime("%Y-%m-%d")
    
    # Attach lat/lon from locations
    games = games.merge(
        loc[["home_team", "lat", "lon"]],
        on="home_team",
        how="left"
    )
    
    # Drop games with missing coordinates
    missing_geo = games["lat"].isna().sum()
    if missing_geo:
        log.warning("%d games missing lat/lon after location join", missing_geo)
        games = games[games["lat"].notna()]

    # Deduplicate by (home_team, game_date_str) for cache keys
    games = games.drop_duplicates(subset=["home_team", "game_date_str"])
    log.info("Unique (home_team, date) combinations: %d", len(games))

    # ---------- Filter Out Future/Recent Games ----------
    # ERA5 archive data has ~7 day lag, so skip games within last 7 days
    cutoff_date = pd.Timestamp(date.today()) - pd.Timedelta(days=7)
    future_games = (games["gameday"] > cutoff_date).sum()
    if future_games > 0:
        log.info("Skipping %d games after %s (ERA5 data not yet available)", 
                 future_games, cutoff_date.strftime("%Y-%m-%d"))
        games = games[games["gameday"] <= cutoff_date]
    log.info("Games with available ERA5 data: %d", len(games))

    # ---------- Load Existing Cache ----------
    cache = {}
    if os.path.exists(args.cache):
        try:
            with open(args.cache, "r") as f:
                cache = json.load(f)
            log.info("Loaded existing cache with %d entries", len(cache))
        except Exception as e:
            log.warning("Could not load existing cache: %s", e)
            cache = {}

    # ---------- Find What Needs Fetching ----------
    keys_to_fetch = []
    for _, row in games.iterrows():
        ht = row["home_team"]
        gd = row["game_date_str"]
        lat = float(row["lat"])
        lon = float(row["lon"])
        
        cache_key = f"{ht}:{gd}:{lat:.4f},{lon:.4f}"
        if cache_key not in cache:
            keys_to_fetch.append({
                "key": cache_key,
                "lat": lat,
                "lon": lon,
                "date": gd,
            })

    log.info("Already cached: %d", len(cache))
    log.info("Need to fetch: %d", len(keys_to_fetch))

    if not keys_to_fetch:
        log.info("Cache is up to date! Nothing to fetch.")
        return

    # ---------- Fetch Weather ----------
    session = requests.Session()
    fetched = 0
    errors = 0

    for i, item in enumerate(keys_to_fetch):
        key = item["key"]
        lat = item["lat"]
        lon = item["lon"]
        date_str = item["date"]

        log.info("Fetching %d/%d: %s", i + 1, len(keys_to_fetch), key)
        
        try:
            wx_json = fetch_hourly_one_day(lat, lon, date_str, session)
            cache[key] = wx_json
            fetched += 1
        except Exception as e:
            log.error("Failed to fetch %s: %s", key, e)
            errors += 1

        # Rate limit
        time.sleep(0.15)

        # Save checkpoint every 50 fetches
        if (i + 1) % 50 == 0:
            with open(args.cache, "w") as f:
                json.dump(cache, f)
            log.info("Checkpoint saved (%d entries)", len(cache))

    # ---------- Final Save ----------
    with open(args.cache, "w") as f:
        json.dump(cache, f)
    
    log.info("=" * 50)
    log.info("DONE!")
    log.info("  Fetched: %d", fetched)
    log.info("  Errors: %d", errors)
    log.info("  Total cache entries: %d", len(cache))
    log.info("  Saved to: %s", args.cache)


if __name__ == "__main__":
    main()


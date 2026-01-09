#!/usr/bin/env python3
"""
build_weather_cache_curr.py

Fetches weather from Open-Meteo for all games in games_schedule.parquet
and saves to JSON cache. No player table required.

For recent games where ERA5 data isn't available yet, falls back to
nfl_games_weather.json from the generate_bets_curr folder.

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
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dateutil import parser as dtparse

# Fallback weather file for recent games
SCRIPT_DIR = Path(__file__).parent
FALLBACK_WEATHER_PATH = SCRIPT_DIR.parent.parent / "scripts" / "generate_bets_curr" / "nfl_games_weather.json"

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


def load_fallback_weather():
    """Load fallback weather from nfl_games_weather.json."""
    if not FALLBACK_WEATHER_PATH.exists():
        log.warning("Fallback weather file not found: %s", FALLBACK_WEATHER_PATH)
        return {}
    
    try:
        with open(FALLBACK_WEATHER_PATH, "r") as f:
            data = json.load(f)
        
        # Build lookup by (home_team, gameday)
        fallback = {}
        for game in data.get("games", []):
            key = (game["home_team"], game["gameday"])
            fallback[key] = {
                "temp_f": game.get("temp_f"),
                "wind_mph": game.get("wind_mph"),
                "is_dome": game.get("is_dome", False),
                "source": game.get("source", "fallback"),
            }
        
        log.info("Loaded fallback weather for %d games from nfl_games_weather.json", len(fallback))
        return fallback
    except Exception as e:
        log.warning("Failed to load fallback weather: %s", e)
        return {}


def create_synthetic_cache_entry(temp_f, wind_mph, date_str, gametime_str="13:00"):
    """
    Create a synthetic cache entry that mimics the ERA5 format.
    This allows the downstream code to read it the same way.
    """
    # Convert F to C for temperature
    temp_c = (temp_f - 32) * 5 / 9 if temp_f is not None else 20.0
    
    # Convert mph to km/h for wind
    wind_kmh = wind_mph * 1.60934 if wind_mph is not None else 0.0
    
    # Create hourly arrays (24 hours)
    hours = [f"{date_str}T{h:02d}:00" for h in range(24)]
    temps = [temp_c] * 24
    winds = [wind_kmh] * 24
    precip = [0.0] * 24  # Assume no precipitation for fallback
    
    return {
        "hourly": {
            "time": hours,
            "temperature_2m": temps,
            "wind_speed_10m": winds,
            "precipitation": precip,
        },
        "source": "fallback_nfl_games_weather"
    }


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

    # ---------- Split Games: ERA5 Available vs Fallback ----------
    # ERA5 archive data has ~7 day lag
    cutoff_date = pd.Timestamp(date.today()) - pd.Timedelta(days=7)
    
    games_era5 = games[games["gameday"] <= cutoff_date].copy()
    games_fallback = games[games["gameday"] > cutoff_date].copy()
    
    log.info("Games with ERA5 data available: %d", len(games_era5))
    log.info("Games needing fallback weather: %d", len(games_fallback))
    
    # Load fallback weather data
    fallback_weather = load_fallback_weather() if len(games_fallback) > 0 else {}

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

    # ---------- Process Fallback Games First ----------
    fallback_added = 0
    fallback_missing = 0
    
    for _, row in games_fallback.iterrows():
        ht = row["home_team"]
        gd = row["game_date_str"]
        lat = float(row["lat"])
        lon = float(row["lon"])
        
        cache_key = f"{ht}:{gd}:{lat:.4f},{lon:.4f}"
        
        if cache_key in cache:
            continue  # Already have it
        
        # Look up in fallback weather
        fallback_key = (ht, gd)
        if fallback_key in fallback_weather:
            fb = fallback_weather[fallback_key]
            cache[cache_key] = create_synthetic_cache_entry(
                temp_f=fb["temp_f"],
                wind_mph=fb["wind_mph"],
                date_str=gd
            )
            fallback_added += 1
            log.info("Added fallback weather for %s on %s (source: %s)", ht, gd, fb.get("source", "unknown"))
        else:
            fallback_missing += 1
            log.warning("No fallback weather found for %s on %s", ht, gd)
    
    if fallback_added > 0 or fallback_missing > 0:
        log.info("Fallback weather: %d added, %d missing", fallback_added, fallback_missing)

    # ---------- Find What Needs Fetching from ERA5 ----------
    # Also upgrade any fallback entries that are now old enough for ERA5
    keys_to_fetch = []
    fallback_upgrades = 0
    
    for _, row in games_era5.iterrows():
        ht = row["home_team"]
        gd = row["game_date_str"]
        lat = float(row["lat"])
        lon = float(row["lon"])
        
        cache_key = f"{ht}:{gd}:{lat:.4f},{lon:.4f}"
        
        # Check if we need to fetch
        needs_fetch = False
        if cache_key not in cache:
            needs_fetch = True
        else:
            # Check if existing entry is fallback data that should be upgraded
            existing = cache[cache_key]
            if isinstance(existing, dict) and existing.get("source") == "fallback_nfl_games_weather":
                needs_fetch = True
                fallback_upgrades += 1
                log.info("Will upgrade fallback to ERA5: %s on %s", ht, gd)
        
        if needs_fetch:
            keys_to_fetch.append({
                "key": cache_key,
                "lat": lat,
                "lon": lon,
                "date": gd,
            })

    log.info("Already cached (real ERA5): %d", len(cache) - fallback_upgrades)
    log.info("Fallback entries to upgrade: %d", fallback_upgrades)
    log.info("Need to fetch from ERA5: %d", len(keys_to_fetch))

    if not keys_to_fetch and fallback_added == 0:
        log.info("Cache is up to date! Nothing to fetch.")
        return
    
    if not keys_to_fetch:
        # Only fallback was added, save and exit
        with open(args.cache, "w") as f:
            json.dump(cache, f)
        log.info("=" * 50)
        log.info("DONE! (fallback only)")
        log.info("  Fallback added: %d", fallback_added)
        log.info("  Total cache entries: %d", len(cache))
        log.info("  Saved to: %s", args.cache)
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
    log.info("  Fetched from ERA5: %d (includes %d upgrades from fallback)", fetched, fallback_upgrades)
    log.info("  Added from fallback: %d", fallback_added)
    log.info("  Errors: %d", errors)
    log.info("  Total cache entries: %d", len(cache))
    log.info("  Saved to: %s", args.cache)


if __name__ == "__main__":
    main()


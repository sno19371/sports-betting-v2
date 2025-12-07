#!/usr/bin/env python3
"""
build_game_weather_curr.py

Fetches weather (temp, wind) for NFL games using Open-Meteo API.
- Uses ERA5 archive for historical games (>7 days old)
- Uses forecast API for recent/upcoming games

Inputs:
  - locations.csv: Stadium coordinates (homeTeamAbbr, lat, long)
  - games_schedule.parquet: Game schedule with dates/times

Output:
  - nfl_games_weather.json: Weather data per game

Usage:
  python build_game_weather_curr.py
"""

import json
import os
import time
from datetime import datetime, timedelta, date
from pathlib import Path

import pandas as pd
import requests

# =========================
# CONFIGURATION
# =========================
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = Path(__file__).parent.parent.parent / "data" / "processed"

LOCATIONS_PATH = BASE_DIR / "locations.csv"
SCHEDULE_PATH = BASE_DIR / "games_schedule.parquet"
OUTPUT_PATH = SCRIPT_DIR / "nfl_games_weather.json"

# Open-Meteo APIs
ERA5_URL = "https://archive-api.open-meteo.com/v1/era5"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

HOURLY_FIELDS = ["temperature_2m", "wind_speed_10m"]

# Team normalization
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


def parse_gametime(gametime_str):
    """Parse gametime string to hour (24h format)."""
    if not gametime_str or pd.isna(gametime_str):
        return 13  # Default 1 PM
    
    try:
        s = str(gametime_str).strip()
        # Try parsing various formats
        for fmt in ["%H:%M:%S", "%H:%M", "%I:%M%p", "%I:%M %p"]:
            try:
                dt = datetime.strptime(s.upper(), fmt)
                return dt.hour
            except ValueError:
                continue
        return 13
    except Exception:
        return 13


def fetch_weather_era5(lat, lon, date_str, hour, session):
    """Fetch historical weather from ERA5 archive."""
    params = {
        "latitude": f"{lat:.6f}",
        "longitude": f"{lon:.6f}",
        "start_date": date_str,
        "end_date": date_str,
        "hourly": ",".join(HOURLY_FIELDS),
        "timezone": "auto",
    }
    
    for attempt in range(3):
        try:
            r = session.get(ERA5_URL, params=params, timeout=30)
            if r.status_code == 429:
                time.sleep(2.0 + attempt)
                continue
            r.raise_for_status()
            data = r.json()
            
            # Extract weather at game hour
            hourly = data.get("hourly", {})
            times = hourly.get("time", [])
            temps = hourly.get("temperature_2m", [])
            winds = hourly.get("wind_speed_10m", [])
            
            if times and temps and winds:
                # Find closest hour
                for i, t in enumerate(times):
                    if f"T{hour:02d}:" in t or t.endswith(f"T{hour:02d}:00"):
                        temp_c = temps[i] if i < len(temps) else None
                        wind_kmh = winds[i] if i < len(winds) else None
                        
                        # Convert to F and mph
                        temp_f = (temp_c * 9/5 + 32) if temp_c is not None else None
                        wind_mph = (wind_kmh * 0.621371) if wind_kmh is not None else None
                        
                        return {
                            "temp_f": round(temp_f, 1) if temp_f else None,
                            "wind_mph": round(wind_mph, 1) if wind_mph else None,
                            "source": "era5"
                        }
            return None
        except Exception as e:
            time.sleep(1.0 + attempt)
    return None


def fetch_weather_forecast(lat, lon, date_str, hour, session):
    """Fetch forecast weather from Open-Meteo forecast API."""
    params = {
        "latitude": f"{lat:.6f}",
        "longitude": f"{lon:.6f}",
        "hourly": ",".join(HOURLY_FIELDS),
        "timezone": "auto",
        "start_date": date_str,
        "end_date": date_str,
    }
    
    for attempt in range(3):
        try:
            r = session.get(FORECAST_URL, params=params, timeout=30)
            if r.status_code == 429:
                time.sleep(2.0 + attempt)
                continue
            r.raise_for_status()
            data = r.json()
            
            hourly = data.get("hourly", {})
            times = hourly.get("time", [])
            temps = hourly.get("temperature_2m", [])
            winds = hourly.get("wind_speed_10m", [])
            
            if times and temps and winds:
                for i, t in enumerate(times):
                    if f"T{hour:02d}:" in t or t.endswith(f"T{hour:02d}:00"):
                        temp_c = temps[i] if i < len(temps) else None
                        wind_kmh = winds[i] if i < len(winds) else None
                        
                        temp_f = (temp_c * 9/5 + 32) if temp_c is not None else None
                        wind_mph = (wind_kmh * 0.621371) if wind_kmh is not None else None
                        
                        return {
                            "temp_f": round(temp_f, 1) if temp_f else None,
                            "wind_mph": round(wind_mph, 1) if wind_mph else None,
                            "source": "forecast"
                        }
            return None
        except Exception as e:
            time.sleep(1.0 + attempt)
    return None


def get_current_nfl_week(schedule_df):
    """Determine the current NFL week based on today's date."""
    today = pd.Timestamp(date.today())
    
    # Get current season (assume season starts in September)
    current_year = today.year
    if today.month < 3:  # Jan-Feb is previous year's season
        current_season = current_year - 1
    elif today.month >= 9:  # Sep-Dec is current year's season
        current_season = current_year
    else:  # Mar-Aug is offseason, use previous season
        current_season = current_year - 1
    
    # Filter to current season
    season_games = schedule_df[schedule_df["season"] == current_season].copy()
    
    if season_games.empty:
        # Try next season if current is empty
        current_season = current_year
        season_games = schedule_df[schedule_df["season"] == current_season].copy()
    
    if season_games.empty:
        return current_season, 1
    
    season_games["gameday"] = pd.to_datetime(season_games["gameday"])
    
    # Find the week where games are closest to today
    for week in sorted(season_games["week"].unique()):
        week_games = season_games[season_games["week"] == week]
        week_start = week_games["gameday"].min()
        week_end = week_games["gameday"].max()
        
        # If today is within this week's games (±3 days buffer)
        if week_start - pd.Timedelta(days=3) <= today <= week_end + pd.Timedelta(days=3):
            return current_season, int(week)
        
        # If this week hasn't happened yet, it's the upcoming week
        if week_start > today:
            return current_season, int(week)
    
    # Default to last week of season
    return current_season, int(season_games["week"].max())


def main():
    print("=" * 60)
    print("BUILD GAME WEATHER")
    print("=" * 60)
    
    # 1. Load locations
    print(f"\n📍 Loading locations from: {LOCATIONS_PATH}")
    locations = pd.read_csv(LOCATIONS_PATH)
    locations["homeTeamAbbr"] = locations["homeTeamAbbr"].map(norm_team)
    locations = locations.rename(columns={"homeTeamAbbr": "home_team", "long": "lon"})
    
    # Create lookup dict
    loc_lookup = {}
    for _, row in locations.iterrows():
        loc_lookup[row["home_team"]] = {
            "lat": row["lat"],
            "lon": row["lon"],
            "roof": row["roof"]
        }
    
    print(f"   Loaded {len(loc_lookup)} stadium locations")
    
    # 2. Load schedule
    print(f"\n📅 Loading schedule from: {SCHEDULE_PATH}")
    schedule = pd.read_parquet(SCHEDULE_PATH)
    schedule["gameday"] = pd.to_datetime(schedule["gameday"])
    schedule["home_team"] = schedule["home_team"].map(norm_team)
    
    # Determine current week
    current_season, current_week = get_current_nfl_week(schedule)
    print(f"   Current NFL Week: Season {current_season}, Week {current_week}")
    
    # Filter to current season and nearby weeks
    current_games = schedule[
        (schedule["season"] == current_season) & 
        (schedule["week"].between(current_week - 1, current_week + 1))
    ].copy()
    
    print(f"   Games to fetch weather for: {len(current_games)}")
    
    # 3. Fetch weather for each game
    print("\n🌤️  Fetching weather data...")
    session = requests.Session()
    
    today = date.today()
    era5_cutoff = today - timedelta(days=7)  # ERA5 has ~7 day lag
    
    results = {}
    
    for idx, game in current_games.iterrows():
        game_id = game["game_id"]
        home_team = game["home_team"]
        away_team = game.get("away_team", "UNK")
        gameday = game["gameday"].date()
        gametime = game.get("gametime", "13:00")
        game_hour = parse_gametime(gametime)
        week = int(game["week"])
        
        # Get stadium location
        if home_team not in loc_lookup:
            print(f"   ⚠️  No location for {home_team}, skipping")
            continue
        
        loc = loc_lookup[home_team]
        lat, lon = loc["lat"], loc["lon"]
        roof = loc["roof"]
        
        date_str = gameday.strftime("%Y-%m-%d")
        
        # Check if dome/indoor
        is_dome = roof in ["yes", "retractable"]
        
        if is_dome:
            # Indoor games have controlled climate
            weather = {
                "temp_f": 70.0,
                "wind_mph": 0.0,
                "source": "dome"
            }
        elif gameday <= era5_cutoff:
            # Historical game - use ERA5
            weather = fetch_weather_era5(lat, lon, date_str, game_hour, session)
            time.sleep(0.1)
        else:
            # Recent/upcoming game - use forecast
            weather = fetch_weather_forecast(lat, lon, date_str, game_hour, session)
            time.sleep(0.1)
        
        if weather:
            results[game_id] = {
                "game_id": game_id,
                "home_team": home_team,
                "away_team": away_team,
                "season": int(game["season"]),
                "week": week,
                "gameday": date_str,
                "gametime": str(gametime),
                "temp_f": weather["temp_f"],
                "wind_mph": weather["wind_mph"],
                "is_dome": is_dome,
                "source": weather["source"]
            }
            print(f"   ✓ {away_team} @ {home_team} (Wk{week}): {weather['temp_f']}°F, {weather['wind_mph']} mph [{weather['source']}]")
        else:
            print(f"   ✗ {away_team} @ {home_team} (Wk{week}): Failed to fetch")
    
    # 4. Save results
    print(f"\n💾 Saving to: {OUTPUT_PATH}")
    
    # Convert to list sorted by week/game
    output = {
        "generated_at": datetime.now().isoformat(),
        "current_season": current_season,
        "current_week": current_week,
        "games": list(results.values())
    }
    
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=4)
    
    # 5. Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Games processed: {len(results)}")
    print(f"Output saved to: {OUTPUT_PATH}")
    
    # Show sample
    if results:
        print("\n--- Sample Output ---")
        for game in list(results.values())[:5]:
            dome = "🏟️" if game["is_dome"] else "🌤️"
            print(f"{dome} Wk{game['week']}: {game['away_team']} @ {game['home_team']} - {game['temp_f']}°F, {game['wind_mph']} mph")
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()


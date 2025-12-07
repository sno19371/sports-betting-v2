#!/usr/bin/env python3
"""
generate_player_model_rows_curr.py

Generates player model input rows for prediction by combining:
1. Players from odds_history (those with betting lines)
2. Last 6 weeks of historical data from rb_combined_curr_v2.parquet
3. Current week metadata from JSON files (weather, depth chart, odds)

Output: player_model_rows.json ready for the trained model

Usage:
  python generate_player_model_rows_curr.py
"""

import json
import os
import glob
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# =========================
# CONFIGURATION
# =========================
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = Path(__file__).parent.parent.parent / "data" / "processed"

# Input files
HISTORICAL_DATA_PATH = BASE_DIR / "rb_combined_curr_v2.parquet"
ODDS_HISTORY_DIR = SCRIPT_DIR / "odds_history"
WEATHER_JSON_PATH = SCRIPT_DIR / "nfl_games_weather.json"
DEPTH_CHART_JSON_PATH = SCRIPT_DIR / "nfl_rbs_depth_chart.json"

# Output
OUTPUT_PATH = SCRIPT_DIR / "player_model_rows.json"

# Model parameters (from train_rush_volume_model_v10_curr.py)
L = 6  # Window size (last 6 weeks)

# TCN Features (sequential history)
TCN_COLS = [
    "rb_carries", "rb_ed_carry_share_all", "rb_rz_carries",
    "rb_rush_yards", "rb_stuffed_rate",
    "starter_qb_scramble_rate_db",
    "implied_game_script", "team_spread", "is_home"
]

# Meta Features (current week context)
META_COLS = [
    "season", "week", "posteam", "opponent", "player_key", "full_name",
    "implied_game_script", "team_spread", "is_home", "wind_effective",
    "opponent_rush_epa_allowed", "opponent_rush_yards_per_game_allowed",
    "depth_rank", "is_projected_starter", "is_active",
    "last_week_carries", "last_week_snap_share"
]

# Team name mapping (full name -> abbreviation)
TEAM_NAME_MAP = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LAR", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS"
}

TEAM_NORM = {
    "JAC": "JAX", "WSH": "WAS", "STL": "LAR", "LA": "LAR", "SD": "LAC", "OAK": "LV"
}


def norm_team(x):
    """Normalize team abbreviations."""
    s = (str(x) if x is not None else "").strip().upper()
    return TEAM_NORM.get(s, s)


def normalize_name(name):
    """Normalize player name for matching."""
    if not name:
        return ""
    name = str(name).strip()
    for suffix in [" Jr.", " Sr.", " III", " II", " IV", " V"]:
        name = name.replace(suffix, "")
    return name.lower().strip()


def to_initial_format(full_name):
    """
    Convert full name to initial format used in parquet.
    'Kyren Williams' -> 'K.Williams'
    'Mark Ingram II' -> 'M.Ingram II'
    'De'Von Achane' -> 'D.Achane'
    """
    if not full_name:
        return ""
    
    name = str(full_name).strip()
    
    # Extract suffix if present
    suffix = ""
    for suf in [" Jr.", " Jr", " Sr.", " Sr", " III", " II", " IV", " V"]:
        if name.endswith(suf):
            suffix = suf.replace(" Jr", " Jr.").replace(" Sr", " Sr.")
            name = name[:-len(suf)].strip()
            break
    
    # Split into parts
    parts = name.split()
    if len(parts) < 2:
        return name + suffix
    
    # Get first initial (handle De'Von, D'Andre, etc.)
    first = parts[0]
    first_initial = first[0].upper()
    
    # Get last name (could be multi-part like "St. Brown")
    last_name = parts[-1]
    
    # Build result: "K.Williams" or "M.Ingram II"
    result = f"{first_initial}.{last_name}"
    if suffix:
        result += suffix
    
    return result


def names_match(name1, name2):
    """Check if two names match (fuzzy)."""
    n1 = normalize_name(name1)
    n2 = normalize_name(name2)
    
    if n1 == n2:
        return True
    
    # Handle apostrophes and hyphens
    n1_clean = n1.replace("'", "").replace("-", " ").replace(".", "")
    n2_clean = n2.replace("'", "").replace("-", " ").replace(".", "")
    if n1_clean == n2_clean:
        return True
    
    # Convert both to initial format and compare
    init1 = to_initial_format(name1).lower()
    init2 = to_initial_format(name2).lower()
    if init1 == init2:
        return True
    
    # Last name + first initial match
    parts1 = n1.split()
    parts2 = n2.split()
    if len(parts1) >= 2 and len(parts2) >= 2:
        if parts1[-1] == parts2[-1] and parts1[0][0] == parts2[0][0]:
            return True
    
    return False


def parse_odds_history():
    """Parse odds_history files to get players with betting lines."""
    print("📊 Parsing odds history files...")
    
    players_with_lines = []
    games_data = {}
    
    history_files = glob.glob(str(ODDS_HISTORY_DIR / "history_*.json"))
    
    for filepath in history_files:
        with open(filepath, "r") as f:
            history = json.load(f)
        
        if not history:
            continue
        
        # Get the latest snapshot
        latest = history[-1] if isinstance(history, list) else history
        
        game_id = latest.get("id")
        home_team_full = latest.get("home_team", "")
        away_team_full = latest.get("away_team", "")
        
        home_team = TEAM_NAME_MAP.get(home_team_full, home_team_full)
        away_team = TEAM_NAME_MAP.get(away_team_full, away_team_full)
        
        # Get game context (spread, total)
        game_context = latest.get("game_context", {})
        home_spread = game_context.get("home_spread")
        game_total = game_context.get("game_total")
        
        games_data[game_id] = {
            "home_team": norm_team(home_team),
            "away_team": norm_team(away_team),
            "home_spread": home_spread,
            "game_total": game_total,
            "commence_time": latest.get("commence_time")
        }
        
        # Extract player names from rush attempts market
        bookmakers = latest.get("bookmakers", [])
        for book in bookmakers:
            markets = book.get("markets", [])
            for market in markets:
                if market.get("key") == "player_rush_attempts":
                    outcomes = market.get("outcomes", [])
                    for outcome in outcomes:
                        player_name = outcome.get("description")
                        line = outcome.get("point")
                        
                        if player_name and line:
                            # Determine which team (need to look up)
                            players_with_lines.append({
                                "name": player_name,
                                "game_id": game_id,
                                "home_team": norm_team(home_team),
                                "away_team": norm_team(away_team),
                                "line": line,
                                "home_spread": home_spread,
                                "game_total": game_total
                            })
    
    # Deduplicate players
    seen = set()
    unique_players = []
    for p in players_with_lines:
        key = (p["name"], p["game_id"])
        if key not in seen:
            seen.add(key)
            unique_players.append(p)
    
    print(f"   Found {len(unique_players)} unique player-game combinations")
    return unique_players, games_data


def load_historical_data():
    """Load historical RB data."""
    print(f"📂 Loading historical data from: {HISTORICAL_DATA_PATH}")
    df = pd.read_parquet(HISTORICAL_DATA_PATH)
    df["player_key"] = df["player_id"].astype(str)
    df = df.sort_values(["player_key", "season", "week"]).reset_index(drop=True)
    
    # Add recency features
    df["last_week_carries"] = df.groupby("player_key")["rb_carries"].shift(1).fillna(0)
    df["last_week_snap_share"] = df.groupby("player_key")["rb_ed_carry_share_all"].shift(1).fillna(0)
    
    print(f"   Loaded {len(df)} rows, {df['player_key'].nunique()} unique players")
    return df


def load_weather_data():
    """Load weather data for current games."""
    print(f"🌤️  Loading weather data from: {WEATHER_JSON_PATH}")
    if not WEATHER_JSON_PATH.exists():
        print("   ⚠️  Weather file not found")
        return {}
    
    with open(WEATHER_JSON_PATH, "r") as f:
        data = json.load(f)
    
    # Index by (home_team, away_team)
    weather_lookup = {}
    for game in data.get("games", []):
        key = (norm_team(game["home_team"]), norm_team(game["away_team"]))
        weather_lookup[key] = {
            "temp_f": game.get("temp_f"),
            "wind_mph": game.get("wind_mph"),
            "is_dome": game.get("is_dome", False)
        }
    
    print(f"   Loaded weather for {len(weather_lookup)} games")
    return weather_lookup


def load_depth_chart_data():
    """Load depth chart data."""
    print(f"📋 Loading depth chart from: {DEPTH_CHART_JSON_PATH}")
    if not DEPTH_CHART_JSON_PATH.exists():
        print("   ⚠️  Depth chart file not found")
        return {}
    
    with open(DEPTH_CHART_JSON_PATH, "r") as f:
        data = json.load(f)
    
    # Index by (team, normalized_name) - store both normalized and original for matching
    depth_lookup = {}
    for team, players in data.items():
        team = norm_team(team)
        for player in players:
            # Store with normalized name as key
            name_key = normalize_name(player["name"])
            depth_lookup[(team, name_key)] = {
                "depth_rank": player.get("depth_rank", 99),
                "is_projected_starter": player.get("is_projected_starter", 0),
                "is_active": player.get("is_active", 1),
                "player_id": player.get("player_id"),
                "original_name": player["name"]
            }
    
    print(f"   Loaded depth chart for {len(depth_lookup)} player-team combinations")
    return depth_lookup


def find_player_in_historical(player_name, hist_df, team_hint=None):
    """Find a player in historical data by name matching."""
    # Convert to initial format (e.g., "Kyren Williams" -> "K.Williams")
    initial_name = to_initial_format(player_name)
    
    # Try exact match on initial format first
    matches = hist_df[hist_df["full_name"] == initial_name]
    
    # If no exact match, try fuzzy match
    if len(matches) == 0:
        matches = hist_df[hist_df["full_name"].apply(lambda x: names_match(x, player_name))]
    
    if len(matches) > 0:
        # If team hint provided, filter further
        if team_hint:
            team_matches = matches[matches["posteam"] == team_hint]
            if len(team_matches) > 0:
                matches = team_matches
        
        # Return the most recent player_id
        latest = matches.sort_values(["season", "week"], ascending=False).iloc[0]
        return latest["player_key"], latest["full_name"], latest["posteam"]
    
    return None, None, None


def get_player_history(player_key, hist_df, current_season, current_week, n_weeks=6):
    """Get the last n weeks of data for a player."""
    player_data = hist_df[hist_df["player_key"] == player_key].copy()
    player_data = player_data.sort_values(["season", "week"], ascending=False)
    
    # Filter to games before current week
    # Create a sortable key: season * 100 + week
    player_data["sort_key"] = player_data["season"] * 100 + player_data["week"]
    current_key = current_season * 100 + current_week
    
    player_data = player_data[player_data["sort_key"] < current_key]
    
    # Get last n weeks
    history = player_data.head(n_weeks)
    
    return history


def compute_opponent_defense_stats(hist_df, opponent, current_season):
    """Compute season-to-date defensive stats for opponent."""
    # Get all games where this team was the opponent (defteam)
    opp_games = hist_df[
        (hist_df["opponent"] == opponent) & 
        (hist_df["season"] == current_season)
    ]
    
    if len(opp_games) == 0:
        # Fall back to league average
        return {
            "opponent_rush_epa_allowed": 0.0,
            "opponent_rush_yards_per_game_allowed": 100.0
        }
    
    # Aggregate defensive stats
    return {
        "opponent_rush_epa_allowed": opp_games["opponent_rush_epa_allowed"].mean(),
        "opponent_rush_yards_per_game_allowed": opp_games["opponent_rush_yards_per_game_allowed"].mean()
    }


def main():
    print("=" * 60)
    print("GENERATE PLAYER MODEL ROWS")
    print("=" * 60)
    
    # 1. Parse odds history to get players with lines
    players_with_lines, games_data = parse_odds_history()
    
    if not players_with_lines:
        print("❌ No players with betting lines found!")
        return
    
    # 2. Load historical data
    hist_df = load_historical_data()
    
    # Determine current season/week from historical data
    current_season = int(hist_df["season"].max())
    current_week = int(hist_df[hist_df["season"] == current_season]["week"].max()) + 1
    print(f"\n📅 Predicting for: Season {current_season}, Week {current_week}")
    
    # 3. Load current week metadata
    weather_data = load_weather_data()
    depth_chart_data = load_depth_chart_data()
    
    # 4. Build model rows for each player
    print("\n🔧 Building model rows...")
    model_rows = []
    skipped_non_rb = []
    
    for player_info in players_with_lines:
        player_name = player_info["name"]
        home_team = player_info["home_team"]
        away_team = player_info["away_team"]
        home_spread = player_info["home_spread"]
        line = player_info["line"]
        
        # Try to determine player's team from depth chart (RBs only)
        player_team = None
        for team in [home_team, away_team]:
            name_key = normalize_name(player_name)
            if (team, name_key) in depth_chart_data:
                player_team = team
                break
        
        # Skip if not found in RB depth chart (likely a QB or other position)
        if not player_team:
            skipped_non_rb.append(player_name)
            continue
        
        # Find player in historical data
        player_key, matched_name, hist_team = find_player_in_historical(
            player_name, hist_df, team_hint=player_team
        )
        
        if not player_key:
            print(f"   ⚠️  Could not find historical data for: {player_name}")
            continue
        
        # Use historical team if we couldn't determine from depth chart
        if not player_team:
            player_team = hist_team
        
        # Determine is_home and opponent
        if player_team == home_team:
            is_home = 1
            opponent = away_team
            team_spread = home_spread if home_spread else 0
        else:
            is_home = 0
            opponent = home_team
            team_spread = -home_spread if home_spread else 0
        
        # Get weather
        weather_key = (home_team, away_team)
        weather = weather_data.get(weather_key, {})
        wind_effective = weather.get("wind_mph", 0) or 0
        is_dome = 1 if weather.get("is_dome", False) else 0
        
        # Get depth chart info
        name_key = normalize_name(player_name)
        depth_info = depth_chart_data.get((player_team, name_key), {})
        depth_rank = depth_info.get("depth_rank", 4)
        is_projected_starter = depth_info.get("is_projected_starter", 0)
        is_active = depth_info.get("is_active", 1)
        
        # Get opponent defense stats
        def_stats = compute_opponent_defense_stats(hist_df, opponent, current_season)
        
        # Get player's last 6 weeks of history
        history = get_player_history(player_key, hist_df, current_season, current_week, n_weeks=L)
        
        # Build TCN history (last 6 weeks of features)
        tcn_history = []
        for _, row in history.iterrows():
            week_data = {col: float(row[col]) if pd.notna(row[col]) else 0.0 for col in TCN_COLS if col in row}
            tcn_history.append(week_data)
        
        # Pad if less than 6 weeks
        while len(tcn_history) < L:
            tcn_history.append({col: 0.0 for col in TCN_COLS})
        
        # Reverse so oldest is first (for TCN)
        tcn_history = list(reversed(tcn_history))
        
        # Get last week stats for recency features
        if len(history) > 0:
            last_week = history.iloc[0]
            last_week_carries = float(last_week["rb_carries"]) if pd.notna(last_week["rb_carries"]) else 0
            last_week_snap_share = float(last_week["rb_ed_carry_share_all"]) if pd.notna(last_week["rb_ed_carry_share_all"]) else 0
        else:
            last_week_carries = 0
            last_week_snap_share = 0
        
        # Build the model row
        model_row = {
            # Identification
            "player_name": player_name,
            "player_key": player_key,
            "full_name": matched_name,
            "posteam": player_team,
            "opponent": opponent,
            
            # Game context
            "season": current_season,
            "week": current_week,
            "is_home": is_home,
            "home_team": home_team,
            "away_team": away_team,
            
            # Betting context
            "rush_attempts_line": line,
            "home_spread": home_spread,
            "team_spread": team_spread,
            "implied_game_script": team_spread,
            
            # Weather
            "wind_effective": wind_effective,
            "is_dome": is_dome,
            
            # Role
            "depth_rank": depth_rank,
            "is_projected_starter": is_projected_starter,
            "is_active": is_active,
            
            # Opponent defense
            "opponent_rush_epa_allowed": def_stats["opponent_rush_epa_allowed"],
            "opponent_rush_yards_per_game_allowed": def_stats["opponent_rush_yards_per_game_allowed"],
            
            # Recency
            "last_week_carries": last_week_carries,
            "last_week_snap_share": last_week_snap_share,
            
            # Historical TCN features (last 6 weeks)
            "tcn_history": tcn_history,
            
            # Games in history
            "history_weeks": len(history)
        }
        
        model_rows.append(model_row)
        print(f"   ✓ {player_name} ({player_team}) - Line: {line}, Spread: {team_spread:+.1f}")
    
    # 5. Save output
    print(f"\n💾 Saving to: {OUTPUT_PATH}")
    
    output = {
        "generated_at": datetime.now().isoformat(),
        "prediction_season": current_season,
        "prediction_week": current_week,
        "tcn_window_size": L,
        "tcn_features": TCN_COLS,
        "meta_features": META_COLS,
        "players": model_rows
    }
    
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Players with betting lines: {len(players_with_lines)}")
    print(f"Skipped (not in RB depth chart): {len(skipped_non_rb)}")
    if skipped_non_rb:
        print(f"   Non-RBs skipped: {', '.join(skipped_non_rb[:10])}{'...' if len(skipped_non_rb) > 10 else ''}")
    print(f"Players matched to history: {len(model_rows)}")
    print(f"Output saved to: {OUTPUT_PATH}")
    
    # Show sample
    if model_rows:
        print("\n--- Sample Output (first 3 players) ---")
        for row in model_rows[:3]:
            print(f"\n{row['player_name']} ({row['posteam']}):")
            print(f"  Line: {row['rush_attempts_line']} | Spread: {row['team_spread']:+.1f}")
            print(f"  Depth: {row['depth_rank']} | Starter: {row['is_projected_starter']} | Active: {row['is_active']}")
            print(f"  History weeks: {row['history_weeks']} | Last wk carries: {row['last_week_carries']}")
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()


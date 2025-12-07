#!/usr/bin/env python3
"""
build_rb_depth_chart_curr.py

Fetches NFL depth charts from RapidAPI and generates a JSON file with RB info:
  - depth_rank: Player's position on the depth chart (1=RB1, 2=RB2, etc.)
  - is_projected_starter: 1 if this is the top active RB, 0 otherwise
  - is_active: 1 if active, 0 if inactive (from injury report)

Uses:
  - Tank01 NFL API: getNFLDepthCharts endpoint
  - nfl_rbs_health.json: For active/inactive status

Output:
  - nfl_rbs_depth_chart.json

Usage:
  python build_rb_depth_chart_curr.py
"""

import os
import json
import time
import requests
from pathlib import Path
from dotenv import load_dotenv

# =========================
# CONFIGURATION
# =========================
load_dotenv()
API_KEY = os.getenv('RAPID_API_KEY')

if not API_KEY:
    raise ValueError("❌ RAPID_API_KEY not found in .env file")

BASE_URL = "https://tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com"
HEADERS = {
    "X-RapidAPI-Key": API_KEY,
    "X-RapidAPI-Host": "tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com"
}

SCRIPT_DIR = Path(__file__).parent
HEALTH_JSON_PATH = SCRIPT_DIR / "nfl_rbs_health.json"
OUTPUT_PATH = SCRIPT_DIR / "nfl_rbs_depth_chart.json"

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


def normalize_name(name):
    """Normalize player name for matching."""
    if not name:
        return ""
    name = str(name).strip()
    # Remove suffixes
    for suffix in [" Jr.", " Sr.", " III", " II", " IV", " V"]:
        name = name.replace(suffix, "")
    return name.lower().strip()


def names_match(name1, name2):
    """Check if two names match (fuzzy)."""
    n1 = normalize_name(name1)
    n2 = normalize_name(name2)
    
    # Exact match
    if n1 == n2:
        return True
    
    # One contains the other (handles "De'Von Achane" vs "Devon Achane")
    n1_clean = n1.replace("'", "").replace("-", "")
    n2_clean = n2.replace("'", "").replace("-", "")
    if n1_clean == n2_clean:
        return True
    
    # Last name match (handles first name differences)
    parts1 = n1.split()
    parts2 = n2.split()
    if parts1 and parts2 and parts1[-1] == parts2[-1]:
        # Same last name, check first initial
        if parts1[0][0] == parts2[0][0]:
            return True
    
    return False


def fetch_depth_charts():
    """Fetch depth charts from Tank01 NFL API."""
    print("🌐 Fetching depth charts from API...")
    
    url = f"{BASE_URL}/getNFLDepthCharts"
    
    try:
        response = requests.get(url, headers=HEADERS)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ API Error: {response.status_code}")
            print(response.text)
            return None
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return None


def parse_depth_charts(api_data):
    """
    Parse API response to extract RB depth charts.
    Returns dict: { team_abbr: [list of RB players with depth_rank] }
    """
    result = {}
    
    if not api_data or 'body' not in api_data:
        print("❌ Invalid API response format")
        return result
    
    body = api_data['body']
    
    # Handle body as a list (each item is a team's depth chart)
    if isinstance(body, list):
        for team_entry in body:
            if not isinstance(team_entry, dict):
                continue
            
            # Get team abbreviation
            team_abbr = team_entry.get('teamAbv') or team_entry.get('team') or team_entry.get('teamAbbr')
            if not team_abbr:
                continue
            team_abbr = norm_team(team_abbr)
            
            rbs = []
            
            # Look for RB depth chart - could be nested various ways
            # Option 1: Direct 'RB' key with list of players
            rb_data = team_entry.get('RB') or team_entry.get('rb')
            
            # Option 2: Nested under 'depthChart' or 'offense'
            if not rb_data:
                depth_chart = team_entry.get('depthChart') or team_entry.get('depth_chart') or {}
                rb_data = depth_chart.get('RB') or depth_chart.get('rb')
            
            if not rb_data:
                offense = team_entry.get('offense') or {}
                rb_data = offense.get('RB') or offense.get('rb')
            
            # Parse RB list
            if isinstance(rb_data, list):
                for idx, player in enumerate(rb_data):
                    if isinstance(player, dict):
                        name = (player.get('longName') or player.get('name') or 
                                player.get('playerName') or player.get('espnName') or 'Unknown')
                        player_id = (player.get('playerID') or player.get('espnID') or 
                                     player.get('nflPlayerID') or player.get('playerId'))
                        depth = player.get('depthOrder') or player.get('depth') or (idx + 1)
                        
                        rbs.append({
                            "name": name,
                            "player_id": player_id,
                            "depth_rank": int(depth)
                        })
                    elif isinstance(player, str):
                        rbs.append({
                            "name": player,
                            "player_id": None,
                            "depth_rank": idx + 1
                        })
            
            if rbs:
                # Sort by depth_rank
                rbs.sort(key=lambda x: x['depth_rank'])
                result[team_abbr] = rbs
    
    # Handle body as a dict (keyed by team)
    elif isinstance(body, dict):
        for team_key, team_data in body.items():
            team_abbr = norm_team(team_key)
            
            rbs = []
            
            if isinstance(team_data, dict):
                rb_data = team_data.get('RB') or team_data.get('rb')
                
                if isinstance(rb_data, list):
                    for idx, player in enumerate(rb_data):
                        if isinstance(player, dict):
                            name = player.get('longName') or player.get('name', 'Unknown')
                            player_id = player.get('playerID')
                            depth = player.get('depthOrder') or (idx + 1)
                            
                            rbs.append({
                                "name": name,
                                "player_id": player_id,
                                "depth_rank": int(depth)
                            })
            
            if rbs:
                rbs.sort(key=lambda x: x['depth_rank'])
                result[team_abbr] = rbs
    
    return result


def main():
    print("=" * 60)
    print("BUILD RB DEPTH CHART (from API)")
    print("=" * 60)
    
    # 1. Fetch depth charts from API
    api_data = fetch_depth_charts()
    
    if not api_data:
        print("❌ Failed to fetch depth charts. Exiting.")
        return
    
    # Debug: print raw response structure
    print("\n📋 API Response Structure:")
    if 'body' in api_data:
        body = api_data['body']
        print(f"Body type: {type(body).__name__}")
        
        if isinstance(body, list) and body:
            print(f"Number of teams: {len(body)}")
            sample = body[0]
            print(f"Sample team entry keys: {list(sample.keys()) if isinstance(sample, dict) else 'N/A'}")
            if isinstance(sample, dict):
                team_name = sample.get('teamAbv') or sample.get('team', 'Unknown')
                print(f"Sample team: {team_name}")
                # Show RB data if present
                rb_data = sample.get('RB') or sample.get('rb')
                if rb_data:
                    print(f"RB data sample: {rb_data[:2] if isinstance(rb_data, list) else rb_data}")
        elif isinstance(body, dict):
            print(f"Teams: {list(body.keys())[:5]}...")
    
    # 2. Parse depth charts
    depth_charts = parse_depth_charts(api_data)
    
    print(f"\n✅ Parsed depth charts for {len(depth_charts)} teams")
    
    # 3. Load health/injury data
    print(f"\n📋 Loading health data from: {HEALTH_JSON_PATH}")
    if HEALTH_JSON_PATH.exists():
        with open(HEALTH_JSON_PATH, "r") as f:
            health_data = json.load(f)
    else:
        print("⚠️  Health JSON not found, assuming all players active")
        health_data = {}
    
    # 4. Build final output
    print("\n🔧 Building final depth chart with active status...")
    output = {}
    
    all_teams = set(depth_charts.keys()) | set(health_data.keys())
    
    for team in sorted(all_teams):
        team_depth = depth_charts.get(team, [])
        team_health = health_data.get(team, [])
        
        # Create health lookup
        health_status = {}
        for p in team_health:
            health_status[p["name"]] = p["status"]
        
        players = []
        
        # Process depth chart players
        for player in team_depth:
            name = player["name"]
            depth_rank = player["depth_rank"]
            player_id = player.get("player_id")
            
            # Find matching health status
            is_active = 1  # Default to active
            for health_name, status in health_status.items():
                if names_match(name, health_name):
                    is_active = 1 if status == "Active" else 0
                    break
            
            players.append({
                "name": name,
                "player_id": player_id,
                "depth_rank": depth_rank,
                "is_active": is_active,
                "is_projected_starter": 0
            })
        
        # Add players from health that aren't in depth chart
        for health_player in team_health:
            health_name = health_player["name"]
            already_added = any(names_match(health_name, p["name"]) for p in players)
            
            if not already_added:
                is_active = 1 if health_player["status"] == "Active" else 0
                players.append({
                    "name": health_name,
                    "player_id": None,
                    "depth_rank": 99,
                    "is_active": is_active,
                    "is_projected_starter": 0
                })
        
        # Calculate is_projected_starter
        # Logic: Among ACTIVE players, the one with the lowest depth_rank is the starter
        active_players = [p for p in players if p["is_active"] == 1]
        
        if active_players:
            min_rank = min(p["depth_rank"] for p in active_players)
            for p in players:
                if p["is_active"] == 1 and p["depth_rank"] == min_rank:
                    p["is_projected_starter"] = 1
        
        # Sort by depth_rank
        players.sort(key=lambda x: (x["depth_rank"], x["name"]))
        
        output[team] = players
    
    # 5. Save output
    print(f"\n💾 Saving to: {OUTPUT_PATH}")
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=4)
    
    # 6. Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    total_players = sum(len(v) for v in output.values())
    total_active = sum(sum(1 for p in v if p["is_active"] == 1) for v in output.values())
    total_starters = sum(sum(1 for p in v if p["is_projected_starter"] == 1) for v in output.values())
    
    print(f"Teams: {len(output)}")
    print(f"Total RBs: {total_players}")
    print(f"Active RBs: {total_active}")
    print(f"Projected Starters: {total_starters}")
    
    # Show sample
    print("\n--- Sample Output (first 3 teams) ---")
    for i, (team, players) in enumerate(output.items()):
        if i >= 3:
            break
        print(f"\n{team}:")
        for p in players[:5]:  # Show top 5 per team
            starter = "★" if p["is_projected_starter"] == 1 else " "
            active = "✓" if p["is_active"] == 1 else "✗"
            print(f"  {starter} {p['depth_rank']:2d}. {p['name']:<25} Active: {active}")
    
    print(f"\n✅ Done! Output saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()


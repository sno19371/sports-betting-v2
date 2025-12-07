import os
import json
import time
import requests
from dotenv import load_dotenv

# 1. Load your API Key
load_dotenv()
API_KEY = os.getenv('RAPID_API_KEY')

if not API_KEY:
    raise ValueError("❌ RAPID_API_KEY not found in .env file")

# Configuration
BASE_URL = "https://tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com"
HEADERS = {
    "X-RapidAPI-Key": API_KEY,
    "X-RapidAPI-Host": "tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com"
}

# Hardcoded list of all 32 NFL Team Abbreviations to save you 1 API call
ALL_TEAMS = [
    'ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE', 'DAL', 'DEN', 'DET', 'GB', 'HOU', 'IND', 
    'JAX', 'KC', 'LV', 'LAC', 'LAR', 'MIA', 'MIN', 'NE', 'NO', 'NYG', 'NYJ', 'PHI', 'PIT', 'SEA', 
    'SF', 'TB', 'TEN', 'WAS'
]

def get_roster(team_abv):
    """Fetches the full roster for a specific team."""
    url = f"{BASE_URL}/getNFLTeamRoster"
    params = {'teamAbv': team_abv, 'getStats': 'true'} # getStats often triggers better metadata
    
    try:
        response = requests.get(url, headers=HEADERS, params=params)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"Error connecting for {team_abv}: {e}")
    return None

def determine_status(player):
    """Parses the injury object to return simple Active/Inactive."""
    # Default to Active
    status = "Active"
    
    # Check if injury object exists and has data
    if 'injury' in player:
        injury = player['injury']
        # Some endpoints return empty strings, others return None. Handle both.
        designation = injury.get('designation', '').lower() if injury.get('designation') else ""
        description = injury.get('description', '').lower() if injury.get('description') else ""

        # Logic: If explicitly Out or IR, they are Inactive.
        # Note: "Questionable" and "Doubtful" are technically "Active" until gameday inactives.
        if "out" in designation or "ir" in designation or "injured reserve" in description:
            status = "Inactive"
            
    return status

def main():
    print(f"🚀 Starting scan of {len(ALL_TEAMS)} NFL teams for Running Backs...")
    
    all_rbs_data = {}

    for team in ALL_TEAMS:
        print(f"Fetching roster for {team}...")
        data = get_roster(team)
        
        team_rbs = []
        
        if data and 'body' in data and 'roster' in data['body']:
            roster = data['body']['roster']
            
            for player in roster:
                # FILTER: Only Running Backs (RB)
                # Sometimes pos is 'RB', sometimes 'FB/RB'. We look for 'RB'
                pos = player.get('pos', '')
                if 'RB' in pos:
                    name = player.get('longName', 'Unknown Player')
                    status = determine_status(player)
                    
                    team_rbs.append({
                        "name": name,
                        "status": status
                    })
        
        # Save to our master dictionary
        all_rbs_data[team] = team_rbs
        
        # IMPORTANT: Sleep to respect rate limits (standard is roughly 1-2 calls per sec safe zone)
        time.sleep(0.5)

    # Output to JSON
    output_file = 'nfl_rbs_health.json'
    with open(output_file, 'w') as f:
        json.dump(all_rbs_data, f, indent=4)
        
    print(f"\n✅ Done! Data saved to {output_file}")
    
    # Preview
    print("\n--- Sample Output (First Team) ---")
    first_team = list(all_rbs_data.keys())[0]
    print(f"Team: {first_team}")
    print(json.dumps(all_rbs_data[first_team], indent=2))

if __name__ == "__main__":
    main()
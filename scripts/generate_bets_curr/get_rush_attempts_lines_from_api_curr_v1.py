import os
import json
import time
import requests
from dotenv import load_dotenv
from datetime import datetime, timezone, timedelta

# --- CONFIGURATION ---
load_dotenv()
API_KEY = os.getenv('THE_ODDS_API_KEY')
BASE_URL = 'https://api.the-odds-api.com/v4/sports'
SPORT = 'americanfootball_nfl'
PROP_MARKET = 'player_rush_attempts' 
CACHE_DIR = 'odds_history'
DAYS_AHEAD = 3  # Only fetch games within this many days

if not API_KEY:
    raise ValueError("API Key not found in .env file")

if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

# --- HELPER FUNCTIONS ---
def get_existing_history(filename):
    filepath = os.path.join(CACHE_DIR, filename)
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []

def save_history(filename, new_data):
    filepath = os.path.join(CACHE_DIR, filename)
    history = get_existing_history(filename)
    
    # Add timestamp and append
    new_data['_fetched_at'] = datetime.now(timezone.utc).isoformat()
    history.append(new_data)
    
    with open(filepath, 'w') as f:
        json.dump(history, f, indent=4)

# --- CORE FUNCTIONS ---

def get_bulk_game_lines():
    """
    Efficiently fetches Spreads and Totals for ALL games in 1 call.
    Returns a dictionary keyed by game_id.
    """
    print("🌐 Bulk fetching Spreads & Totals for all games...")
    url = f"{BASE_URL}/{SPORT}/odds"
    params = {
        'apiKey': API_KEY,
        'regions': 'us',
        'markets': 'spreads,totals', # Bulk fetch these cheaper markets
        'oddsFormat': 'american',
    }
    response = requests.get(url, params=params)
    
    if response.status_code != 200:
        print(f"Error fetching game lines: {response.text}")
        return {}
    
    data = response.json()
    
    # Convert list into a dictionary for easy lookup: { 'game_id': {data} }
    lines_map = {game['id']: game for game in data}
    return lines_map

def extract_home_spread_and_total(game_lines, home_team):
    """Parses the complex odds response to find just the numbers you need."""
    result = {'spread': 'N/A', 'total': 'N/A'}
    
    if not game_lines or 'bookmakers' not in game_lines:
        return result

    # We'll just grab the first bookmaker (usually DraftKings or FanDuel) for the 'consensus' line
    # You can iterate through them if you prefer a specific book
    book = game_lines['bookmakers'][0] 
    
    for market in book['markets']:
        # 1. FIND SPREAD
        if market['key'] == 'spreads':
            for outcome in market['outcomes']:
                if outcome['name'] == home_team:
                    result['spread'] = outcome['point']
        
        # 2. FIND TOTAL
        elif market['key'] == 'totals':
            # Totals are usually the same for over/under, just grab the first one
            result['total'] = market['outcomes'][0]['point']
            
    return result

def fetch_and_store_props(game_id, home_team, away_team, game_lines):
    """Fetches player props and merges in the game lines."""
    filename = f"history_{home_team}_vs_{away_team}_{game_id}.json".replace(" ", "_")
    
    # 15-minute safety rule
    history = get_existing_history(filename)
    if history:
        last_snapshot = history[-1]
        last_fetch = datetime.fromisoformat(last_snapshot['_fetched_at'])
        minutes_since_last = (datetime.now(timezone.utc) - last_fetch).total_seconds() / 60
        if minutes_since_last < 15:
            print(f"zzz Skipping {away_team} @ {home_team} (Data is fresh)")
            return

    print(f"🌐 Fetching PROPS for {away_team} @ {home_team}...")
    url = f"{BASE_URL}/{SPORT}/events/{game_id}/odds"
    params = {
        'apiKey': API_KEY,
        'regions': 'us',
        'markets': PROP_MARKET,
        'oddsFormat': 'american'
    }
    response = requests.get(url, params=params)
    
    if response.status_code != 200:
        print(f"Error: {response.text}")
        return

    prop_data = response.json()
    
    # --- MERGE STEP ---
    # We take the "bulk" lines we got earlier and inject them into this record
    lines_summary = extract_home_spread_and_total(game_lines, home_team)
    
    # Add a new easy-to-read field at the top of the JSON
    prop_data['game_context'] = {
        'home_team': home_team,
        'home_spread': lines_summary['spread'],
        'game_total': lines_summary['total']
    }
    
    save_history(filename, prop_data)

def main():
    # 1. Get Schedule (re-using the bulk lines call as the schedule source!)
    # The 'odds' endpoint returns the schedule AND the odds, so we kill two birds with one stone.
    games_map = get_bulk_game_lines()
    games = list(games_map.values())
    
    print(f"Found {len(games)} games.")
    now = datetime.now(timezone.utc)
    
    for game in games:
        game_id = game['id']
        home_team = game['home_team']
        away_team = game['away_team']
        commence_time = datetime.fromisoformat(game['commence_time'].replace('Z', '+00:00'))

        # Time Filter: skip past games
        if commence_time < now:
            continue
        
        # Time Filter: skip games too far in the future
        max_future = now + timedelta(days=DAYS_AHEAD)
        if commence_time > max_future:
            print(f"⏭️ Skipping {away_team} @ {home_team} (more than {DAYS_AHEAD} days away)")
            continue
        
        # Pass the specific lines for this game into the prop fetcher
        this_game_lines = games_map.get(game_id)
        fetch_and_store_props(game_id, home_team, away_team, this_game_lines)

if __name__ == "__main__":
    main()
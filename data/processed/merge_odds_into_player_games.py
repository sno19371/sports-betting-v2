import pandas as pd
import numpy as np

# --- 1. CONFIGURATION ---
ODDS_CSV = 'nfl_odds_historical.csv'
PLAYER_PARQUET = 'player_games_with_weather_3h_patched.parquet'
OUTPUT_FILE = 'player_games_with_odds_final_lar_fixed.parquet'

# --- 2. TEAM MAPPING (Full Name -> Abbreviation) ---
# UPDATED: 'Los Angeles Rams' is now mapped to 'LAR' to match your data
team_map = {
    'Arizona Cardinals': 'ARI', 
    'Atlanta Falcons': 'ATL', 
    'Baltimore Ravens': 'BAL',
    'Buffalo Bills': 'BUF', 
    'Carolina Panthers': 'CAR', 
    'Chicago Bears': 'CHI',
    'Cincinnati Bengals': 'CIN', 
    'Cleveland Browns': 'CLE', 
    'Dallas Cowboys': 'DAL',
    'Denver Broncos': 'DEN', 
    'Detroit Lions': 'DET', 
    'Green Bay Packers': 'GB',
    'Houston Texans': 'HOU', 
    'Indianapolis Colts': 'IND', 
    'Jacksonville Jaguars': 'JAX',
    'Kansas City Chiefs': 'KC', 
    'Las Vegas Raiders': 'LV', 
    'Oakland Raiders': 'LV',       # Mapped to modern abbreviation
    'Los Angeles Chargers': 'LAC', 
    'San Diego Chargers': 'LAC',   # Mapped to modern abbreviation
    'Los Angeles Rams': 'LAR',     # <--- FIXED (Was 'LA')
    'St. Louis Rams': 'LAR',       # <--- FIXED (Was 'LA')
    'Miami Dolphins': 'MIA', 
    'Minnesota Vikings': 'MIN', 
    'New England Patriots': 'NE',
    'New Orleans Saints': 'NO', 
    'New York Giants': 'NYG', 
    'New York Jets': 'NYJ',
    'Philadelphia Eagles': 'PHI', 
    'Pittsburgh Steelers': 'PIT', 
    'San Francisco 49ers': 'SF',
    'Seattle Seahawks': 'SEA', 
    'Tampa Bay Buccaneers': 'TB', 
    'Tennessee Titans': 'TEN',
    'Washington Commanders': 'WAS', 
    'Washington Football Team': 'WAS', 
    'Washington Redskins': 'WAS'
}

def process_odds_data(csv_path):
    print(f"Loading Odds CSV from: {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Standardize Headers
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('?', '')
    
    # Fix Date
    print("Parsing dates...")
    df['join_date'] = pd.to_datetime(df['date'], format='mixed').dt.date
    
    # Map Teams
    df['join_home'] = df['home_team'].map(team_map)
    df['join_away'] = df['away_team'].map(team_map)
    
    # DEBUG: Check specifically for Rams in your CSV to ensure they are mapping
    rams_check = df[df['home_team'] == 'Los Angeles Rams']
    if not rams_check.empty:
        print(f"DEBUG: 'Los Angeles Rams' mapped to: {rams_check['join_home'].iloc[0]} (Should be LAR)")

    # Check for unmapped teams
    missing_home = df[df['join_home'].isna()]['home_team'].unique()
    if len(missing_home) > 0:
        print(f"⚠️ WARNING: These teams in CSV could not be mapped: {missing_home}")

    # Select Columns
    cols_to_keep = [
        'join_date', 
        'join_home', 
        'join_away', 
        'home_line_close', 
        'away_line_close', 
        'total_score_open'
    ]
    
    try:
        df_filtered = df[cols_to_keep].copy()
    except KeyError as e:
        print(f"❌ Error: Column not found. Available columns are: {list(df.columns)}")
        raise e
        
    return df_filtered

def process_player_data(parquet_path):
    print(f"Loading Player Parquet from: {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    
    # Fix Date
    df['join_date'] = pd.to_datetime(df['game_date'], errors='coerce').dt.date
    
    # Use existing team columns
    df['join_home'] = df['home_team']
    df['join_away'] = df['away_team']
    
    return df

def main():
    try:
        df_odds = process_odds_data(ODDS_CSV)
        df_players = process_player_data(PLAYER_PARQUET)
    except Exception as e:
        print(f"❌ Error loading files: {e}")
        return
    
    print(f"Player Rows: {len(df_players)} | Odds Rows: {len(df_odds)}")

    # Merge
    print("Merging data...")
    merged_df = pd.merge(
        df_players,
        df_odds,
        how='left',
        on=['join_date', 'join_home', 'join_away']
    )
    
    # Validation check specifically for LAR
    lar_rows = merged_df[merged_df['home_team'] == 'LAR']
    lar_matched = lar_rows['home_line_close'].notna().sum()
    print(f"✅ Found odds for {lar_matched} / {len(lar_rows)} Rams (LAR) home games.")
    
    total_matched = merged_df['home_line_close'].notna().sum()
    print(f"✅ Total matched rows: {total_matched} / {len(merged_df)}")

    # Cleanup Join Keys
    merged_df.drop(columns=['join_date', 'join_home', 'join_away'], inplace=True)
    
    # Save
    print(f"Saving to {OUTPUT_FILE}...")
    merged_df.to_parquet(OUTPUT_FILE)
    print("Done.")

if __name__ == "__main__":
    main()
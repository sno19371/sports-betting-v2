import pandas as pd
import numpy as np
from datetime import timedelta

# --- 1. CONFIGURATION ---
ODDS_CSV = 'nfl_odds_historical.csv'
PLAYER_PARQUET = 'player_games_with_weather_3h_patched.parquet'
OUTPUT_FILE = 'player_games_with_odds_flexed_fixed.parquet'

# --- 2. TEAM MAPPING ---
# Maps CSV Full Names -> Parquet Abbreviations
# CRITICAL: Rams are mapped to 'LAR'
team_map = {
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

def process_odds_data(csv_path):
    print(f"Loading Odds CSV from: {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # 1. Clean Headers (remove spaces, lowercase)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('?', '')
    
    # 2. Fix Date (format='mixed' prevents crashes on inconsistent CSV dates)
    print("Parsing Odds dates...")
    df['join_date'] = pd.to_datetime(df['date'], format='mixed').dt.date
    
    # 3. Map Teams
    df['join_home'] = df['home_team'].map(team_map)
    df['join_away'] = df['away_team'].map(team_map)
    
    # 4. Filter Columns (Only keep what we need)
    cols_to_keep = [
        'join_date', 
        'join_home', 
        'join_away', 
        'home_line_close', 
        'away_line_close', 
        'total_score_open'
    ]
    
    return df[cols_to_keep].copy()

def process_player_data(parquet_path):
    print(f"Loading Player Parquet from: {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    
    # Fix Date
    df['join_date'] = pd.to_datetime(df['game_date'], errors='coerce').dt.date
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
    print("Starting Fuzzy Merge (Exact Match -> Then +/- 1 Day)...")

    # --- STEP 1: Exact Date Match ---
    merged_df = pd.merge(
        df_players, 
        df_odds, 
        how='left', 
        on=['join_date', 'join_home', 'join_away']
    )
    
    exact_match_count = merged_df['home_line_close'].notna().sum()
    print(f"   Exact Matches: {exact_match_count}")

    # --- STEP 2: Try Match with Date + 1 Day (Fixes games played 'yesterday' relative to odds) ---
    # Create a temp odds table shifted by 1 day
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

    # --- STEP 3: Try Match with Date - 1 Day (Fixes games played 'tomorrow' relative to odds) ---
    # Create a temp odds table shifted by -1 day
    df_odds_minus = df_odds.copy()
    df_odds_minus['join_date'] = df_odds_minus['join_date'] - timedelta(days=1)
    
    merged_df = pd.merge(
        merged_df,
        df_odds_minus,
        how='left',
        on=['join_date', 'join_home', 'join_away'],
        suffixes=('', '_minus')
    )
    
    # Fill in missing values
    for col in cols_to_fill:
        merged_df[col] = merged_df[col].fillna(merged_df[f"{col}_minus"])

    # --- CLEANUP ---
    # Drop all the extra columns created by the fuzzy merges
    cols_to_drop = [c for c in merged_df.columns if c.endswith('_plus') or c.endswith('_minus')]
    cols_to_drop.extend(['join_date', 'join_home', 'join_away'])
    merged_df.drop(columns=cols_to_drop, inplace=True)

    # --- VALIDATION ---
    final_matched = merged_df['home_line_close'].notna().sum()
    print(f"✅ Final Matched Rows: {final_matched} / {len(merged_df)}")
    
    # Validation Check for that specific Titans/Jags game (Week 18, 2022)
    check_game = merged_df[
        (merged_df['home_team'] == 'JAX') & 
        (merged_df['season'] == 2022) & 
        (merged_df['week'] == 18)
    ]
    
    if not check_game.empty:
        # Check the first row of that game
        val = check_game['home_line_close'].iloc[0]
        status = f"✅ Fixed! (Value: {val})" if pd.notna(val) else "❌ Still Null"
        print(f"Validation Check (TEN@JAX 2022 Week 18): {status}")

    # --- SAVE ---
    print(f"Saving to {OUTPUT_FILE}...")
    merged_df.to_parquet(OUTPUT_FILE)
    print("Done.")

if __name__ == "__main__":
    main()
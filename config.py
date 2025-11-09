# config.py
import os
from pathlib import Path
from datetime import datetime, timedelta

# Optional dotenv import (hardened)
try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    def load_dotenv(*args, **kwargs):  # fallback no-op if python-dotenv not installed
        return False

# Load environment variables (from .env if present)
load_dotenv()

# ----------------------------------------------------------------------------- 
# Project / storage paths (Phase 0 additions)
# -----------------------------------------------------------------------------
# Root of this repository (directory containing this file)
PROJECT_ROOT: Path = Path(__file__).resolve().parent

# Optional external DuckDB path (used in later phases); defaults to local DB
DUCKDB_PATH: Path = Path(os.getenv("DUCKDB_PATH", "databases/seamus.db")).expanduser()
# Table names in DuckDB
# Player-level joined table (from build_player_game.py). Defaults to "joined_games".
DUCKDB_JOINED_TABLE: str = os.getenv("DUCKDB_JOINED_TABLE", "joined_games")
# Games table (schedule/odds metadata). Defaults to "historical_games".
DUCKDB_GAMES_TABLE: str = os.getenv("DUCKDB_GAMES_TABLE", "historical_games")

# =============================================================================
# API CONFIGURATION
# =============================================================================

# API Keys (loaded from .env file)
OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY')
ODDS_API_KEY = os.getenv('ODDS_API_KEY')
MYSPORTSFEEDS_API_KEY = os.getenv('MYSPORTSFEEDS_API_KEY')
MYSPORTSFEEDS_PASSWORD = os.getenv('MYSPORTSFEEDS_PASSWORD')

# Kalshi API Configuration
KALSHI_API_KEY = os.getenv('KALSHI_API_KEY')
KALSHI_PRIVATE_KEY_PATH = os.getenv('KALSHI_PRIVATE_KEY_PATH', 'kalshi_private_key.pem')

# Load Kalshi private key if path exists
KALSHI_PRIVATE_KEY = None
if KALSHI_PRIVATE_KEY_PATH and os.path.exists(KALSHI_PRIVATE_KEY_PATH):
    try:
        with open(KALSHI_PRIVATE_KEY_PATH, 'r') as f:
            KALSHI_PRIVATE_KEY = f.read()
    except Exception as e:
        print(f"⚠️ Warning: Could not load Kalshi private key: {e}")

# API Rate Limits and Settings
API_RATE_LIMITS = {
    'espn': 1.0,  # seconds between requests
    'openweather': 0.1,  # can handle faster requests
    'odds_api': 2.0,  # be conservative with free tier
    'mysportsfeeds': 1.0,
    'kalshi': 0.5,  # Kalshi rate limiting
    'kalshi_ws': 0.1  # WebSocket messages
}

# API Base URLs
API_URLS = {
    'espn_scoreboard': 'https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard',
    'espn_roster': 'https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{team_id}/roster',  # NEW: For live week roster data
    'espn_player_stats': 'http://sports.core.api.espn.com/v2/sports/football/leagues/nfl/athletes/{player_id}/statistics',
    'espn_team_injuries': 'http://sports.core.api.espn.com/v2/sports/football/leagues/nfl/teams/{team_id}/injuries',
    'openweather': 'http://api.openweathermap.org/data/2.5/weather',
    'odds_api': 'https://api.the-odds-api.com/v4/sports/americanfootball_nfl',
    'odds_api_scores': 'https://api.the-odds-api.com/v4/sports/americanfootball_nfl/scores',
    'odds_api_events': 'https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events',
    'mysportsfeeds': 'https://api.mysportsfeeds.com/v2.1/pull/nfl',
    'kalshi_api': 'https://api.elections.kalshi.com',
    'kalshi_ws': 'wss://api.elections.kalshi.com/trade-api/ws/v2'
}

# Kalshi Market Configuration
KALSHI_CONFIG = {
    'enabled': bool(KALSHI_API_KEY and KALSHI_PRIVATE_KEY),
    'max_markets_to_track': 50,  # Maximum number of markets to track simultaneously
    'min_liquidity': 1000,  # Minimum market liquidity in dollars
    'max_spread': 0.10,  # Maximum bid-ask spread (10 cents)
    'prop_categories': [
        'NFL-PLAYER-PASSING',
        'NFL-PLAYER-RUSHING', 
        'NFL-PLAYER-RECEIVING',
        'NFL-PLAYER-SCORING',
        'NFL-PLAYER-DEFENSE'
    ]
}

# =============================================================================
# DATA CONFIGURATION
# =============================================================================

# Seasons and Time Periods
CURRENT_SEASON = 2025
HISTORICAL_YEARS = [2020, 2021, 2022, 2023, 2024]
PROP_ANALYSIS_WEEKS = 10  # How many weeks back to analyze for props

# Data Directories
DATA_DIR = 'data'
RAW_DATA_DIR = f'{DATA_DIR}/raw'
PROCESSED_DATA_DIR = f'{DATA_DIR}/processed'
MODEL_DIR = 'models'
LOG_DIR = 'logs'
KALSHI_DATA_DIR = f'{DATA_DIR}/kalshi'
DB_PATH = 'databases/seamus.db' # Central database for analytical data

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, LOG_DIR, KALSHI_DATA_DIR]:
    os.makedirs(directory, exist_ok=True)

# Data File Naming Conventions
DATA_FILES = {
    'weekly_stats': f'{PROCESSED_DATA_DIR}/weekly_stats_{{start_year}}_to_{{end_year}}.csv',
    'pbp_data': f'{PROCESSED_DATA_DIR}/pbp_data_{{year}}.parquet',
    'player_props': f'{PROCESSED_DATA_DIR}/player_props_{{date}}.csv',
    'odds_data': f'{RAW_DATA_DIR}/odds_{{date}}.json',
    'weather_data': f'{RAW_DATA_DIR}/weather_{{date}}.json',
    'injury_reports': f'{RAW_DATA_DIR}/injuries_{{date}}.json',
    'kalshi_markets': f'{KALSHI_DATA_DIR}/markets_{{date}}.json',
    'kalshi_orderbook': f'{KALSHI_DATA_DIR}/orderbook_{{market}}_{{date}}.json',
    'kalshi_trades': f'{KALSHI_DATA_DIR}/trades_{{date}}.csv'
}

# =============================================================================
# PROP BETTING CONFIGURATION
# =============================================================================

# Public betting thresholds (consumed by betting.py)
# These mirror MODEL_CONFIG defaults but are explicit for runtime checks.
# Minimum edge required to place a bet (decimal, e.g., 0.03 = 3%)
MIN_EDGE = 0.03
# Minimum Kelly fraction to place a bet (decimal, e.g., 0.25 = 25% of Kelly)
MIN_KELLY_FRACTION = 0.25
# Maximum bet size as fraction of bankroll
MAX_BET_SIZE = 0.05
# Threshold to label bets as “strong” for reporting
STRONG_EDGE_THRESHOLD = 0.05

# Prop Types to Track (Traditional and Kalshi)
PROP_TYPES = {
    'receiving': ['receiving_yards', 'receptions', 'receiving_tds', 'receiving_yards_ou'],
    'rushing': ['rushing_yards', 'carries', 'rushing_tds', 'rushing_yards_ou'],
    'passing': ['passing_yards', 'completions', 'passing_tds', 'interceptions', 'passing_yards_ou'],
    'kicking': ['field_goals_made', 'extra_points_made'],
    'defense': ['sacks', 'interceptions', 'fumble_recoveries'],
    'scoring': ['anytime_td', 'first_td', 'player_to_score']
}

# Positions and Their Primary Props
POSITION_PROPS = {
    'QB': ['passing_yards', 'passing_tds', 'completions', 'interceptions', 'rushing_yards'],
    'RB': ['rushing_yards', 'carries', 'rushing_tds', 'receptions', 'receiving_yards', 'anytime_td'],
    'WR': ['receiving_yards', 'receptions', 'receiving_tds', 'targets', 'anytime_td'],
    'TE': ['receiving_yards', 'receptions', 'receiving_tds', 'targets', 'anytime_td'],
    'K': ['field_goals_made', 'extra_points_made'],
    'DEF': ['sacks', 'interceptions', 'fumble_recoveries']
}

# Positions to include in live week roster (relevant for betting)
RELEVANT_POSITIONS = ['QB', 'RB', 'WR', 'TE', 'K']  # NEW: Filter for positions we model

# Model Parameters
MODEL_CONFIG = {
    'min_sample_size': 5,  # Minimum games to analyze
    'confidence_threshold': 0.55,  # Minimum probability for recommendations
    'kelly_fraction': 0.25,  # Fraction of Kelly to bet (conservative)
    'max_bet_size': 0.05,  # Max 5% of bankroll per bet
    'prop_edge_threshold': 0.03,  # Minimum 3% edge to bet
    'kalshi_fee': 0.007,  # Kalshi's 0.7% fee on profits
    'min_kalshi_edge': 0.04  # Minimum edge for Kalshi bets (accounts for fees)
}

# =============================================================================
# NFL TEAM CONFIGURATION
# =============================================================================

# NFL Team Mappings
NFL_TEAMS = {
    'ARI': {'name': 'Arizona Cardinals', 'espn_id': '22', 'city': 'Phoenix'},
    'ATL': {'name': 'Atlanta Falcons', 'espn_id': '1', 'city': 'Atlanta'},
    'BAL': {'name': 'Baltimore Ravens', 'espn_id': '33', 'city': 'Baltimore'},
    'BUF': {'name': 'Buffalo Bills', 'espn_id': '2', 'city': 'Buffalo'},
    'CAR': {'name': 'Carolina Panthers', 'espn_id': '29', 'city': 'Charlotte'},
    'CHI': {'name': 'Chicago Bears', 'espn_id': '3', 'city': 'Chicago'},
    'CIN': {'name': 'Cincinnati Bengals', 'espn_id': '4', 'city': 'Cincinnati'},
    'CLE': {'name': 'Cleveland Browns', 'espn_id': '5', 'city': 'Cleveland'},
    'DAL': {'name': 'Dallas Cowboys', 'espn_id': '6', 'city': 'Dallas'},
    'DEN': {'name': 'Denver Broncos', 'espn_id': '7', 'city': 'Denver'},
    'DET': {'name': 'Detroit Lions', 'espn_id': '8', 'city': 'Detroit'},
    'GB': {'name': 'Green Bay Packers', 'espn_id': '9', 'city': 'Green Bay'},
    'HOU': {'name': 'Houston Texans', 'espn_id': '34', 'city': 'Houston'},
    'IND': {'name': 'Indianapolis Colts', 'espn_id': '11', 'city': 'Indianapolis'},
    'JAX': {'name': 'Jacksonville Jaguars', 'espn_id': '30', 'city': 'Jacksonville'},
    'KC': {'name': 'Kansas City Chiefs', 'espn_id': '12', 'city': 'Kansas City'},
    'LV': {'name': 'Las Vegas Raiders', 'espn_id': '13', 'city': 'Las Vegas'},
    'LAC': {'name': 'Los Angeles Chargers', 'espn_id': '24', 'city': 'Los Angeles'},
    'LAR': {'name': 'Los Angeles Rams', 'espn_id': '14', 'city': 'Los Angeles'},
    'MIA': {'name': 'Miami Dolphins', 'espn_id': '15', 'city': 'Miami'},
    'MIN': {'name': 'Minnesota Vikings', 'espn_id': '16', 'city': 'Minneapolis'},
    'NE': {'name': 'New England Patriots', 'espn_id': '17', 'city': 'Boston'},
    'NO': {'name': 'New Orleans Saints', 'espn_id': '18', 'city': 'New Orleans'},
    'NYG': {'name': 'New York Giants', 'espn_id': '19', 'city': 'New York'},
    'NYJ': {'name': 'New York Jets', 'espn_id': '20', 'city': 'New York'},
    'PHI': {'name': 'Philadelphia Eagles', 'espn_id': '21', 'city': 'Philadelphia'},
    'PIT': {'name': 'Pittsburgh Steelers', 'espn_id': '23', 'city': 'Pittsburgh'},
    'SF': {'name': 'San Francisco 49ers', 'espn_id': '25', 'city': 'San Francisco'},
    'SEA': {'name': 'Seattle Seahawks', 'espn_id': '26', 'city': 'Seattle'},
    'TB': {'name': 'Tampa Bay Buccaneers', 'espn_id': '27', 'city': 'Tampa'},
    'TEN': {'name': 'Tennessee Titans', 'espn_id': '10', 'city': 'Nashville'},
    'WAS': {'name': 'Washington Commanders', 'espn_id': '28', 'city': 'Washington'}
}

# Team abbreviation aliases - ESPN sometimes uses different abbreviations
TEAM_ABBREVIATION_ALIASES = {
    'WSH': 'WAS',  # ESPN uses WSH, we use WAS for Washington Commanders
    'LA': 'LAR',   # ESPN sometimes uses LA for Rams
}

def normalize_team_abbr(team_abbr: str) -> str:
    """
    Normalize team abbreviation to match our config.
    Handles ESPN's different abbreviations (e.g., WSH -> WAS).
    
    Args:
        team_abbr: Team abbreviation from ESPN or other source
        
    Returns:
        Normalized team abbreviation that exists in NFL_TEAMS
    """
    if not team_abbr:
        return team_abbr
    
    # Check if it's an alias
    if team_abbr in TEAM_ABBREVIATION_ALIASES:
        return TEAM_ABBREVIATION_ALIASES[team_abbr]
    
    # Return as-is if it's already valid
    return team_abbr

# Stadium Information (for weather)
STADIUM_INFO = {
    'KC': {'name': 'Arrowhead Stadium', 'dome': False, 'city': 'Kansas City'},
    'GB': {'name': 'Lambeau Field', 'dome': False, 'city': 'Green Bay'},
    'NO': {'name': 'Caesars Superdome', 'dome': True, 'city': 'New Orleans'},
    'DET': {'name': 'Ford Field', 'dome': True, 'city': 'Detroit'},
    'MIN': {'name': 'U.S. Bank Stadium', 'dome': True, 'city': 'Minneapolis'},
    # Add more as needed
}

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
        'file': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.FileHandler',
            'filename': f'{LOG_DIR}/nfl_props.log',
        },
    },
    'loggers': {
        '': {
            'handlers': ['default', 'file'],
            'level': 'INFO',
            'propagate': False
        }
    }
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_current_week():
    """Get current NFL week"""
    # NFL season starts around week 36 of the year
    # This is a simplified calculation
    now = datetime.now()
    if now.month >= 9:  # September start
        week = ((now - datetime(now.year, 9, 1)).days // 7) + 1
        return min(week, 18)  # Max 18 weeks
    elif now.month <= 2:  # Playoffs/Super Bowl
        return 18 + ((now - datetime(now.year, 1, 1)).days // 7)
    else:
        return 1  # Off-season

def get_team_espn_id(team_abbr):
    """Get ESPN team ID from abbreviation"""
    return NFL_TEAMS.get(team_abbr, {}).get('espn_id')

def get_team_city(team_abbr):
    """Get team city for weather lookup"""
    return NFL_TEAMS.get(team_abbr, {}).get('city')

def is_dome_team(team_abbr):
    """Check if team plays in a dome"""
    return STADIUM_INFO.get(team_abbr, {}).get('dome', False)

def is_kalshi_enabled():
    """Check if Kalshi integration is enabled"""
    return KALSHI_CONFIG.get('enabled', False)

def get_espn_team_abbr_from_id(espn_id: str) -> str:
    """
    Get team abbreviation from ESPN ID (reverse lookup).
    Useful when parsing ESPN API responses.
    
    Args:
        espn_id: ESPN team ID (e.g., '12' for Kansas City)
        
    Returns:
        Team abbreviation (e.g., 'KC') or None if not found
    """
    for abbr, team_info in NFL_TEAMS.items():
        if team_info.get('espn_id') == str(espn_id):
            return abbr
    return None

# Current week for convenience
CURRENT_WEEK = get_current_week()

print(f"🏈 Configuration loaded for NFL Season {CURRENT_SEASON}, Week {CURRENT_WEEK}")
if is_kalshi_enabled():
    print("✅ Kalshi integration enabled")
else:
    print("⚠️ Kalshi integration disabled (missing API key or private key)")



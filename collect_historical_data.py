# collect_historical_data.py
"""
ONE-TIME SCRIPT: Backfills the analytical database with historical game
and weather data.

This script performs two main tasks:
1.  Fetches all historical game schedules (2020-2024) from nfl-data-py,
    joins them with the 'stadiums' table to get stadium_id, and saves
    the results (including betting spreads) to a new 'historical_games' table.
    
2.  Queries the 'historical_games' table for all outdoor games, then
    calls the FREE Open-Meteo Historical API to get the precise
    weather at kickoff for every single game. This data is saved to a
    new 'historical_weather' table.

This script is designed to be run once and is resumable. If it's stopped,
it will pick up where it left off on the next run.
"""

import duckdb
import os
import nfl_data_py as nfl
import pandas as pd
import requests
import time
import logging
from tqdm import tqdm
from datetime import datetime

# Import config from your existing file
try:
    # We no longer need the OPENWEATHER_API_KEY
    from config import DB_PATH, HISTORICAL_YEARS
except ImportError:
    print("❌ Error: Could not import DB_PATH or HISTORICAL_YEARS from config.py.")
    print("Please ensure config.py is in the same directory and those variables are set.")
    exit()

# NEW: Open-Meteo Historical Weather API URL
OPENMETEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# Database and Table Creation
# =============================================================================

def create_tables(con: duckdb.DuckDBPyConnection):
    """
    Creates the 'historical_games' and 'historical_weather' tables if they
    don't already exist.
    """
    logger.info("Checking and creating tables...")
    
    # Table for game schedules, spreads, and stadium context
    con.execute("""
        CREATE TABLE IF NOT EXISTS historical_games (
            game_id VARCHAR PRIMARY KEY,
            season INTEGER,
            week INTEGER,
            gameday TIMESTAMP,
            home_team VARCHAR,
            away_team VARCHAR,
            stadium_id INTEGER,
            roof VARCHAR,
            spread_line FLOAT,
            total_line FLOAT
        )
    """)
    
    # Table for historical weather data, linked to games
    con.execute("""
        CREATE TABLE IF NOT EXISTS historical_weather (
            game_id VARCHAR PRIMARY KEY,
            game_timestamp TIMESTAMP,
            temperature_f FLOAT,
            apparent_temp_f FLOAT,
            wind_speed_mph FLOAT,
            wind_gust_mph FLOAT,
            wind_direction_deg INTEGER,
            humidity_pct FLOAT,
            precipitation_inches FLOAT,
            cloud_cover_pct FLOAT,
            weather_summary VARCHAR,
            FOREIGN KEY (game_id) REFERENCES historical_games(game_id)
        )
    """)
    logger.info("✅ Tables 'historical_games' and 'historical_weather' are ready.")

# =============================================================================
# Step 1: Backfill Historical Games
# =============================================================================

def backfill_historical_games(con: duckdb.DuckDBPyConnection):
    """
    Fetches schedule data from nfl-data-py for all historical years,
    joins with the 'stadiums' table, and inserts into 'historical_games'.
    """
    logger.info("Starting Step 1: Backfilling 'historical_games'...")
    
    # Check if table is already populated
    try:
        count = con.execute("SELECT COUNT(*) FROM historical_games").fetchone()[0]
        if count > 0:
            logger.info(f"✅ 'historical_games' already contains {count} records. Skipping Step 1.")
            return
    except duckdb.CatalogException:
        logger.error("Error checking historical_games count. Has 'complete_stadium_data.py' been run?")
        raise

    logger.info(f"Fetching schedules for {HISTORICAL_YEARS} from nfl-data-py...")
    sched_df = nfl.import_schedules(years=HISTORICAL_YEARS)
    
    sched_df = sched_df[sched_df['game_type'].isin(['REG', 'WC', 'DIV', 'CON', 'SB'])]
    
    sched_df = sched_df[[
        'game_id', 'season', 'week', 'gameday', 'home_team', 'away_team', 
        'roof', 'spread_line', 'total_line'
    ]].copy()
    
    sched_df['gameday'] = pd.to_datetime(sched_df['gameday'])
    
    logger.info(f"Found {len(sched_df)} total games.")
    
    con.register('sched_df_temp', sched_df)
    
    logger.info("Joining schedules with 'stadiums' table to get stadium_id...")
    
    insert_query = """
    INSERT INTO historical_games (
        game_id, season, week, gameday, home_team, away_team, 
        stadium_id, roof, spread_line, total_line
    )
    SELECT 
        sched.game_id,
        sched.season,
        sched.week,
        sched.gameday,
        sched.home_team,
        sched.away_team,
        s.stadium_id,
        sched.roof,
        sched.spread_line,
        sched.total_line
    FROM sched_df_temp AS sched
    JOIN stadiums AS s 
      ON s.team = sched.home_team 
     AND sched.gameday::date >= s.effective_from 
     AND (sched.gameday::date < s.effective_to OR s.effective_to IS NULL)
    ON CONFLICT (game_id) DO NOTHING
    """
    
    con.execute(insert_query)
    
    count = con.execute("SELECT COUNT(*) FROM historical_games").fetchone()[0]
    logger.info(f"✅ Step 1 Complete: 'historical_games' now contains {count} records.")

# =============================================================================
# Step 2: Backfill Historical Weather
# =============================================================================

def get_games_needing_weather(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Finds all outdoor games that do not yet have an entry in the
    'historical_weather' table.
    """
    logger.info("Finding games that need weather data...")
    query = """
    SELECT 
        g.game_id,
        g.gameday,
        s.latitude,
        s.longitude
    FROM historical_games AS g
    JOIN stadiums AS s ON g.stadium_id = s.stadium_id
    WHERE 
        g.roof = 'outdoors'
    AND g.game_id NOT IN (SELECT game_id FROM historical_weather)
    ORDER BY
        g.gameday
    """
    games_df = con.execute(query).df()
    logger.info(f"Found {len(games_df)} outdoor games missing weather data.")
    return games_df

def parse_openmeteo_response(data: dict, gameday: datetime) -> dict:
    """
    Parses the Open-Meteo JSON response to find the single hour of data
    we need (at kickoff) and converts units to Imperial.
    """
    try:
        hourly_data = data['hourly']
        
        # Find the index for the kickoff hour.
        # Open-Meteo returns 24 hourly values (0-23).
        hour_index = gameday.hour
        
        # Helper for safe unit conversion
        def safe_convert(value, func):
            return func(value) if value is not None else None

        # Conversion functions
        c_to_f = lambda c: (c * 9/5) + 32
        kmh_to_mph = lambda kmh: kmh * 0.621371
        mm_to_in = lambda mm: mm / 25.4

        # Extract data for the specific kickoff hour
        temp_c = hourly_data['temperature_2m'][hour_index]
        feels_c = hourly_data['apparent_temperature'][hour_index]
        wind_kmh = hourly_data['wind_speed_10m'][hour_index]
        gust_kmh = hourly_data['wind_gusts_10m'][hour_index]
        precip_mm = hourly_data['precipitation'][hour_index]

        return {
            'temp_f': safe_convert(temp_c, c_to_f),
            'feels_f': safe_convert(feels_c, c_to_f),
            'wind_mph': safe_convert(wind_kmh, kmh_to_mph),
            'wind_gust_mph': safe_convert(gust_kmh, kmh_to_mph),
            'wind_deg': hourly_data['wind_direction_10m'][hour_index],
            'humidity': hourly_data['relative_humidity_2m'][hour_index],
            'precip_in': safe_convert(precip_mm, mm_to_in),
            'clouds': hourly_data['cloud_cover'][hour_index],
            'summary': 'N/A' # Open-Meteo doesn't provide a text summary
        }
    except Exception as e:
        logger.error(f"Error parsing Open-Meteo data: {e} - Data: {data}")
        return None

def fetch_and_insert_weather(con: duckdb.DuckDBPyConnection, games_df: pd.DataFrame):
    """
    Loops through games, fetches weather from Open-Meteo, and inserts into DB.
    """
    if games_df.empty:
        logger.info("No games to fetch weather for. Skipping.")
        return

    logger.info(f"Starting Step 2: Fetching weather for {len(games_df)} games using Open-Meteo...")

    for _, row in tqdm(games_df.iterrows(), total=len(games_df), desc="Fetching Weather"):
        game_id = row['game_id']
        gameday = row['gameday']
        game_date_str = gameday.strftime('%Y-%m-%d')

        params = {
            'latitude': row['latitude'],
            'longitude': row['longitude'],
            'start_date': game_date_str,
            'end_date': game_date_str,
            'hourly': 'temperature_2m,apparent_temperature,precipitation,relative_humidity_2m,cloud_cover,wind_speed_10m,wind_gusts_10m,wind_direction_10m',
            'timezone': 'auto' # Automatically adjust to stadium's local time
        }
        
        try:
            response = requests.get(OPENMETEO_ARCHIVE_URL, params=params, timeout=10)
            
            # No API key, but good to be polite
            time.sleep(0.1) 

            if response.status_code == 200:
                data = response.json()
                
                clean_data = parse_openmeteo_response(data, gameday)
                
                if clean_data:
                    # Insert into the database
                    con.execute(
                        "INSERT INTO historical_weather VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        [
                            game_id,
                            gameday,
                            clean_data['temp_f'],
                            clean_data['feels_f'],
                            clean_data['wind_mph'],
                            clean_data['wind_gust_mph'],
                            clean_data['wind_deg'],
                            clean_data['humidity'],
                            clean_data['precip_in'],
                            clean_data['clouds'],
                            clean_data['summary'],
                        ],
                    )
                else:
                    logger.warning(f"Could not parse weather data for game {game_id}")
                    
            else:
                logger.error(f"Error for game {game_id}: {response.status_code} - {response.text}")

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for game {game_id}: {e}")
            time.sleep(5) # Wait before retrying next game
        
    logger.info("✅ Step 2 Complete: Weather fetching finished.")

# =============================================================================
# Main Execution
# =============================================================================

def main():
    logger.info("🚀 Starting historical data backfill process (with Open-Meteo)...")
    
    db_file = DB_PATH
    
    if not os.path.exists(db_file):
        # Check the directory path from config
        db_dir = os.path.dirname(db_file)
        if not os.path.exists(db_dir):
             logger.warning(f"Database directory '{db_dir}' not found. Creating it.")
             os.makedirs(db_dir, exist_ok=True)
             
        logger.error(f"❌ Database not found at '{db_file}'.")
        logger.error("Please run 'python complete_stadium_data.py' first.")
        
        # Check if the user ran complete_stadium_data.py but it failed
        if not os.path.exists('complete_stadium_data.py'):
            logger.error("File 'complete_stadium_data.py' not found.")
            return
        else:
            logger.info("Attempting to run 'python complete_stadium_data.py' to create the database...")
            try:
                os.system('python complete_stadium_data.py')
                if not os.path.exists(db_file):
                    logger.error("Failed to create database with 'complete_stadium_data.py'. Please check that script for errors.")
                    return
                logger.info("Database created. Proceeding with backfill...")
            except Exception as e:
                logger.error(f"Failed to run 'complete_stadium_data.py': {e}")
                return

    con = None
    try:
        con = duckdb.connect(db_file)
        
        # Step 0: Create tables if they don't exist
        create_tables(con)
        
        # Step 1: Fill historical_games table
        backfill_historical_games(con)
        
        # Step 2: Get list of games needing weather
        games_to_fetch_df = get_games_needing_weather(con)
        
        # Step 3: Fetch and insert weather data
        fetch_and_insert_weather(con, games_to_fetch_df)
        
        logger.info("🎉 Historical data backfill complete.")
        
        # Final summary
        game_count = con.execute("SELECT COUNT(*) FROM historical_games").fetchone()[0]
        weather_count = con.execute("SELECT COUNT(*) FROM historical_weather").fetchone()[0]
        logger.info(f"Database now contains {game_count} games and {weather_count} weather records.")
        
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
    finally:
        if con:
            con.close()

if __name__ == "__main__":
    main()
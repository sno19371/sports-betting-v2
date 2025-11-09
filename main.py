#!/usr/bin/env python3
# main.py
"""
Main orchestrator for NFL player prop betting system.
Executes the complete workflow from data fetching to bet recommendations.

Usage:
    python main.py --season 2024 --week 5
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import nfl_data_py as nfl
import duckdb

from config import CURRENT_SEASON, CURRENT_WEEK, HISTORICAL_YEARS, MODEL_DIR, DB_PATH
from data_sources import NFLDataSources
from feature_engineering import FeatureEngineer
from modeling import ModelManager
from betting import BetFinder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/main.log')
    ]
)
logger = logging.getLogger(__name__)


def load_schedule_from_db(years: list[int]) -> pd.DataFrame | None:
    """
    Load game-level context from DuckDB (stadiums + games + weather),
    aligned with the training pipeline.
    Returns None on failure.
    """
    query = f"""
    SELECT
        g.game_id,
        g.season,
        g.week,
        g.gameday,
        g.home_team,
        g.away_team,
        g.spread_line,
        g.total_line,
        s.roof_type AS roof,
        s.surface,
        s.elevation_feet,
        w.temperature_f AS temp,
        w.apparent_temp_f,
        w.wind_speed_mph AS wind,
        w.wind_gust_mph,
        w.precipitation_inches,
        w.cloud_cover_pct,
        w.humidity_pct
    FROM historical_games AS g
    JOIN stadiums AS s 
      ON g.stadium_id = s.stadium_id
     AND g.gameday::date >= s.effective_from
     AND (g.gameday::date < s.effective_to OR s.effective_to IS NULL)
    LEFT JOIN historical_weather AS w
      ON g.game_id = w.game_id
    WHERE g.season IN ({', '.join(str(y) for y in years)})
    """
    try:
        con = duckdb.connect(DB_PATH, read_only=True)
        try:
            df = con.execute(query).df()
        finally:
            con.close()
        if df is None or df.empty:
            logger.warning("Schedule query returned no rows from DuckDB")
            return None
        return df
    except Exception as e:
        logger.warning(f"Failed to load schedule from DuckDB ({DB_PATH}): {e}")
        return None

def run_workflow(season: int, week: int) -> None:
    """
    Execute the complete betting workflow for a given season and week.
    
    Workflow steps:
    1. Fetch historical player data (the "library")
    2. Fetch live week roster data (the "newspaper")
    3. Combine and engineer features from both datasets
    4. Load pre-trained model
    5. Generate predictions for target week
    6. Fetch current market odds
    7. Identify +EV betting opportunities
    8. Save recommendations to log file
    
    Args:
        season: NFL season year (e.g., 2024)
        week: NFL week number (1-18)
    """
    print("=" * 80)
    print(f"NFL PROP BETTING SYSTEM - {season} SEASON, WEEK {week}")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    try:
        # =====================================================================
        # STEP 1: INITIALIZE DATA SOURCES
        # =====================================================================
        logger.info("Step 1: Initializing data sources")
        data_api = NFLDataSources()
        print("\n✓ Data sources initialized")
        
        # =====================================================================
        # STEP 2: FETCHING DATA (Library + Newspaper Strategy)
        # =====================================================================
        logger.info("Step 2: Fetching data")
        print("\n" + "=" * 80)
        print("FETCHING DATA")
        print("=" * 80)
        
        # Strategy: "Library" + "Newspaper" approach
        # Library: Historical data for lookback features (rolling averages, etc.)
        # Newspaper: Live roster data for who's playing this upcoming week
        
        # 1. Fetch HISTORICAL data (the "library")
        logger.info(f"Fetching historical data for years: {HISTORICAL_YEARS}")
        print(f"\n📚 Fetching historical data (the 'library')...")
        print(f"   Years: {HISTORICAL_YEARS}")
        
        historical_df = data_api.get_weekly_stats(years=HISTORICAL_YEARS, save=False)
        
        if historical_df is None or historical_df.empty:
            raise ValueError("Failed to fetch historical weekly player data")
        
        print(f"✓ Loaded {len(historical_df):,} historical player-week records")
        print(f"  Seasons: {historical_df['season'].min()} - {historical_df['season'].max()}")
        print(f"  Unique players: {historical_df['player_id'].nunique():,}")
        
        # 2. Fetch LIVE WEEK data (the "newspaper")
        logger.info(f"Fetching live roster data for Season {season}, Week {week}")
        print(f"\n🗞️  Fetching live week data (the 'newspaper')...")
        print(f"   Season: {season}, Week: {week}")
        
        live_week_df = data_api.get_live_week_data_espn(season=season, week=week)
        
        if live_week_df is None or live_week_df.empty:
            raise ValueError(
                f"Failed to fetch live roster data for Season {season}, Week {week}.\n"
                f"Possible reasons:\n"
                f"  • No games scheduled for this week (bye week, playoffs)\n"
                f"  • ESPN API is unavailable\n"
                f"  • Week/season parameters are invalid"
            )
        
        print(f"✓ Loaded {len(live_week_df):,} players for upcoming week")
        print(f"  Teams playing: {live_week_df['team'].nunique()}")
        print(f"  Games scheduled: {len(live_week_df['team'].unique()) // 2}")
        print(f"  Position breakdown: {live_week_df['position'].value_counts().to_dict()}")
        
        # 3. Combine historical + live week data
        logger.info("Combining historical and live week data")
        print(f"\n🔗 Combining datasets...")
        
        weekly_df = pd.concat([historical_df, live_week_df], ignore_index=True)
        
        print(f"✓ Combined dataset: {len(weekly_df):,} total records")
        print(f"  Historical records: {len(historical_df):,}")
        print(f"  Live week records: {len(live_week_df):,}")
        
        # 4. Fetch schedule data for all relevant years (from DuckDB to match training)
        print(f"\n📅 Fetching schedule data (from DuckDB)...")
        all_years = sorted(list(set(HISTORICAL_YEARS + [season])))
        schedule_df = load_schedule_from_db(all_years)
        if schedule_df is None:
            print("⚠️  Schedule data unavailable from DuckDB; falling back to nfl-data-py")
            try:
                schedule_df = nfl.import_schedules(years=all_years)
                logger.info(f"Loaded schedule data via nfl-data-py for years {min(all_years)}-{max(all_years)}")
                print(f"✓ Loaded {len(schedule_df):,} games for schedule data (fallback)")
            except Exception as e:
                logger.warning(f"Failed to load schedule data via fallback: {e}")
                logger.info("Continuing without schedule data (limited features)")
                print(f"⚠️  Schedule data unavailable: {e}")
                print("   Continuing with player stats only (limited features)")
                schedule_df = None
        
        # =====================================================================
        # STEP 3: ENGINEER FEATURES
        # =====================================================================
        logger.info("Step 3: Engineering features")
        print("\n" + "=" * 80)
        print("FEATURE ENGINEERING")
        print("=" * 80)
        
        engineer = FeatureEngineer()
        
        print("\n🔧 Generating features (this may take several minutes)...")
        features_df = engineer.create_features(
            weekly_df=weekly_df,
            schedule_df=schedule_df,
            include_rolling=True,
            include_ewma=True,
            include_matchup=True,
            include_weather=True,
            include_game_script=True
        )
        
        print(f"✓ Feature engineering complete")
        print(f"  Total rows: {len(features_df):,}")
        print(f"  Total columns: {features_df.shape[1]}")
        
        # Filter to target week for prediction
        target_mask = (features_df['season'] == season) & (features_df['week'] == week)
        current_week_features = features_df[target_mask].copy()
        
        if current_week_features.empty:
            raise ValueError(f"No data found for Season {season}, Week {week} after feature engineering")
        
        logger.info(f"Target week data: {len(current_week_features)} player-game records")
        print(f"✓ Filtered to Week {week}: {len(current_week_features)} player-game records")
        
        # =====================================================================
        # STEP 4: LOAD PRE-TRAINED MODEL
        # =====================================================================
        logger.info("Step 4: Loading pre-trained model")
        print("\n" + "=" * 80)
        print("LOADING MODEL")
        print("=" * 80)
        
        model_path = Path(MODEL_DIR) / "model_manager.pkl"
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                "Please train a model first using: python train.py"
            )
        
        model_manager = ModelManager.load_manager(str(model_path))
        print(f"✓ Model loaded from: {model_path}")
        print(f"  Props in model: {model_manager.target_props}")
        print(f"  Quantiles: {model_manager.quantiles}")
        
        # Check if model has feature names
        if hasattr(model_manager, 'feature_names') and model_manager.feature_names:
            print(f"  Features: {len(model_manager.feature_names)} (saved with model)")
        else:
            print(f"  ⚠️  Model missing feature names (older version)")
        
        # =====================================================================
        # STEP 5: GENERATE PREDICTIONS
        # =====================================================================
        logger.info("Step 5: Generating predictions")
        print("\n" + "=" * 80)
        print("GENERATING PREDICTIONS")
        print("=" * 80)
        
        # Use feature names from model if available (guarantees exact match with training)
        if hasattr(model_manager, 'feature_names') and model_manager.feature_names:
            logger.info("Using feature names saved with model")
            feature_cols = model_manager.feature_names
            
            # Check if all required features are present
            missing_features = set(feature_cols) - set(current_week_features.columns)
            if missing_features:
                raise ValueError(
                    f"Missing required features: {missing_features}\n"
                    "The current data is missing features that the model was trained on."
                )
            
            X_predict = current_week_features[feature_cols]
            # Ensure dtypes compatible with LightGBM
            if 'is_home' in X_predict.columns and X_predict['is_home'].dtype == 'O':
                X_predict['is_home'] = X_predict['is_home'].fillna(False).astype(bool)
            # Convert any remaining object columns that are numeric-like
            for col in X_predict.select_dtypes(include=['object']).columns:
                X_predict[col] = pd.to_numeric(X_predict[col], errors='ignore')
            print(f"✓ Using {len(feature_cols)} features from model definition")
            
        else:
            # Fallback: manually determine features (less robust)
            logger.warning("Model doesn't have saved feature names, using fallback method")
            print("⚠️  Using fallback feature selection (consider retraining model)")
            
            metadata_cols = ['player_id', 'player_display_name', 'player_name', 
                            'season', 'week', 'team', 'opponent', 'position']
            target_cols = model_manager.target_props
            
            # Get feature columns (exclude metadata and targets)
            feature_cols = [col for col in current_week_features.columns 
                           if col not in metadata_cols and col not in target_cols 
                           and not col.startswith('matchup_') 
                           and not col.endswith('_actual')]
            
            X_predict = current_week_features[feature_cols]
            print(f"✓ Selected {len(feature_cols)} features using exclusion method")
        
        # Generate predictions
        predictions = model_manager.predict_all(X_predict)
        
        # Add player metadata to predictions
        predictions['player_name'] = current_week_features['player_display_name'].values
        
        logger.info(f"Generated predictions for {len(predictions)} players")
        print(f"✓ Predictions generated for {len(predictions)} players")
        print(f"  Prediction columns: {predictions.shape[1]}")
        
        # Show sample predictions
        print("\n📊 Sample predictions (first 3 players):")
        sample_cols = ['player_name'] + [col for col in predictions.columns if 'q0.5' in col][:3]
        if all(col in predictions.columns for col in sample_cols):
            print(predictions[sample_cols].head(3).to_string(index=False))
        else:
            print("  (Sample display unavailable - missing median columns)")
        
        # Persist predictions and live features for diagnostics
        try:
            Path('results').mkdir(parents=True, exist_ok=True)
            pred_out = f'results/predictions_{season}_week_{week}.csv'
            predictions.to_csv(pred_out, index=False)
            logger.info(f"Saved predictions to {pred_out}")
            # Save live features (full row context) to inspect spread/weather presence
            live_feat_out = 'results/live_features_latest.parquet'
            current_week_features.to_parquet(live_feat_out, index=False)
            logger.info(f"Saved live features for diagnostics: {live_feat_out}")
        except Exception as e:
            logger.warning(f"Failed to save predictions or live features: {e}")
        
        # =====================================================================
        # STEP 6: FETCH MARKET ODDS
        # =====================================================================
        logger.info("Step 6: Fetching market odds")
        print("\n" + "=" * 80)
        print("FETCHING MARKET ODDS")
        print("=" * 80)
        
        market_odds = data_api.get_player_props_odds()
        
        if market_odds is None or len(market_odds) == 0:
            logger.warning("No market odds available from API")
            print("\n⚠️  WARNING: No market odds available from API")
            print("This could be because:")
            print("  1. The Odds API free tier doesn't include player props")
            print("  2. API key is not configured")
            print("  3. No games scheduled for this week")
            print("\nUsing sample odds for demonstration...")
            
            # Create sample odds for demonstration
            market_odds = create_sample_odds(predictions)
        else:
            logger.info(f"Fetched {len(market_odds)} market lines")
            print(f"✓ Fetched {len(market_odds)} market lines from API")
        
        # =====================================================================
        # STEP 7: FIND +EV BETS
        # =====================================================================
        logger.info("Step 7: Identifying +EV betting opportunities")
        print("\n" + "=" * 80)
        print("FINDING +EV BETS")
        print("=" * 80)
        
        bet_finder = BetFinder(
            model_predictions=predictions,
            market_odds=market_odds
        )
        
        recommended_bets = bet_finder.find_bets()
        
        # =====================================================================
        # STEP 8: SAVE AND DISPLAY RESULTS
        # =====================================================================
        logger.info("Step 8: Saving results")
        print("\n" + "=" * 80)
        print("RESULTS")
        print("=" * 80)
        
        if recommended_bets:
            print(f"\n✓ Found {len(recommended_bets)} +EV betting opportunities")
            
            # Save to log file
            bet_finder.save_log()
            
            # Display summary
            bets_df = pd.DataFrame(recommended_bets)
            
            print("\n" + "-" * 80)
            print("RECOMMENDED BETS")
            print("-" * 80)
            
            for i, bet in enumerate(recommended_bets, 1):
                print(f"\nBet #{i}:")
                print(f"  Player: {bet['player_name']}")
                print(f"  Prop: {bet['prop_type']} {bet['side']} {bet['line']}")
                print(f"  Odds: {bet['american_odds']:+d} ({bet['decimal_odds']:.3f})")
                print(f"  Model Probability: {bet['model_probability']:.1%}")
                print(f"  Edge: {bet['edge']:+.1%}")
                print(f"  Expected Value: ${bet['expected_value']:.2f} per $1")
                print(f"  Recommended Stake: {bet['recommended_stake_pct']}")
            
            # Summary statistics
            print("\n" + "-" * 80)
            print("SUMMARY")
            print("-" * 80)
            print(f"Total bets: {len(bets_df)}")
            print(f"Average edge: {bets_df['edge'].mean():.2%}")
            print(f"Total expected value: ${bets_df['expected_value'].sum():.2f}")
            print(f"Total Kelly allocation: {bets_df['kelly_fraction'].sum():.2%}")
            
        else:
            print("\n⚠️  No bets found meeting criteria")
            print("This could mean:")
            print("  1. No significant edges detected this week")
            print("  2. Edge threshold too high (check config.py)")
            print("  3. Market odds are efficient")
        
        # =====================================================================
        # WORKFLOW COMPLETE
        # =====================================================================
        print("\n" + "=" * 80)
        print("WORKFLOW COMPLETE")
        print("=" * 80)
        print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        logger.info("Workflow completed successfully")
        
    except Exception as e:
        logger.error(f"Workflow failed: {e}", exc_info=True)
        print(f"\n❌ ERROR: {e}")
        raise


def create_sample_odds(predictions: pd.DataFrame) -> list:
    """
    Create sample market odds for demonstration when API odds unavailable.
    
    Args:
        predictions: DataFrame with player predictions
        
    Returns:
        List of market odds dictionaries
    """
    sample_odds = []
    
    # Take first 5 players with receiving yards predictions
    receiving_cols = [col for col in predictions.columns if 'receiving_yards_q0.5' in col]
    if receiving_cols:
        for idx, row in predictions.head(5).iterrows():
            median_pred = row[receiving_cols[0]]
            
            # Create lines slightly above and below median
            sample_odds.append({
                'player_name': row['player_name'],
                'prop_type': 'receiving_yards',
                'line': float(median_pred - 10),  # Easier line
                'over_odds': -110,
                'under_odds': -110
            })
            
            sample_odds.append({
                'player_name': row['player_name'],
                'prop_type': 'receiving_yards',
                'line': float(median_pred + 10),  # Harder line
                'over_odds': +120,
                'under_odds': -140
            })
    
    return sample_odds


def main():
    """
    Main entry point for the betting system.
    Parses command-line arguments and executes workflow.
    """
    # Setup argument parser
    parser = argparse.ArgumentParser(
        description='NFL Player Prop Betting System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                           # Use current season/week
  python main.py --season 2024 --week 5    # Specific season and week
  python main.py --week 10                 # Current season, week 10
        """
    )
    
    parser.add_argument(
        '--season',
        type=int,
        default=CURRENT_SEASON,
        help=f'NFL season year (default: {CURRENT_SEASON})'
    )
    
    parser.add_argument(
        '--week',
        type=int,
        default=CURRENT_WEEK,
        help=f'NFL week number (default: {CURRENT_WEEK})'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate arguments
    if args.week < 1 or args.week > 18:
        parser.error("Week must be between 1 and 18")
    
    if args.season < 2020 or args.season > 2030:
        parser.error("Season must be between 2020 and 2030")
    
    # Execute workflow
    try:
        run_workflow(season=args.season, week=args.week)
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n\n⚠️  Workflow interrupted by user")
        logger.warning("Workflow interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        logger.error(f"Fatal error in main: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()


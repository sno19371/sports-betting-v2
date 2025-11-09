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

from config import CURRENT_SEASON, CURRENT_WEEK, HISTORICAL_YEARS, MODEL_DIR
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


def run_workflow(season: int, week: int) -> None:
    """
    Execute the complete betting workflow for a given season and week.
    
    Workflow steps:
    1. Fetch historical player data and current schedule
    2. Engineer features from raw data
    3. Load pre-trained model
    4. Generate predictions for target week
    5. Fetch current market odds
    6. Identify +EV betting opportunities
    7. Save recommendations to log file
    
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
        # STEP 2: FETCHING DATA
        # =====================================================================
        logger.info("Step 2: Fetching data")
        print("\n" + "=" * 80)
        print("FETCHING DATA")
        print("=" * 80)

        # 1. Fetch historical data for lookback features
        logger.info(f"Fetching historical data for years: {HISTORICAL_YEARS}")
        historical_df = data_api.get_weekly_stats(years=HISTORICAL_YEARS, save=False)
        if historical_df is None or historical_df.empty:
            raise ValueError("Failed to fetch historical weekly player data")
        print(f"✓ Loaded {len(historical_df):,} historical player-week records")

        # 2. Construct live data for the current prediction week
        logger.info(f"Constructing live data for Season {season}, Week {week}")
        live_week_df = data_api.get_live_week_data(season=season, week=week)
        if live_week_df is None or live_week_df.empty:
            raise ValueError(f"Failed to construct live data for Season {season}, Week {week}. Data may not be available yet.")
        print(f"✓ Constructed live data for {len(live_week_df)} players for the upcoming week")

        # 3. Combine historical and live data for feature engineering
        weekly_df = pd.concat([historical_df, live_week_df], ignore_index=True)
        logger.info(f"Combined historical and live data: {len(weekly_df)} total rows")

        # 4. Fetch schedule data for all relevant years
        try:
            all_years = sorted(list(set(HISTORICAL_YEARS + [season])))
            schedule_df = nfl.import_schedules(years=all_years)
            logger.info(f"Loaded schedule data for years {min(all_years)}-{max(all_years)}")
            print(f"✓ Loaded {len(schedule_df):,} total games for schedule data")
        except Exception as e:
            logger.warning(f"Failed to load schedule data: {e}")
            logger.info("Continuing without schedule data (limited features)")
            schedule_df = None
        
        # =====================================================================
        # STEP 3: ENGINEER FEATURES
        # =====================================================================
        logger.info("Step 3: Engineering features")
        print("\n" + "=" * 80)
        print("FEATURE ENGINEERING")
        print("=" * 80)
        
        engineer = FeatureEngineer()
        
        # Create features for all data (rolling windows need history)
        features_df = engineer.create_features(
            weekly_df=weekly_df,
            schedule_df=schedule_df,
            include_rolling=True,
            include_ewma=True,
            include_matchup=True,
            include_weather=True,
            include_game_script=True
        )
        
        print(f"✓ Features engineered: {features_df.shape[1]} total features")
        
        # Filter to target week for prediction
        target_mask = (features_df['season'] == season) & (features_df['week'] == week)
        current_week_features = features_df[target_mask].copy()
        
        if current_week_features.empty:
            raise ValueError(f"No data found for Season {season}, Week {week}")
        
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
                "Please train a model first before running inference."
            )
        
        model_manager = ModelManager.load_manager(str(model_path))
        print(f"✓ Model loaded from: {model_path}")
        print(f"  Props in model: {model_manager.target_props}")
        print(f"  Quantiles: {model_manager.quantiles}")
        
        # Check if model has feature names
        if hasattr(model_manager, 'feature_names') and model_manager.feature_names:
            print(f"  Features: {len(model_manager.feature_names)} (saved with model)")
        else:
            print(f"  ⚠ Model missing feature names (older version)")
        
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
            print(f"✓ Using {len(feature_cols)} features from model definition")
            
        else:
            # Fallback: manually determine features (less robust)
            logger.warning("Model doesn't have saved feature names, using fallback method")
            print("⚠ Using fallback feature selection (consider retraining model)")
            
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
        print("\nSample predictions (first 3 players):")
        sample_cols = ['player_name'] + [col for col in predictions.columns if 'q0.5' in col][:3]
        print(predictions[sample_cols].head(3).to_string(index=False))
        
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
            print("\n⚠ WARNING: No market odds available from API")
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
            print("\n⚠ No bets found meeting criteria")
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
        print("\n\n⚠ Workflow interrupted by user")
        logger.warning("Workflow interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        logger.error(f"Fatal error in main: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()


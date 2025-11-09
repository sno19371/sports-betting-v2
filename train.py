#!/usr/bin/env python3
# train.py
"""
Training script for NFL player prop betting models.
Trains ModelManager on historical data and saves the trained model artifact.

Usage:
    python train.py
    python train.py --output-file my_model.pkl
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import nfl_data_py as nfl

from config import HISTORICAL_YEARS, MODEL_DIR, PROP_TYPES
from data_sources import NFLDataSources
from feature_engineering import FeatureEngineer
from modeling import ModelManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/training.log')
    ]
)
logger = logging.getLogger(__name__)


def run_training(output_file: str) -> None:
    """
    Execute the complete model training workflow.
    
    Workflow steps:
    1. Fetch all historical player and game data
    2. Engineer features from raw data
    3. Prepare feature matrix (X), target matrix (y), and positions
    4. Train ModelManager on all target props with position filtering
    5. Save trained model to disk
    
    Args:
        output_file: Filename for the saved model (will be saved in MODEL_DIR)
    """
    print("=" * 80)
    print("NFL PROP BETTING MODEL - TRAINING")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Training years: {HISTORICAL_YEARS}")
    print("=" * 80)
    
    try:
        # =====================================================================
        # STEP 1: FETCH HISTORICAL DATA
        # =====================================================================
        logger.info("Step 1: Fetching historical data")
        print("\n" + "=" * 80)
        print("STEP 1: FETCHING HISTORICAL DATA")
        print("=" * 80)
        
        data_api = NFLDataSources()
        
        # Fetch weekly player statistics
        print("\nFetching weekly player statistics...")
        weekly_df = data_api.get_weekly_stats(years=HISTORICAL_YEARS, save=True)
        
        if weekly_df is None or weekly_df.empty:
            raise ValueError("Failed to fetch weekly player data")
        
        logger.info(f"Loaded {len(weekly_df)} player-week records")
        print(f"✓ Loaded {len(weekly_df):,} player-week records")
        print(f"  Seasons: {weekly_df['season'].min()} - {weekly_df['season'].max()}")
        print(f"  Unique players: {weekly_df['player_id'].nunique():,}")
        
        # Fetch schedule data for game-level features
        print("\nFetching schedule data...")
        try:
            schedule_df = nfl.import_schedules(years=HISTORICAL_YEARS)
            logger.info(f"Loaded schedule data: {len(schedule_df)} games")
            print(f"✓ Loaded schedule data: {len(schedule_df):,} games")
        except Exception as e:
            logger.warning(f"Failed to load schedule data: {e}")
            logger.info("Training without schedule data (limited features)")
            print(f"⚠ Schedule data unavailable: {e}")
            print("  Continuing with player stats only (limited features)")
            schedule_df = None
        
        # =====================================================================
        # STEP 2: ENGINEER FEATURES
        # =====================================================================
        logger.info("Step 2: Engineering features")
        print("\n" + "=" * 80)
        print("STEP 2: FEATURE ENGINEERING")
        print("=" * 80)
        
        engineer = FeatureEngineer()
        
        print("\nGenerating features (this may take several minutes)...")
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
        
        # =====================================================================
        # STEP 3: PREPARE TRAINING DATA
        # =====================================================================
        logger.info("Step 3: Preparing training matrices")
        print("\n" + "=" * 80)
        print("STEP 3: PREPARING TRAINING DATA")
        print("=" * 80)
        
        # Define target props to train models for
        target_props = [
            'receiving_yards',
            'rushing_yards',
            'receptions',
            'targets',
            'carries',
            'receiving_tds',
            'rushing_tds',
            'passing_yards',
            'passing_tds',
            'completions'
        ]
        
        # Filter to only props that exist in the data
        available_props = [prop for prop in target_props if prop in features_df.columns]
        
        if not available_props:
            raise ValueError(f"None of the target props found in data. Available columns: {features_df.columns.tolist()}")
        
        logger.info(f"Training on {len(available_props)} props: {available_props}")
        print(f"\nTarget props ({len(available_props)}):")
        for prop in available_props:
            non_null = features_df[prop].notna().sum()
            print(f"  {prop}: {non_null:,} non-null values ({non_null/len(features_df):.1%})")
        
        # Define metadata columns to exclude from features
        metadata_cols = [
            'player_id',
            'player_display_name', 
            'player_name',
            'season',
            'week',
            'team',
            'opponent',
            'position',
            'game_id',
            'game_date'
        ]
        
        # Create target matrix (y)
        y_train = features_df[available_props].copy()
        
        # Extract positions column (needed for position-based filtering)
        if 'position' not in features_df.columns:
            logger.warning("No 'position' column found - models will train on all positions")
            positions = pd.Series(['UNKNOWN'] * len(features_df), index=features_df.index, name='position')
        else:
            positions = features_df['position'].copy()
            logger.info(f"Position distribution: {positions.value_counts().to_dict()}")
        
        # Create feature matrix (X) by programmatically selecting only numeric types
        X_train = features_df.select_dtypes(include=np.number)

        # Drop target prop columns that might also be numeric
        cols_to_drop = [prop for prop in available_props if prop in X_train.columns]
        if cols_to_drop:
            X_train = X_train.drop(columns=cols_to_drop)

        # Also drop 'season' and 'week' as identifiers, not predictive features
        if 'season' in X_train.columns:
            X_train = X_train.drop(columns='season')
        if 'week' in X_train.columns:
            X_train = X_train.drop(columns='week')

        feature_cols = X_train.columns.tolist()
        
        logger.info(f"Training data prepared: X={X_train.shape}, y={y_train.shape}")
        print(f"\n✓ Training data prepared:")
        print(f"  Features (X): {X_train.shape}")
        print(f"  Targets (y): {y_train.shape}")
        print(f"  Positions: {len(positions)} ({positions.nunique()} unique)")
        print(f"\nPosition distribution:")
        for pos, count in positions.value_counts().head(10).items():
            print(f"  {pos}: {count:,} ({count/len(positions):.1%})")
        
        print(f"\nFeature columns ({len(feature_cols)}):")
        
        # Group and display feature types
        feature_types = {}
        for col in feature_cols:
            if 'rolling' in col:
                feature_types.setdefault('Rolling averages', []).append(col)
            elif 'ewma' in col:
                feature_types.setdefault('EWMA', []).append(col)
            elif 'matchup' in col:
                feature_types.setdefault('Matchup (DvP)', []).append(col)
            elif col in ['passing_penalty', 'game_script_factor']:
                feature_types.setdefault('Game context', []).append(col)
            else:
                feature_types.setdefault('Other', []).append(col)
        
        for feature_type, cols in feature_types.items():
            print(f"  {feature_type}: {len(cols)}")
        
        # =====================================================================
        # STEP 4: TRAIN MODEL
        # =====================================================================
        logger.info("Step 4: Training models")
        print("\n" + "=" * 80)
        print("STEP 4: TRAINING MODELS")
        print("=" * 80)
        
        # Define quantiles to predict
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        
        print(f"\nInitializing ModelManager...")
        print(f"  Target props: {len(available_props)}")
        print(f"  Quantiles: {quantiles}")
        
        manager = ModelManager(
            target_props=available_props,
            quantiles=quantiles
        )
        
        print(f"\nTraining {len(available_props)} models (this will take several minutes)...")
        print("=" * 80)
        
        # Train all models with position filtering
        manager.train_all(X_train, y_train, positions)
        
        print("\n" + "=" * 80)
        print(f"✓ Training complete: {len(manager.managers)}/{len(available_props)} models trained")
        
        # Display feature importances for validation
        print("\n" + "=" * 80)
        print("TOP FEATURES BY MODEL")
        print("=" * 80)
        
        importances = manager.get_all_feature_importances(top_n=5)
        for prop, importance_df in importances.items():
            print(f"\n{prop} (Top 5):")
            for _, row in importance_df.iterrows():
                print(f"  {row['feature']}: {row['importance']:.0f}")
        
        # =====================================================================
        # STEP 5: SAVE MODEL
        # =====================================================================
        logger.info("Step 5: Saving model")
        print("\n" + "=" * 80)
        print("STEP 5: SAVING MODEL")
        print("=" * 80)
        
        # Construct output path
        model_dir = Path(MODEL_DIR)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = model_dir / output_file
        
        print(f"\nSaving model to: {output_path}")
        manager.save_manager(str(output_path))
        
        # Verify the file was created
        if output_path.exists():
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"✓ Model saved successfully")
            print(f"  File size: {file_size_mb:.2f} MB")
            print(f"  Location: {output_path.absolute()}")
        else:
            raise IOError(f"Model file was not created at {output_path}")
        
        # =====================================================================
        # TRAINING COMPLETE
        # =====================================================================
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)
        print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        print("\n📊 Training Summary:")
        print(f"  Training samples: {len(X_train):,}")
        print(f"  Features: {X_train.shape[1]}")
        print(f"  Target props: {len(available_props)}")
        print(f"  Quantiles per prop: {len(quantiles)}")
        print(f"  Total models trained: {len(manager.managers) * len(quantiles)}")
        print(f"  Model file: {output_path.name}")
        print(f"  Position filtering: ENABLED")
        
        print("\n✅ Model is ready for inference!")
        print(f"   Run: python main.py --season {max(HISTORICAL_YEARS)} --week 1")
        
        logger.info("Training workflow completed successfully")
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        print(f"\n❌ ERROR: {e}")
        raise


def main():
    """
    Main entry point for the training script.
    Parses command-line arguments and executes training workflow.
    """
    # Setup argument parser
    parser = argparse.ArgumentParser(
        description='Train NFL Player Prop Betting Models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script trains quantile regression models for NFL player props and saves
the trained ModelManager to disk. The saved model can then be loaded by main.py
for making predictions on future games.

Examples:
  python train.py                              # Use default filename
  python train.py --output-file my_model.pkl   # Custom filename
  
The trained model will be saved to the models/ directory.
        """
    )
    
    parser.add_argument(
        '--output-file',
        type=str,
        default='model_manager.pkl',
        help='Output filename for the trained model (default: model_manager.pkl)'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate output filename
    if not args.output_file.endswith('.pkl'):
        args.output_file += '.pkl'
    
    # Execute training
    try:
        run_training(output_file=args.output_file)
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user")
        logger.warning("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        logger.error(f"Fatal error in training: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
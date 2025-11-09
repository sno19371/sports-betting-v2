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

from config import HISTORICAL_YEARS, MODEL_DIR, PROP_TYPES, PROCESSED_DATA_DIR, LOG_DIR
from data_sources import NFLDataSources
from feature_engineering import FeatureEngineer
from modeling import ModelManager

# Configure logging (ensure log directory exists)
_log_dir = Path(LOG_DIR)
_log_dir.mkdir(parents=True, exist_ok=True)
_log_file = _log_dir / "training.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(_log_file)
    ]
)
logger = logging.getLogger(__name__)


DATA_FILE = f"{PROCESSED_DATA_DIR}/model_ready_features.parquet"

def load_data() -> pd.DataFrame | None:
    """
    Load prebuilt model-ready features if available.
    Returns None if the file is missing so we can fall back.
    """
    try:
        print("\n" + "=" * 80)
        print("STEP 0: LOADING PREBUILT DATASET (if available)")
        print("=" * 80)
        print(f"Attempting to load: {DATA_FILE}")
        df = pd.read_parquet(DATA_FILE)
        print(f"✓ Loaded prebuilt dataset. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"⚠ Prebuilt dataset not found at {DATA_FILE}. Falling back to building on the fly.")
        return None

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
        # STEP 1: LOAD OR BUILD FEATURES
        # =====================================================================
        prebuilt_df = load_data()
        if prebuilt_df is None:
            logger.info("Step 1a: Building features from raw sources (no prebuilt file found)")
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
        else:
            features_df = prebuilt_df
            logger.info("Using prebuilt model-ready dataset from disk")
            print("\n" + "=" * 80)
            print("STEP 1: USING PREBUILT DATASET")
            print("=" * 80)
            print(f"✓ Features loaded: {features_df.shape}")
        
        print(f"✓ Features ready")
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
        
        # =====================================================================
        # STEP 3A: FEATURE SELECTION WITH LEAKAGE PREVENTION
        # =====================================================================
        logger.info("Step 3a: Selecting features with leakage prevention")
        print("\n" + "=" * 80)
        print("STEP 3A: FEATURE SELECTION (LEAKAGE PREVENTION)")
        print("=" * 80)
        
        # Create target matrix (y)
        y_train = features_df[available_props].copy()
        
        # Extract positions column (needed for position-based filtering)
        if 'position' not in features_df.columns:
            logger.warning("No 'position' column found - models will train on all positions")
            positions = pd.Series(['UNKNOWN'] * len(features_df), index=features_df.index, name='position')
        else:
            positions = features_df['position'].copy()
            logger.info(f"Position distribution: {positions.value_counts().to_dict()}")
        
        # Allowed vs forbidden features
        ALLOWED_FEATURE_PREFIXES = [
            '_rolling_',           # lagged rolling
            '_ewma_',              # lagged EWMA
            'matchup_',            # DvP ranks
            'passing_penalty',     # weather-based passing difficulty
            'game_script_factor',  # spread-based script
            'is_home',
            'is_dome',
            'spread_line',
            'total_line',
            # Weather (pre-game)
            'wind',
            'temp',
        ]
        FORBIDDEN_FEATURES = [
            # Advanced outcome metrics
            'racr','wopr','pacr','dakota',
            # Fantasy/aggregate
            'fantasy_points','fantasy_points_ppr',
            # EPA
            'receiving_epa','rushing_epa','passing_epa','epa',
            # Air yards / YAC
            'receiving_air_yards','receiving_yards_after_catch',
            'passing_air_yards','passing_yards_after_catch',
            # Shares
            'target_share','air_yards_share','wopr_x','wopr_y',
            # First downs
            'receiving_first_downs','rushing_first_downs','passing_first_downs','first_downs',
            # Passing outcomes
            'attempts','completions','interceptions','sacks','sack_yards','sack_fumbles','sack_fumbles_lost',
            # Raw volume stats (targets)
            'targets','carries','receptions','receiving_yards','rushing_yards','passing_yards',
            'receiving_tds','rushing_tds','passing_tds','passing_att',
            # Fumbles/2pt conversions (outcomes)
            'fumbles','fumbles_lost',
            'rushing_fumbles','rushing_fumbles_lost',
            'receiving_fumbles','receiving_fumbles_lost',
            'two_point_conversions',
            'passing_2pt_conversions','rushing_2pt_conversions','receiving_2pt_conversions',
            # Other outcomes
            'special_teams_tds',
        ]
        
        print("\nFeature Selection Strategy:")
        print(f"  Allowed prefixes: {len(ALLOWED_FEATURE_PREFIXES)}")
        print(f"  Forbidden features: {len(FORBIDDEN_FEATURES)}")
        
        # Start with numeric
        X_train = features_df.select_dtypes(include=np.number).copy()
        initial_feature_count = X_train.shape[1]
        print(f"\nInitial numeric columns: {initial_feature_count}")
        
        # Drop target props
        cols_to_drop = [prop for prop in available_props if prop in X_train.columns]
        if cols_to_drop:
            X_train = X_train.drop(columns=cols_to_drop)
            print(f"Dropped {len(cols_to_drop)} target props")
        
        # Drop identifiers
        identifiers = ['season','week','player_id','game_id']
        id_cols_to_drop = [c for c in identifiers if c in X_train.columns]
        if id_cols_to_drop:
            X_train = X_train.drop(columns=id_cols_to_drop)
            print(f"Dropped {len(id_cols_to_drop)} identifier columns")
        
        # Remove forbidden features
        print("\nRemoving outcome-based features...")
        forbidden_in_data = [col for col in FORBIDDEN_FEATURES if col in X_train.columns]
        if forbidden_in_data:
            print(f"  Found {len(forbidden_in_data)} forbidden features:")
            for col in forbidden_in_data[:15]:
                print(f"    - {col}")
            if len(forbidden_in_data) > 15:
                print(f"    ... and {len(forbidden_in_data) - 15} more")
            X_train = X_train.drop(columns=forbidden_in_data)
            logger.info(f"Dropped {len(forbidden_in_data)} outcome-based features to prevent leakage")
        
        # Validate remaining features
        print("\nValidating remaining features...")
        feature_cols = X_train.columns.tolist()
        suspicious_features = []
        allowed_features = []
        for col in feature_cols:
            if any(prefix in col for prefix in ALLOWED_FEATURE_PREFIXES):
                allowed_features.append(col)
            else:
                suspicious_features.append(col)
        print(f"  Allowed features: {len(allowed_features)}")
        if allowed_features:
            rolling_feats = [f for f in allowed_features if '_rolling_' in f]
            ewma_feats = [f for f in allowed_features if '_ewma_' in f]
            matchup_feats = [f for f in allowed_features if 'matchup_' in f]
            context_feats = [f for f in allowed_features if any(ctx in f for ctx in ['passing_penalty','game_script','is_home','is_dome','spread','total'])]
            print(f"    - Rolling features: {len(rolling_feats)}")
            print(f"    - EWMA features: {len(ewma_feats)}")
            print(f"    - Matchup features: {len(matchup_feats)}")
            print(f"    - Context features: {len(context_feats)}")
        if suspicious_features:
            print(f"\n⚠️  Found {len(suspicious_features)} features not in whitelist:")
            for col in suspicious_features[:10]:
                print(f"    - {col}")
            if len(suspicious_features) > 10:
                print(f"    ... and {len(suspicious_features) - 10} more")
            logger.warning(f"Found {len(suspicious_features)} features not in whitelist. If outcomes, add to FORBIDDEN_FEATURES.")
            print("\n" + "=" * 80)
            print("⚠️  SUSPICIOUS FEATURES DETECTED")
            print("=" * 80)
            print("The features listed above are not in the allowed whitelist.")
            print("They may cause data leakage if they are game outcomes.")
            print("Review them and add to FORBIDDEN_FEATURES if needed.")
            print("=" * 80)
        
        print(f"\n✓ Final feature count: {X_train.shape[1]}")
        print(f"  Removed {initial_feature_count - X_train.shape[1]} features total")
        print(f"  Kept {X_train.shape[1]} pre-game predictive features")
        
        logger.info(f"Feature selection complete: {X_train.shape[1]} features retained")
        logger.info(f"Training data prepared: X={X_train.shape}, y={y_train.shape}")
        
        # =====================================================================
        # STEP 3B: TIME-BASED TRAIN/VAL SPLIT
        # =====================================================================
        logger.info("Step 3b: Creating time-based train/validation split")
        print("\n" + "=" * 80)
        print("STEP 3B: TIME-BASED VALIDATION SPLIT")
        print("=" * 80)
        
        # Define split: use 2020-2023 for training, 2024 for validation
        TRAIN_YEARS = [2020, 2021, 2022, 2023]
        VAL_YEAR = 2024
        
        # Create masks for train/val split
        train_mask = features_df['season'].isin(TRAIN_YEARS)
        val_mask = features_df['season'] == VAL_YEAR
        
        # Split features
        X_train_split = X_train[train_mask].copy()
        y_train_split = y_train[train_mask].copy()
        positions_train = positions[train_mask].copy()
        
        X_val = X_train[val_mask].copy()
        y_val = y_train[val_mask].copy()
        positions_val = positions[val_mask].copy()
        
        print(f"\nTrain set: {len(X_train_split):,} samples (years {TRAIN_YEARS})")
        print(f"Val set:   {len(X_val):,} samples (year {VAL_YEAR})")
        if len(X_val) > 0:
            print(f"Train/Val ratio: {len(X_train_split)/len(X_val):.1f}:1")
        else:
            print("Train/Val ratio: N/A (no validation rows found for selected VAL_YEAR)")
        
        # Show position distribution in both sets
        print("\nTrain set position distribution:")
        for pos, count in positions_train.value_counts().head(5).items():
            print(f"  {pos}: {count:,}")
        
        print("\nVal set position distribution:")
        for pos, count in positions_val.value_counts().head(5).items():
            print(f"  {pos}: {count:,}")
        
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
        
        # Train all models with position filtering (use training split)
        manager.train_all(X_train_split, y_train_split, positions_train)
        
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
        # STEP 4B: VALIDATION EVALUATION
        # =====================================================================
        logger.info("Step 4b: Evaluating on validation set")
        print("\n" + "=" * 80)
        print("VALIDATION PERFORMANCE")
        print("=" * 80)
        
        # Generate predictions on validation set
        print("\nGenerating predictions on validation set...")
        val_predictions = manager.predict_all(X_val) if len(X_val) > 0 else pd.DataFrame(index=X_val.index)
        
        # Calculate metrics for each prop
        print("\nValidation Metrics by Prop:")
        print("-" * 80)
        
        validation_results = []
        
        for prop in available_props:
            if prop not in y_val.columns or len(X_val) == 0:
                continue
            
            # Get actual values (drop NaN)
            y_true = y_val[prop].dropna()
            if len(y_true) == 0:
                continue
            
            # Get median predictions for this prop
            pred_col = f'{prop}_q0.5'
            if pred_col not in val_predictions.columns:
                continue
            
            y_pred = val_predictions.loc[y_true.index, pred_col]
            
            # Calculate metrics
            mae = np.mean(np.abs(y_true - y_pred))
            rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
            
            # Calculate R²
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - y_true.mean()) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Mean Absolute Percentage Error (MAPE)
            # Only for non-zero actuals to avoid division by zero
            non_zero_mask = y_true != 0
            if non_zero_mask.sum() > 0:
                mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
            else:
                mape = np.nan
            
            print(f"\n{prop}:")
            print(f"  Samples:      {len(y_true):,}")
            print(f"  MAE:          {mae:.2f}")
            print(f"  RMSE:         {rmse:.2f}")
            print(f"  R²:           {r2:.3f}")
            if not np.isnan(mape):
                print(f"  MAPE:         {mape:.1f}%")
            print(f"  Mean actual:  {y_true.mean():.1f}")
            print(f"  Mean pred:    {y_pred.mean():.1f}")
            print(f"  Std actual:   {y_true.std():.1f}")
            print(f"  Std pred:     {y_pred.std():.1f}")
            
            # Store results
            validation_results.append({
                'prop': prop,
                'samples': len(y_true),
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'mape': mape if not np.isnan(mape) else None,
                'mean_actual': y_true.mean(),
                'mean_pred': y_pred.mean()
            })
        
        # Summary statistics
        print("\n" + "=" * 80)
        print("OVERALL VALIDATION SUMMARY")
        print("=" * 80)
        
        if validation_results:
            avg_r2 = np.mean([r['r2'] for r in validation_results])
            avg_mae = np.mean([r['mae'] for r in validation_results])
            
            print(f"Average R² across all props:  {avg_r2:.3f}")
            print(f"Average MAE across all props: {avg_mae:.2f}")
            
            # Best and worst props
            best_prop = max(validation_results, key=lambda x: x['r2'])
            worst_prop = min(validation_results, key=lambda x: x['r2'])
            
            print(f"\nBest prop:  {best_prop['prop']} (R² = {best_prop['r2']:.3f})")
            print(f"Worst prop: {worst_prop['prop']} (R² = {worst_prop['r2']:.3f})")
        
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
        print(f"  Training samples: {len(X_train_split):,}")
        print(f"  Validation samples: {len(X_val):,}")
        print(f"  Features: {X_train_split.shape[1]}")
        print(f"  Target props: {len(available_props)}")
        print(f"  Quantiles per prop: {len(quantiles)}")
        print(f"  Total models trained: {len(manager.managers) * len(quantiles)}")
        print(f"  Model file: {output_path.name}")
        print(f"  Position filtering: ENABLED")
        if 'validation_results' in locals() and validation_results:
            avg_r2 = np.mean([r['r2'] for r in validation_results])
            avg_mae = np.mean([r['mae'] for r in validation_results])
            print(f"  Validation R²: {avg_r2:.3f}")
            print(f"  Validation MAE: {avg_mae:.2f}")
        
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
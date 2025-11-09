#!/usr/bin/env python3
"""
Build and save model-ready dataset with all features.
This ensures consistent, leak-free feature engineering.

Usage:
    python build_model_dataset.py
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import nfl_data_py as nfl

from config import HISTORICAL_YEARS, PROCESSED_DATA_DIR
from data_sources import NFLDataSources
from feature_engineering import FeatureEngineer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Build model-ready dataset with comprehensive validation."""
    print("=" * 80)
    print("BUILDING MODEL-READY DATASET")
    print("=" * 80)
    print(f"Historical years: {HISTORICAL_YEARS}")
    print("=" * 80)

    try:
        # =====================================================================
        # STEP 1: LOAD RAW DATA
        # =====================================================================
        logger.info("Step 1: Loading raw data")
        print("\n" + "=" * 80)
        print("STEP 1: LOADING RAW DATA")
        print("=" * 80)

        data_api = NFLDataSources()

        print("\nFetching weekly player statistics...")
        weekly_df = data_api.get_weekly_stats(years=HISTORICAL_YEARS, save=True)
        if weekly_df is None or weekly_df.empty:
            raise ValueError("Failed to fetch weekly player data")
        logger.info(f"Loaded {len(weekly_df)} player-week records")
        print(f"✓ Loaded {len(weekly_df):,} player-week records")
        print(f"  Seasons: {weekly_df['season'].min()} - {weekly_df['season'].max()}")
        print(f"  Unique players: {weekly_df['player_id'].nunique():,}")

        print("\nFetching schedule data...")
        try:
            schedule_df = nfl.import_schedules(years=HISTORICAL_YEARS)
            logger.info(f"Loaded schedule data: {len(schedule_df)} games")
            print(f"✓ Loaded schedule data: {len(schedule_df):,} games")
        except Exception as e:
            logger.warning(f"Failed to load schedule data: {e}")
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

        print(f"✓ Features ready")
        print(f"  Total rows: {len(features_df):,}")
        print(f"  Total columns: {features_df.shape[1]}")

        # =====================================================================
        # STEP 3: LEAKAGE VALIDATION
        # =====================================================================
        logger.info("Step 3: Validating dataset for leakage")
        print("\n" + "=" * 80)
        print("STEP 3: LEAKAGE VALIDATION")
        print("=" * 80)

        leakage_detected = False

        # Check 1: Week 1 DvP baseline
        print("\n🔍 Check 1: Week 1 DvP Baseline Validation")
        print("-" * 80)
        week1_mask = features_df['week'] == 1
        week1_data = features_df[week1_mask]
        matchup_cols = [c for c in features_df.columns if c.startswith('matchup_') and c.endswith('_rank')]

        if not matchup_cols:
            print("⚠️  No DvP matchup columns found - skipping DvP validation")
        else:
            print(f"Found {len(matchup_cols)} DvP matchup columns")
            for col in matchup_cols:
                vals = week1_data[col].dropna()
                if len(vals) == 0:
                    print(f"⚠️  {col}: No non-null Week 1 values")
                    continue
                non_median = (vals.sub(0.5).abs() > 0.01).sum()
                percent_bad = (non_median / len(vals)) * 100
                if non_median > len(vals) * 0.1:
                    print(f"❌ LEAKAGE: {col} — {non_median}/{len(vals)} != 0.5 ({percent_bad:.1f}%)")
                    print(f"   Sample values: {vals.head(10).tolist()}")
                    leakage_detected = True
                else:
                    print(f"✓ PASS: {col} - {len(vals)} Week 1 values validated")

        # Check 2: Rolling lag (first game should be NaN)
        print("\n🔍 Check 2: Rolling Feature Lag Validation")
        print("-" * 80)
        sample_players = features_df.get('player_display_name', pd.Series()).dropna().unique()[:3]
        rolling_cols = [c for c in features_df.columns if '_rolling_' in c]
        for player in sample_players:
            player_data = features_df[features_df['player_display_name'] == player].sort_values(['season', 'week']).head(5)
            if len(player_data) == 0 or not rolling_cols:
                continue
            first_rolling_col = rolling_cols[0]
            first_value = player_data.iloc[0][first_rolling_col]
            if pd.notna(first_value):
                print(f"❌ LEAKAGE: {player} has {first_rolling_col} = {first_value} in first game (should be NaN)")
                leakage_detected = True
            else:
                print(f"✓ PASS: {player} - First game rolling feature is NaN (correct)")

        # Check 3: Feature variance sanity
        print("\n🔍 Check 3: Feature Variance Validation")
        print("-" * 80)
        numeric_cols = features_df.select_dtypes(include=['float64', 'int64']).columns
        feature_cols = [c for c in numeric_cols if c not in ['season', 'week', 'player_id', 'game_id']]
        zero_variance_features = []
        for col in feature_cols[:20]:
            variance = features_df[col].var()
            if variance == 0 or pd.isna(variance):
                zero_variance_features.append(col)
        if zero_variance_features:
            print(f"⚠️  Zero-variance features (sample): {zero_variance_features}")
        else:
            print(f"✓ PASS: All checked features have variance")

        print("\n" + "=" * 80)
        if leakage_detected:
            print("❌ VALIDATION FAILED: DATA LEAKAGE DETECTED")
            print("=" * 80)
            print("\nPlease review feature engineering (shift/expanding) and rerun.")
            return 1
        else:
            print("✅ VALIDATION PASSED: NO LEAKAGE DETECTED")
            print("=" * 80)

        # =====================================================================
        # STEP 4: SAVE DATASET (FLAT PARQUET)
        # =====================================================================
        logger.info("Step 4: Saving dataset")
        print("\n" + "=" * 80)
        print("STEP 4: SAVING DATASET")
        print("=" * 80)
        output_path = Path(PROCESSED_DATA_DIR) / "model_ready_features.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"\nSaving to: {output_path}")
        features_df.to_parquet(output_path, index=False)
        if output_path.exists():
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"✓ Dataset saved successfully")
            print(f"  File size: {file_size_mb:.2f} MB")
            print(f"  Location: {output_path.absolute()}")
        else:
            raise IOError(f"Dataset file was not created at {output_path}")

        print("\n" + "=" * 80)
        print("BUILD COMPLETE")
        print("=" * 80)
        print(f"\n📊 Dataset Summary:")
        print(f"  Total samples: {len(features_df):,}")
        print(f"  Features: {features_df.shape[1]}")
        print(f"  Seasons: {features_df['season'].min()} - {features_df['season'].max()}")
        print(f"  File: {output_path.name}")
        print("\n✅ Dataset is ready for training!")
        print("   Run: python train.py")
        logger.info("Dataset build completed successfully")
        return 0

    except Exception as e:
        logger.error(f"Dataset build failed: {e}", exc_info=True)
        print(f"\n❌ ERROR: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
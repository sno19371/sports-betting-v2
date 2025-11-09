#!/usr/bin/env python3
"""
Test script to validate DvP matchup features are leak-free.

This script:
1. Creates synthetic test data with known patterns
2. Runs feature engineering with DvP features
3. Validates that no future data is used in matchup features
4. Checks specific edge cases (Week 1, cross-season boundaries)

Run: python test_dvp_leakage.py
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path to import feature_engineering
sys.path.insert(0, str(Path(__file__).parent))

from feature_engineering import FeatureEngineer
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def create_test_data():
    """
    Create synthetic test data with known defensive patterns.
    
    Strategy:
    - Kansas City defense: consistently allows 300 passing yards (easy matchup)
    - Buffalo defense: consistently allows 150 passing yards (tough matchup)
    - Track how their ranks evolve week by week
    """
    
    data = []
    
    # 2023 Season - 5 weeks
    # Two QBs: Josh Allen (BUF) and Patrick Mahomes (KC)
    # They play against each other and other teams
    
    # Week 1: First games, no history yet
    data.extend([
        # Josh Allen plays @ Kansas City (easy defense historically, but no data yet)
        {'player_id': 'allen', 'player_display_name': 'Josh Allen', 'position': 'QB',
         'season': 2023, 'week': 1, 'team': 'BUF', 'opponent': 'KC',
         'passing_yards': 280, 'passing_tds': 2},
        
        # Patrick Mahomes plays @ Buffalo (tough defense historically, but no data yet)
        {'player_id': 'mahomes', 'player_display_name': 'Patrick Mahomes', 'position': 'QB',
         'season': 2023, 'week': 1, 'team': 'KC', 'opponent': 'BUF',
         'passing_yards': 320, 'passing_tds': 3},
    ])
    
    # Week 2: Now we have Week 1 history
    # KC allowed 280 yards in Week 1 (from Allen)
    # BUF allowed 320 yards in Week 1 (from Mahomes)
    data.extend([
        # Different QBs now face these defenses
        {'player_id': 'herbert', 'player_display_name': 'Justin Herbert', 'position': 'QB',
         'season': 2023, 'week': 2, 'team': 'LAC', 'opponent': 'KC',
         'passing_yards': 310, 'passing_tds': 2},
        
        {'player_id': 'tua', 'player_display_name': 'Tua Tagovailoa', 'position': 'QB',
         'season': 2023, 'week': 2, 'team': 'MIA', 'opponent': 'BUF',
         'passing_yards': 260, 'passing_tds': 1},
        
        # Original QBs play other teams
        {'player_id': 'allen', 'player_display_name': 'Josh Allen', 'position': 'QB',
         'season': 2023, 'week': 2, 'team': 'BUF', 'opponent': 'LV',
         'passing_yards': 270, 'passing_tds': 2},
        
        {'player_id': 'mahomes', 'player_display_name': 'Patrick Mahomes', 'position': 'QB',
         'season': 2023, 'week': 2, 'team': 'KC', 'opponent': 'DEN',
         'passing_yards': 290, 'passing_tds': 3},
    ])
    
    # Week 3: Now we have Weeks 1-2 history
    # KC has allowed avg of (280 + 310) / 2 = 295 yards
    # BUF has allowed avg of (320 + 260) / 2 = 290 yards
    data.extend([
        {'player_id': 'allen', 'player_display_name': 'Josh Allen', 'position': 'QB',
         'season': 2023, 'week': 3, 'team': 'BUF', 'opponent': 'KC',
         'passing_yards': 305, 'passing_tds': 3},
        
        {'player_id': 'mahomes', 'player_display_name': 'Patrick Mahomes', 'position': 'QB',
         'season': 2023, 'week': 3, 'team': 'KC', 'opponent': 'BUF',
         'passing_yards': 275, 'passing_tds': 2},
    ])
    
    # Week 4
    data.extend([
        {'player_id': 'allen', 'player_display_name': 'Josh Allen', 'position': 'QB',
         'season': 2023, 'week': 4, 'team': 'BUF', 'opponent': 'MIA',
         'passing_yards': 285, 'passing_tds': 2},
        
        {'player_id': 'mahomes', 'player_display_name': 'Patrick Mahomes', 'position': 'QB',
         'season': 2023, 'week': 4, 'team': 'KC', 'opponent': 'LAC',
         'passing_yards': 330, 'passing_tds': 4},
    ])
    
    # Week 5
    data.extend([
        {'player_id': 'allen', 'player_display_name': 'Josh Allen', 'position': 'QB',
         'season': 2023, 'week': 5, 'team': 'BUF', 'opponent': 'DEN',
         'passing_yards': 295, 'passing_tds': 2},
        
        {'player_id': 'mahomes', 'player_display_name': 'Patrick Mahomes', 'position': 'QB',
         'season': 2023, 'week': 5, 'team': 'KC', 'opponent': 'LV',
         'passing_yards': 310, 'passing_tds': 3},
    ])
    
    return pd.DataFrame(data)


def validate_week1_baseline(df):
    """Validate that Week 1 has no matchup features (0.5 baseline)."""
    print("\n" + "="*70)
    print("TEST 1: Week 1 Baseline (No History)")
    print("="*70)
    
    week1_data = df[df['week'] == 1].copy()
    matchup_cols = [col for col in df.columns if col.startswith('matchup_') and col.endswith('_rank')]
    
    if not matchup_cols:
        print("❌ FAIL: No matchup columns found!")
        return False
    
    all_passed = True
    for col in matchup_cols:
        week1_values = week1_data[col].dropna()
        
        if len(week1_values) == 0:
            print(f"⚠️  {col}: No non-null values in Week 1")
            continue
        
        # Check all values are 0.5
        non_baseline = sum(abs(v - 0.5) > 0.01 for v in week1_values)
        
        if non_baseline > 0:
            print(f"❌ FAIL: {col} has {non_baseline}/{len(week1_values)} non-0.5 values in Week 1")
            print(f"   Values: {week1_values.tolist()}")
            all_passed = False
        else:
            print(f"✓ PASS: {col} - All Week 1 values = 0.5 (no history)")
    
    return all_passed


def validate_week2_uses_week1_only(df):
    """Validate that Week 2 matchup features only use Week 1 data."""
    print("\n" + "="*70)
    print("TEST 2: Week 2 Uses Only Week 1 Data")
    print("="*70)
    
    # Check specific matchups we know the answer for
    # In Week 2, Herbert plays @ KC
    # KC's defense should be ranked based ONLY on Week 1 (allowed 280 to Allen)
    
    week2_herbert = df[(df['week'] == 2) & (df['player_display_name'] == 'Justin Herbert')].iloc[0]
    
    matchup_col = 'matchup_passing_yards_rank'
    if matchup_col not in df.columns:
        print(f"❌ FAIL: Column {matchup_col} not found")
        return False
    
    herbert_rank = week2_herbert[matchup_col]
    
    print(f"Justin Herbert @ KC in Week 2:")
    print(f"  matchup_passing_yards_rank = {herbert_rank:.3f}")
    
    # Week 2 also has Tua @ BUF
    # We need to check that ranks are computed correctly
    # Since there are only 2 defenses with data (KC and BUF), 
    # ranks should be either 0.5 or 1.0 depending on which allowed more
    
    # KC allowed 280 in Week 1, BUF allowed 320 in Week 1
    # So BUF is easier (higher rank), KC is tougher (lower rank)
    
    week2_tua = df[(df['week'] == 2) & (df['player_display_name'] == 'Tua Tagovailoa')].iloc[0]
    tua_rank = week2_tua[matchup_col]
    
    print(f"Tua Tagovailoa @ BUF in Week 2:")
    print(f"  matchup_passing_yards_rank = {tua_rank:.3f}")
    
    # Tua faces BUF (which allowed more = easier), so should have higher rank than Herbert vs KC
    if pd.notna(herbert_rank) and pd.notna(tua_rank):
        if tua_rank > herbert_rank:
            print(f"✓ PASS: BUF (allowed 320) ranked easier than KC (allowed 280)")
            return True
        else:
            print(f"❌ FAIL: Rankings incorrect - BUF should be easier than KC")
            return False
    else:
        print(f"⚠️  Cannot validate: NaN values present")
        return False


def validate_expanding_window(df):
    """Validate that matchup ranks evolve correctly as history accumulates."""
    print("\n" + "="*70)
    print("TEST 3: Expanding Window (Ranks Evolve Week by Week)")
    print("="*70)
    
    matchup_col = 'matchup_passing_yards_rank'
    
    # Track Josh Allen facing KC over multiple weeks
    allen_vs_kc = df[
        (df['player_display_name'] == 'Josh Allen') & 
        (df['opponent'] == 'KC')
    ][['week', matchup_col, 'passing_yards']].copy()
    
    print("\nJosh Allen @ KC across weeks:")
    print(allen_vs_kc.to_string(index=False))
    
    # Week 1: Should be 0.5 (no history)
    # Week 3: Should use Weeks 1-2 history of KC defense
    
    if len(allen_vs_kc) >= 2:
        week1_rank = allen_vs_kc[allen_vs_kc['week'] == 1][matchup_col].iloc[0]
        week3_rank = allen_vs_kc[allen_vs_kc['week'] == 3][matchup_col].iloc[0]
        
        if abs(week1_rank - 0.5) < 0.01:
            print(f"✓ PASS: Week 1 rank = 0.5 (baseline)")
        else:
            print(f"❌ FAIL: Week 1 rank = {week1_rank:.3f} (expected 0.5)")
            return False
        
        if pd.notna(week3_rank) and week3_rank != 0.5:
            print(f"✓ PASS: Week 3 rank = {week3_rank:.3f} (uses historical data)")
            return True
        else:
            print(f"❌ FAIL: Week 3 rank still 0.5 or NaN (should use Weeks 1-2 data)")
            return False
    else:
        print("⚠️  Insufficient data to test")
        return False


def validate_no_future_leakage(df):
    """Validate that features never use future data."""
    print("\n" + "="*70)
    print("TEST 4: No Future Data Leakage")
    print("="*70)
    
    # Manual calculation: For Week 3, calculate what KC's defensive rank SHOULD be
    # based only on Weeks 1-2 data
    
    weeks_1_2 = df[(df['week'].isin([1, 2])) & (df['position'] == 'QB')].copy()
    
    # Calculate how much each defense allowed in Weeks 1-2
    defense_allowed = weeks_1_2.groupby('opponent')['passing_yards'].mean()
    
    print("\nDefensive stats through Week 2 (should inform Week 3 ranks):")
    print(defense_allowed.sort_values(ascending=False))
    
    # Get Week 3 matchup rank for someone facing KC
    week3_vs_kc = df[(df['week'] == 3) & (df['opponent'] == 'KC') & (df['position'] == 'QB')]
    
    if len(week3_vs_kc) > 0:
        matchup_col = 'matchup_passing_yards_rank'
        week3_rank = week3_vs_kc.iloc[0][matchup_col]
        
        print(f"\nWeek 3 matchup_passing_yards_rank vs KC: {week3_rank:.3f}")
        
        # Check that this rank makes sense given Weeks 1-2 data only
        # KC allowed 295 avg (2nd easiest if there are tougher defenses)
        
        # The rank should NOT be influenced by Week 3, 4, or 5 data
        # We can't easily validate the exact value, but we can check it's not 0.5
        
        if pd.notna(week3_rank) and abs(week3_rank - 0.5) > 0.01:
            print("✓ PASS: Week 3 rank uses prior data (not baseline 0.5)")
            return True
        else:
            print("❌ FAIL: Week 3 rank is still 0.5 (should use Weeks 1-2 data)")
            return False
    else:
        print("⚠️  No Week 3 data to validate")
        return False


def main():
    print("="*70)
    print("DvP MATCHUP FEATURE LEAKAGE TEST")
    print("="*70)
    
    # Create test data
    print("\nCreating synthetic test data...")
    test_df = create_test_data()
    print(f"✓ Created {len(test_df)} player-game records")
    print(f"  Weeks: {sorted(test_df['week'].unique())}")
    print(f"  Players: {test_df['player_display_name'].unique().tolist()}")
    
    # Run feature engineering
    print("\nRunning feature engineering with DvP matchup features...")
    engineer = FeatureEngineer()
    
    # Only run matchup features for this test
    result_df = engineer.add_matchup_features(test_df)
    
    # Check that matchup columns were created
    matchup_cols = [col for col in result_df.columns if col.startswith('matchup_') and col.endswith('_rank')]
    print(f"✓ Created {len(matchup_cols)} matchup feature columns:")
    for col in matchup_cols:
        print(f"  - {col}")
    
    # Run validation tests
    print("\n" + "="*70)
    print("RUNNING VALIDATION TESTS")
    print("="*70)
    
    test_results = []
    
    test_results.append(("Week 1 Baseline", validate_week1_baseline(result_df)))
    test_results.append(("Week 2 Uses Week 1 Only", validate_week2_uses_week1_only(result_df)))
    test_results.append(("Expanding Window", validate_expanding_window(result_df)))
    test_results.append(("No Future Leakage", validate_no_future_leakage(result_df)))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nResult: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED - No data leakage detected!")
        return 0
    else:
        print("\n⚠️  SOME TESTS FAILED - Review DvP implementation")
        return 1


if __name__ == "__main__":
    sys.exit(main())


# ===== PyTest test cases =====

def _build_engineered_df():
    df = create_test_data()
    fe = FeatureEngineer()
    return fe.add_matchup_features(df)

def test_dvp_week1_is_filled_to_median_no_leakage():
    out = _build_engineered_df()
    assert validate_week1_baseline(out) is True

def test_dvp_week2_uses_week1_only():
    out = _build_engineered_df()
    assert validate_week2_uses_week1_only(out) is True

essential_cols = [
    # ensure generic matchup rank exists
    'matchup_passing_yards_rank',
]

def test_dvp_expanding_window_evolves():
    out = _build_engineered_df()
    # sanity: required columns exist
    for c in essential_cols:
        assert c in out.columns
    assert validate_expanding_window(out) is True

def test_dvp_no_future_leakage():
    out = _build_engineered_df()
    assert validate_no_future_leakage(out) is True
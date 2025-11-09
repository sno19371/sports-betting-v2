import pandas as pd
import pytest
import numpy as np
from feature_engineering import FeatureEngineer

@pytest.fixture
def phase2_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Creates sample weekly_df and schedule_df for testing Phase 2 features.
    Includes multiple seasons, positions, and specific data points to test all conditions.
    """
    # Player-level weekly data
    weekly_data = {
        'player_id': ['A', 'B', 'C', 'D', 'A', 'B', 'C', 'D'],
        'position': ['QB', 'WR', 'WR', 'RB', 'QB', 'WR', 'WR', 'RB'],
        'team': ['KC', 'KC', 'LAC', 'LAC', 'KC', 'KC', 'LAC', 'LAC'],
        'season': [2023, 2023, 2023, 2023, 2024, 2024, 2024, 2024],
        'week': [1, 1, 1, 1, 1, 1, 1, 1],
        'opponent': ['DET', 'DET', 'MIA', 'MIA', 'BAL', 'BAL', 'LV', 'LV'],
        'passing_yards': [250, 0, 0, 0, 300, 0, 0, 0],
        'receiving_yards': [0, 80, 120, 10, 0, 90, 110, 15],
    }
    weekly_df = pd.DataFrame(weekly_data)
    
    # Game-level schedule data
    schedule_data = {
        'season': [2023, 2023, 2024, 2024],
        'week': [1, 1, 1, 1],
        'team': ['KC', 'LAC', 'KC', 'LAC'],
        # Conditions designed to test each case in the functions
        'spread_line': [-7, 3, 10, -10], # Favorite, Neutral, Underdog, Favorite
        'wind_speed': [5, 10, 20, 8], # Normal, Normal, High Wind, Normal
        'temp': [75, 70, 80, 20], # Normal, Normal, Normal, Low Temp
        'is_dome': [False, True, False, False], # Outdoor, Dome, Outdoor, Outdoor
    }
    schedule_df = pd.DataFrame(schedule_data)
    
    return weekly_df, schedule_df

def test_create_features_pipeline_merge(phase2_data):
    """
    Tests that the main create_features pipeline correctly merges schedule_df.
    """
    weekly_df, schedule_df = phase2_data
    engineer = FeatureEngineer()
    
    # Run the full pipeline (disabling other features for a focused test)
    result_df = engineer.create_features(
        weekly_df, 
        schedule_df, 
        include_rolling=False, 
        include_ewma=False,
        include_matchup=False,
        include_weather=False,
        include_game_script=False
    )
    
    # 1. Check that rows are preserved
    assert len(result_df) == len(weekly_df)
    
    # 2. Check that columns from schedule_df are now present
    assert 'spread_line' in result_df.columns
    assert 'wind_speed' in result_df.columns
    
    # 3. Check a specific merged value
    # Player A in 2023 (KC) should have a spread_line of -7
    player_a_2023 = result_df[(result_df['player_id'] == 'A') & (result_df['season'] == 2023)]
    assert player_a_2023['spread_line'].iloc[0] == -7

def test_add_matchup_features(phase2_data):
    """
    Tests the DvP calculation for matchup ranks.
    Ensures ranks are calculated per-season and per-position correctly.
    """
    weekly_df, _ = phase2_data
    engineer = FeatureEngineer()
    result_df = engineer.add_matchup_features(weekly_df)

    # --- Manually Calculate Expected Ranks for 'receiving_yards' for WRs ---
    # Season 2023:
    # - Opponent DET (vs Player B) allowed 80 yards.
    # - Opponent MIA (vs Player C) allowed 120 yards.
    # - MIA allowed more, so it's the easier matchup (higher rank).
    # - With 2 data points, ranks will be 0.5 (harder) and 1.0 (easier).
    # Expected: Player B (vs DET) gets rank 0.5. Player C (vs MIA) gets rank 1.0.

    # Season 2024:
    # - Opponent BAL (vs Player B) allowed 90 yards.
    # - Opponent LV (vs Player C) allowed 110 yards.
    # - LV is easier matchup.
    # Expected: Player B (vs BAL) gets rank 0.5. Player C (vs LV) gets rank 1.0.

    # Assert Player B's matchups
    player_b_2023_rank = result_df.loc[(result_df['player_id'] == 'B') & (result_df['season'] == 2023), 'matchup_receiving_yards_rank'].iloc[0]
    player_b_2024_rank = result_df.loc[(result_df['player_id'] == 'B') & (result_df['season'] == 2024), 'matchup_receiving_yards_rank'].iloc[0]
    assert player_b_2023_rank == pytest.approx(0.5)
    assert player_b_2024_rank == pytest.approx(0.5)

    # Assert Player C's matchups
    player_c_2023_rank = result_df.loc[(result_df['player_id'] == 'C') & (result_df['season'] == 2023), 'matchup_receiving_yards_rank'].iloc[0]
    player_c_2024_rank = result_df.loc[(result_df['player_id'] == 'C') & (result_df['season'] == 2024), 'matchup_receiving_yards_rank'].iloc[0]
    assert player_c_2023_rank == pytest.approx(1.0)
    assert player_c_2024_rank == pytest.approx(1.0)

def test_add_weather_features(phase2_data):
    """
    Tests that the 'passing_penalty' is applied correctly based on weather conditions.
    """
    weekly_df, schedule_df = phase2_data
    merged_df = pd.merge(weekly_df, schedule_df, on=['season', 'week', 'team'])
    
    engineer = FeatureEngineer()
    result_df = engineer.add_weather_features(merged_df)

    # Case 1: Dome game (LAC in 2023) -> No penalty
    dome_game_penalty = result_df.loc[(result_df['team'] == 'LAC') & (result_df['season'] == 2023), 'passing_penalty'].iloc[0]
    assert dome_game_penalty == 1.0

    # Case 2: High wind game (KC in 2024) -> Penalty applied
    high_wind_penalty = result_df.loc[(result_df['team'] == 'KC') & (result_df['season'] == 2024), 'passing_penalty'].iloc[0]
    assert high_wind_penalty == 0.85

    # Case 3: Low temp game (LAC in 2024) -> Penalty applied
    low_temp_penalty = result_df.loc[(result_df['team'] == 'LAC') & (result_df['season'] == 2024), 'passing_penalty'].iloc[0]
    assert low_temp_penalty == 0.85

    # Case 4: Normal outdoor game (KC in 2023) -> No penalty
    normal_game_penalty = result_df.loc[(result_df['team'] == 'KC') & (result_df['season'] == 2023), 'passing_penalty'].iloc[0]
    assert normal_game_penalty == 1.0

def test_add_game_script_features(phase2_data):
    """
    Tests that the 'game_script_factor' is correctly assigned based on spread_line.
    """
    weekly_df, schedule_df = phase2_data
    merged_df = pd.merge(weekly_df, schedule_df, on=['season', 'week', 'team'])
    
    engineer = FeatureEngineer()
    result_df = engineer.add_game_script_features(merged_df)

    # Case 1: Heavy Favorite (LAC in 2024, spread = -10) -> Running script (-1)
    fav_script = result_df.loc[(result_df['team'] == 'LAC') & (result_df['season'] == 2024), 'game_script_factor'].iloc[0]
    assert fav_script == -1

    # Case 2: Heavy Underdog (KC in 2024, spread = 10) -> Passing script (1)
    dog_script = result_df.loc[(result_df['team'] == 'KC') & (result_df['season'] == 2024), 'game_script_factor'].iloc[0]
    assert dog_script == 1

    # Case 3: Neutral game (LAC in 2023, spread = 3) -> Neutral script (0)
    neutral_script = result_df.loc[(result_df['team'] == 'LAC') & (result_df['season'] == 2023), 'game_script_factor'].iloc[0]
    assert neutral_script == 0
    
    # Case 4: Favorite but not heavy (KC in 2023, spread = -7) -> Running script (-1)
    fav_script_edge = result_df.loc[(result_df['team'] == 'KC') & (result_df['season'] == 2023), 'game_script_factor'].iloc[0]
    assert fav_script_edge == -1
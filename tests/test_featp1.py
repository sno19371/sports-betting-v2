# tests/test_feature_engineering.py

import pandas as pd
import pytest
import numpy as np
from feature_engineering import FeatureEngineer
import re

@pytest.fixture
def sample_player_data() -> pd.DataFrame:
    """
    Creates a sample DataFrame with two players over several weeks.
    This predictable data allows us to manually calculate expected outcomes for tests.
    """
    data = {
        'player_id': [
            'A', 'A', 'A', 'A',  # Player A
            'B', 'B', 'B', 'B',  # Player B
        ],
        'player_display_name': [
            'Player A', 'Player A', 'Player A', 'Player A',
            'Player B', 'Player B', 'Player B', 'Player B',
        ],
        'season': [2024, 2024, 2024, 2024, 2024, 2024, 2024, 2024],
        'week': [1, 2, 3, 4, 1, 2, 3, 4],
        'targets': [10, 20, 30, 40, 5, 5, 10, 10],
        'receiving_yards': [100, 120, 140, 160, 50, 50, 80, 80],
    }
    df = pd.DataFrame(data)
    # Ensure data is sorted, as it would be in the real application
    return df.sort_values(['player_id', 'season', 'week'])

def test_add_rolling_averages(sample_player_data):
    """
    Tests the add_rolling_averages method to ensure:
    1. The calculation is correct.
    2. Data does not leak from the current week (due to shift(1)).
    3. The rolling window resets for each player.
    """
    engineer = FeatureEngineer()
    result_df = engineer.add_rolling_averages(sample_player_data, window=2, stats=['targets'])
    
    # --- Expected values for a 2-week rolling window, shifted ---
    # Player A:
    # Week 1: NaN (no prior data)
    # Week 2: 10.0 (from Week 1)
    # Week 3: 15.0 (avg of Week 1 & 2: (10+20)/2)
    # Week 4: 25.0 (avg of Week 2 & 3: (20+30)/2)
    #
    # Player B:
    # Week 1: NaN (no prior data)
    # Week 2: 5.0 (from Week 1)
    # Week 3: 5.0 (avg of Week 1 & 2: (5+5)/2)
    # Week 4: 7.5 (avg of Week 2 & 3: (5+10)/2)
    expected_values = [np.nan, 10.0, 15.0, 25.0, np.nan, 5.0, 5.0, 7.5]
    
    # Use pd.testing.assert_series_equal for robust comparison, especially with NaNs
    pd.testing.assert_series_equal(
        result_df['targets_rolling_2g'].reset_index(drop=True),
        pd.Series(expected_values, name='targets_rolling_2g'),
        check_dtype=False # Allow for float comparison
    )

def test_add_ewma_features(sample_player_data):
    """
    Tests the add_ewma_features method to ensure:
    1. The EWMA calculation is correct.
    2. Data does not leak from the current week.
    3. The calculation resets for each player.
    """
    engineer = FeatureEngineer()
    result_df = engineer.add_ewma_features(sample_player_data, span=3, stats=['receiving_yards'])

    # --- Manually calculated expected values for EWMA with span=3 (alpha=0.5), shifted ---
    # Player A:
    # Week 1: NaN
    # Week 2: 100.0 (from Week 1)
    # Week 3: 110.0 (0.5 * 120 + 0.5 * 100)
    # Week 4: 125.0 (0.5 * 140 + 0.5 * 110)
    #
    # Player B:
    # Week 1: NaN
    # Week 2: 50.0 (from Week 1)
    # Week 3: 50.0 (0.5 * 50 + 0.5 * 50)
    # Week 4: 65.0 (0.5 * 80 + 0.5 * 50)
    expected_values = [np.nan, 100.0, 110.0, 125.0, np.nan, 50.0, 50.0, 65.0]

    pd.testing.assert_series_equal(
        result_df['receiving_yards_ewma_3'].reset_index(drop=True),
        pd.Series(expected_values, name='receiving_yards_ewma_3'),
        check_dtype=False
    )

def test_validate_dataframe_unsorted_data():
    """
    Tests that the validation method correctly identifies and warns about unsorted data.
    """
    # Create an unsorted DataFrame
    unsorted_data = {
        'player_id': ['A', 'A', 'A'],
        'season': [2024, 2024, 2024],
        'week': [1, 3, 2], # Intentionally out of order
        'targets': [10, 30, 20]
    }
    df = pd.DataFrame(unsorted_data)
    
    engineer = FeatureEngineer()
    # The validate_dataframe method should sort the data in-place and return True
    # We are testing that it doesn't raise an error and corrects the order.
    is_valid = engineer.validate_dataframe(df)
    
    assert is_valid is True
    # Check that the DataFrame is now sorted
    assert df['week'].tolist() == [1, 2, 3]

def test_validate_dataframe_missing_columns():
    """
    Tests that the validation method raises a ValueError for missing required columns.
    """
    df = pd.DataFrame({'player_id': ['A'], 'targets': [10]})
    engineer = FeatureEngineer()
    
    expected_error_msg = "Missing required columns: ['season', 'week']"
    with pytest.raises(ValueError, match=re.escape(expected_error_msg)):
        engineer.validate_dataframe(df)
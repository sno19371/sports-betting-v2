# tests/test_betting_p2.py

import pandas as pd
import pytest
from pathlib import Path
from betting import BetFinder, MODEL_CONFIG

# ---- Test Fixtures ----

@pytest.fixture
def bet_finder_instance_p2() -> BetFinder:
    """
    Provides a BetFinder instance with data designed specifically
    to test Phase 2 functionality (filtering, sizing, logging).
    """
    # Set a specific edge threshold for predictable filtering
    MODEL_CONFIG['prop_edge_threshold'] = 0.05  # 5%
    
    predictions_df = pd.DataFrame({
        'player_name': ['Player A', 'Player B', 'Player C'],
        # Player A: Strong edge, should be recommended
        'test_prop_q0.25': [80.0, 0, 0],
        'test_prop_q0.5': [100.0, 0, 0],
        'test_prop_q0.75': [120.0, 0, 0],
        # Player B: Small edge (below threshold), should be filtered out
        'another_prop_q0.25': [0, 48.0, 0],
        'another_prop_q0.5': [0, 50.0, 0],
        'another_prop_q0.75': [0, 52.0, 0],
    })
    
    market_odds = [
        { # This bet should pass the filter
            'player_name': 'Player A', 'prop_type': 'test_prop',
            'line': 85.5, 'over_odds': -110, 'under_odds': -110
        },
        { # This bet should be filtered out (edge will be < 5%)
            'player_name': 'Player B', 'prop_type': 'another_prop',
            'line': 49.5, 'over_odds': -110, 'under_odds': -110
        }
    ]
    
    return BetFinder(model_predictions=predictions_df, market_odds=market_odds)

# ---- Test Functions for Phase 2 ----

def test_find_bets_filtering(bet_finder_instance_p2):
    """
    Tests that find_bets correctly filters opportunities based on the edge threshold.
    """
    # The fixture is designed so that Player A's bet has a large edge,
    # while Player B's bet has a small but positive edge that is BELOW the 5% threshold.
    
    recommended_bets = bet_finder_instance_p2.find_bets()
    
    # Assert that only one bet was recommended
    assert len(recommended_bets) == 1
    
    # Assert that the recommended bet is for Player A
    assert recommended_bets[0]['player_name'] == 'Player A'
    assert recommended_bets[0]['edge'] >= MODEL_CONFIG['prop_edge_threshold']

def test_find_bets_sizing_and_logging(bet_finder_instance_p2):
    """
    Tests that a correctly identified bet is logged with the right structure
    and includes a calculated Kelly fraction.
    """
    recommended_bets = bet_finder_instance_p2.find_bets()
    
    assert len(recommended_bets) == 1
    bet = recommended_bets[0]
    
    # Check that all expected keys from _log_bet are present
    expected_keys = [
        'timestamp', 'player_name', 'prop_type', 'line', 'side',
        'american_odds', 'decimal_odds', 'model_probability',
        'market_probability', 'edge', 'expected_value', 'kelly_fraction',
        'recommended_stake_pct'
    ]
    assert all(key in bet for key in expected_keys)
    
    # Check that the Kelly fraction was calculated and is a positive number
    assert 'kelly_fraction' in bet
    assert isinstance(bet['kelly_fraction'], float)
    assert bet['kelly_fraction'] > 0

def test_save_log(bet_finder_instance_p2, tmp_path: Path, monkeypatch):
    """
    Tests that the save_log method correctly creates a CSV file with the
    recommended bets. Uses monkeypatch to redirect output to a temp directory.
    """
    # Use monkeypatch to temporarily change the LOG_DIR to our temp test directory
    monkeypatch.setattr('betting.LOG_DIR', str(tmp_path))
    
    # 1. Find bets to populate the internal list
    bet_finder_instance_p2.find_bets()
    assert len(bet_finder_instance_p2.recommended_bets) == 1
    
    # 2. Save the log
    bet_finder_instance_p2.save_log()
    
    # 3. Verify the file was created
    # Use glob to find the created file, since the name is timestamped
    log_files = list(tmp_path.glob("bet_log_*.csv"))
    assert len(log_files) == 1
    
    # 4. Read the file and verify its contents
    log_df = pd.read_csv(log_files[0])
    
    assert len(log_df) == 1
    assert log_df.iloc[0]['player_name'] == 'Player A'
    assert 'kelly_fraction' in log_df.columns
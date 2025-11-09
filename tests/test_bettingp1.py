# tests/test_betting.py

import pandas as pd
import pytest
import numpy as np
from scipy.stats import norm
from betting import BetFinder
from utils import calculate_expected_value

# ---- Test Fixtures ----

@pytest.fixture
def bet_finder_instance() -> BetFinder:
    """
    Provides a BetFinder instance with sample predictions and market odds
    for use in multiple tests.
    """
    # Sample predictions mimicking the output of ModelManager
    predictions_df = pd.DataFrame({
        'player_name': ['Player A', 'Player B'],
        # Player A has a wide, uncertain distribution
        'test_prop_q0.1': [50.0, 80.0],
        'test_prop_q0.25': [70.0, 90.0],
        'test_prop_q0.5': [90.0, 100.0],
        'test_prop_q0.75': [110.0, 110.0],
        'test_prop_q0.9': [130.0, 120.0],
    })
    
    # Sample market odds
    market_odds = [
        {
            'player_name': 'Player A',
            'prop_type': 'test_prop',
            'line': 75.5,
            'over_odds': -110,
            'under_odds': -110
        },
        {
            'player_name': 'Player B',
            'prop_type': 'test_prop',
            'line': 105.5,
            'over_odds': -110,
            'under_odds': -110
        }
    ]
    
    return BetFinder(model_predictions=predictions_df, market_odds=market_odds)

# ---- Test Functions ----

def test_estimate_probability_from_quantiles_center():
    """
    Tests the core math of the probability estimator with a line equal to the median.
    """
    finder = BetFinder(pd.DataFrame(), []) # Init with empty data for this isolated test
    
    # Create quantiles with a mean of 100
    quantiles = pd.Series({
        'prop_q0.25': 90.0,
        'prop_q0.5': 100.0,
        'prop_q0.75': 110.0
    })
    
    # When the line is exactly the mean, probs should be 50/50
    prob_over, prob_under = finder._estimate_probability_from_quantiles(quantiles, line=100.0)
    
    assert prob_over == pytest.approx(0.5)
    assert prob_under == pytest.approx(0.5)

def test_estimate_probability_from_quantiles_off_center():
    """Tests the probability estimator with a line away from the median."""
    finder = BetFinder(pd.DataFrame(), [])
    
    quantiles = pd.Series({
        'prop_q0.25': 90.0,
        'prop_q0.5': 100.0,
        'prop_q0.75': 110.0
    })
    
    # Manual calculation for expected probability
    mean = 100.0
    iqr = 110.0 - 90.0  # 20
    std_dev = iqr / 1.349
    line = 110.0
    
    expected_prob_under = norm.cdf(line, loc=mean, scale=std_dev)
    expected_prob_over = 1 - expected_prob_under
    
    prob_over, prob_under = finder._estimate_probability_from_quantiles(quantiles, line)
    
    assert prob_under == pytest.approx(expected_prob_under)
    assert prob_over == pytest.approx(expected_prob_over)

def test_estimate_probability_bad_data():
    """Tests that the estimator handles bad data gracefully."""
    finder = BetFinder(pd.DataFrame(), [])
    
    # Case 1: Non-positive IQR
    bad_iqr_quantiles = pd.Series({'prop_q0.25': 100, 'prop_q0.5': 100, 'prop_q0.75': 100})
    prob_over, prob_under = finder._estimate_probability_from_quantiles(bad_iqr_quantiles, 100)
    assert (prob_over, prob_under) == (0.5, 0.5)
    
    # Case 2: Missing quantiles
    missing_quantiles = pd.Series({'prop_q0.5': 100})
    prob_over, prob_under = finder._estimate_probability_from_quantiles(missing_quantiles, 100)
    assert (prob_over, prob_under) == (0.5, 0.5)

def test_find_bets_identifies_positive_ev(bet_finder_instance, capsys):
    """
    Tests the main find_bets loop. Since Phase 1 prints output, we capture
    the stdout and check that a known +EV bet is found and printed.
    """
    # For Player A, our model has a median of 90. The market line is 75.5.
    # We expect a strong "Over" signal. Let's verify the calculation.
    # Manually calculate the EV for the first market in the fixture
    market = bet_finder_instance.market_odds[0]
    preds = bet_finder_instance.predictions.iloc[0]
    
    quantiles = preds.filter(like='_q')
    prob_over, _ = bet_finder_instance._estimate_probability_from_quantiles(quantiles, market['line'])
    
    # The probability of being over 75.5 should be high
    assert prob_over > 0.6 
    
    # Now check the EV calculation
    ev = calculate_expected_value(1.909, prob_over)
    assert ev > 0 # This confirms the logic to find a +EV bet is sound
    
    # Now run the actual function and check the printed output
    bet_finder_instance.find_bets()
    captured = capsys.readouterr()
    
    assert "+EV OPPORTUNITY (OVER):" in captured.out
    assert "Player: Player A" in captured.out
    assert "Prop: test_prop Over 75.5" in captured.out
    
    # For Player B, our model has a median of 100. The line is 105.5.
    # We expect a weak "Under" signal, which might not be +EV after odds.
    assert "Player: Player B" in captured.out
    assert "OPPORTUNITY (UNDER)" in captured.out
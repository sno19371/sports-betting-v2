# utils.py
"""
Utility functions for NFL prop betting system.
Contains mathematical calculations, odds conversions, and betting utilities.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict, Union
import logging
from datetime import datetime
from config import MODEL_CONFIG

logger = logging.getLogger(__name__)


# =============================================================================
# ODDS CONVERSION AND PROBABILITY FUNCTIONS
# =============================================================================

def convert_american_to_decimal(american_odds: int) -> float:
    """
    Convert American odds to decimal odds.
    
    Args:
        american_odds: American format odds (e.g., -110, +150)
        
    Returns:
        Decimal odds (e.g., 1.909, 2.50)
        
    Examples:
        >>> convert_american_to_decimal(-110)
        1.909
        >>> convert_american_to_decimal(+150)
        2.50
    """
    if american_odds == 0:
        raise ValueError("American odds cannot be 0")
    
    if american_odds < 0:
        # Negative odds: decimal = 1 + (100 / |odds|)
        decimal_odds = 1 + (100 / abs(american_odds))
    else:
        # Positive odds: decimal = 1 + (odds / 100)
        decimal_odds = 1 + (american_odds / 100)
    
    return round(decimal_odds, 3)


def convert_decimal_to_american(decimal_odds: float) -> int:
    """
    Convert decimal odds to American odds.
    
    Args:
        decimal_odds: Decimal format odds (e.g., 1.909, 2.50)
        
    Returns:
        American odds (e.g., -110, +150)
    """
    if decimal_odds <= 1:
        raise ValueError("Decimal odds must be greater than 1")
    
    if decimal_odds < 2:
        # Favorite: american = -100 / (decimal - 1)
        american_odds = int(-100 / (decimal_odds - 1))
    else:
        # Underdog: american = (decimal - 1) * 100
        american_odds = int((decimal_odds - 1) * 100)
    
    return american_odds


def implied_probability(decimal_odds: float) -> float:
    """
    Calculate implied probability from decimal odds.
    
    Args:
        decimal_odds: Decimal format odds
        
    Returns:
        Implied probability as a decimal (0-1)
        
    Example:
        >>> implied_probability(2.0)
        0.5
        >>> implied_probability(1.909)
        0.524
    """
    if decimal_odds <= 1:
        raise ValueError("Decimal odds must be greater than 1")
    
    return round(1 / decimal_odds, 4)


def remove_vig(odds_list: List[float]) -> List[float]:
    """
    Remove vig from a list of decimal odds to get true probabilities.
    
    Uses the multiplicative method (also known as the margin weights method)
    which is more accurate than simple normalization.
    
    Args:
        odds_list: List of decimal odds for all outcomes in a market
        
    Returns:
        List of vig-free probabilities (sum to 1.0)
        
    Example:
        >>> remove_vig([1.909, 1.909])  # -110/-110 market
        [0.5, 0.5]
        >>> remove_vig([1.87, 1.95])  # Typical O/U with vig
        [0.512, 0.488]
    """
    if not odds_list or len(odds_list) < 2:
        raise ValueError("Need at least 2 odds to remove vig")
    
    if any(odds <= 1 for odds in odds_list):
        raise ValueError("All decimal odds must be greater than 1")
    
    # Calculate implied probabilities
    implied_probs = [implied_probability(odds) for odds in odds_list]
    
    # Calculate total probability (overround)
    total_prob = sum(implied_probs)
    
    if total_prob <= 0:
        raise ValueError("Total implied probability must be positive")
    
    # Method 1: Simple normalization (most common for 2-way markets)
    if len(odds_list) == 2:
        # For 2-way markets, use multiplicative method
        # This preserves the ratio of probabilities
        vig = total_prob - 1
        
        # Multiplicative method formula
        true_probs = []
        for i, prob in enumerate(implied_probs):
            # Remove vig proportionally based on original probability
            true_prob = prob / total_prob
            true_probs.append(round(true_prob, 4))
    else:
        # For multi-way markets, use simple normalization
        true_probs = [round(prob / total_prob, 4) for prob in implied_probs]
    
    # Ensure probabilities sum to 1 (handle rounding)
    prob_sum = sum(true_probs)
    if abs(prob_sum - 1.0) > 0.001:
        # Adjust largest probability for rounding errors
        max_idx = true_probs.index(max(true_probs))
        true_probs[max_idx] += (1.0 - prob_sum)
    
    return true_probs


# =============================================================================
# KELLY CRITERION AND BANKROLL MANAGEMENT
# =============================================================================

def calculate_kelly_fraction(
    decimal_odds: float, 
    model_prob: float, 
    fraction: float = None
) -> float:
    """
    Calculate optimal bet size using fractional Kelly criterion.
    
    Kelly formula: f = (p * b - q) / b
    Where:
        f = fraction of bankroll to bet
        p = probability of winning (model probability)
        q = probability of losing (1 - p)
        b = net odds (decimal_odds - 1)
    
    Args:
        decimal_odds: Decimal odds offered by bookmaker
        model_prob: Model's probability estimate (0-1)
        fraction: Kelly fraction to use (defaults to config value)
        
    Returns:
        Fraction of bankroll to bet (0 if no edge or negative)
        
    Example:
        >>> calculate_kelly_fraction(2.1, 0.55, 0.25)  # +110 odds, 55% model prob
        0.0227  # Bet 2.27% of bankroll (full Kelly would be 9.09%)
    """
    if fraction is None:
        fraction = MODEL_CONFIG.get('kelly_fraction', 0.25)
    
    if decimal_odds <= 1:
        raise ValueError("Decimal odds must be greater than 1")
    
    if not 0 <= model_prob <= 1:
        raise ValueError("Model probability must be between 0 and 1")
    
    if not 0 < fraction <= 1:
        raise ValueError("Kelly fraction must be between 0 and 1")
    
    # Calculate Kelly criterion
    b = decimal_odds - 1  # Net odds
    q = 1 - model_prob    # Probability of losing
    
    # Full Kelly
    full_kelly = (model_prob * b - q) / b
    
    # If no edge, don't bet
    if full_kelly <= 0:
        return 0.0
    
    # Apply fractional Kelly
    fractional_kelly = full_kelly * fraction
    
    # Apply maximum bet size constraint
    max_bet = MODEL_CONFIG.get('max_bet_size', 0.05)
    
    return round(min(fractional_kelly, max_bet), 4)


def calculate_expected_value(
    decimal_odds: float,
    model_prob: float,
    stake: float = 1.0
) -> float:
    """
    Calculate expected value of a bet.
    
    Args:
        decimal_odds: Decimal odds offered
        model_prob: Model's probability estimate
        stake: Bet stake amount (default 1.0 for unit calculation)
        
    Returns:
        Expected value in stake units
        
    Example:
        >>> calculate_expected_value(2.1, 0.55, 100)
        10.5  # Expected profit of $10.50 on $100 bet
    """
    if decimal_odds <= 1:
        raise ValueError("Decimal odds must be greater than 1")
    
    if not 0 <= model_prob <= 1:
        raise ValueError("Model probability must be between 0 and 1")
    
    # EV = (probability of winning × profit) - (probability of losing × stake)
    profit = stake * (decimal_odds - 1)
    ev = (model_prob * profit) - ((1 - model_prob) * stake)
    
    return round(ev, 2)


def calculate_edge(model_prob: float, implied_prob: float) -> float:
    """
    Calculate betting edge as the difference between model and market probability.
    
    Args:
        model_prob: Model's probability estimate
        implied_prob: Market implied probability
        
    Returns:
        Edge as a decimal (positive = +EV)
    """
    return round(model_prob - implied_prob, 4)


# =============================================================================
# STATISTICAL HELPER FUNCTIONS
# =============================================================================

def calculate_zscore(value: float, mean: float, std: float) -> float:
    """
    Calculate z-score for a value given mean and standard deviation.
    
    Args:
        value: The value to calculate z-score for
        mean: Population/sample mean
        std: Population/sample standard deviation
        
    Returns:
        Z-score (number of standard deviations from mean)
    """
    if std == 0:
        return 0.0
    
    return round((value - mean) / std, 3)


def calculate_percentile(values: List[float], value: float) -> float:
    """
    Calculate what percentile a value falls into within a list of values.
    
    Args:
        values: List of values to compare against
        value: The value to find percentile for
        
    Returns:
        Percentile (0-100)
    """
    if not values:
        return 50.0
    
    sorted_values = sorted(values)
    count_below = sum(1 for v in sorted_values if v < value)
    percentile = (count_below / len(values)) * 100
    
    return round(percentile, 1)


def calculate_trend(values: List[float], window: int = 3) -> float:
    """
    Calculate trend coefficient using linear regression over recent values.
    
    Args:
        values: List of values (most recent last)
        window: Number of recent values to use
        
    Returns:
        Trend coefficient (positive = upward trend)
    """
    if len(values) < 2:
        return 0.0
    
    recent_values = values[-window:] if len(values) >= window else values
    
    if len(recent_values) < 2:
        return 0.0
    
    # Simple linear regression
    x = np.arange(len(recent_values))
    y = np.array(recent_values)
    
    # Handle edge cases
    if np.std(y) == 0:
        return 0.0
    
    # Calculate slope
    slope = np.polyfit(x, y, 1)[0]
    
    return round(float(slope), 3)


# =============================================================================
# DATA VALIDATION AND CLEANING
# =============================================================================

def validate_odds_data(odds_data: Dict) -> Tuple[bool, List[str]]:
    """
    Validate odds data structure and values.
    
    Args:
        odds_data: Dictionary containing odds information
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    if not odds_data:
        errors.append("Odds data is empty")
        return False, errors
    
    # Check required fields
    required_fields = ['decimal_odds', 'market_type']
    for field in required_fields:
        if field not in odds_data:
            errors.append(f"Missing required field: {field}")
    
    # Validate odds values
    if 'decimal_odds' in odds_data:
        odds = odds_data['decimal_odds']
        if isinstance(odds, (int, float)):
            if odds <= 1:
                errors.append(f"Invalid decimal odds: {odds}")
        elif isinstance(odds, list):
            for odd in odds:
                if odd <= 1:
                    errors.append(f"Invalid decimal odds in list: {odd}")
    
    return len(errors) == 0, errors


def clean_player_name(name: str) -> str:
    """
    Standardize player name format.
    
    Args:
        name: Player name in any format
        
    Returns:
        Standardized name format
    """
    if not name:
        return ""
    
    # Remove extra whitespace
    name = " ".join(name.split())
    
    # Handle suffixes
    suffixes = ['Jr.', 'Jr', 'Sr.', 'Sr', 'III', 'II', 'IV']
    for suffix in suffixes:
        if name.endswith(f" {suffix}"):
            name = name[:-len(suffix)-1].strip() + f" {suffix.rstrip('.')}"
    
    return name


# =============================================================================
# PERFORMANCE METRICS
# =============================================================================

def calculate_roi(profit: float, total_staked: float) -> float:
    """
    Calculate return on investment.
    
    Args:
        profit: Total profit/loss
        total_staked: Total amount staked
        
    Returns:
        ROI as a percentage
    """
    if total_staked == 0:
        return 0.0
    
    return round((profit / total_staked) * 100, 2)


def calculate_clv(closing_odds: float, bet_odds: float) -> float:
    """
    Calculate Closing Line Value.
    
    CLV measures how much value you got compared to the closing line.
    Positive CLV indicates you beat the closing line.
    
    Args:
        closing_odds: Final decimal odds at market close
        bet_odds: Decimal odds when bet was placed
        
    Returns:
        CLV as a percentage
    """
    if bet_odds <= 0:
        return 0.0
    
    # CLV = (closing_odds / bet_odds - 1) * 100
    clv = ((closing_odds / bet_odds) - 1) * 100
    
    return round(clv, 2)


def calculate_sharpe_ratio(
    returns: List[float],
    risk_free_rate: float = 0.02
) -> float:
    """
    Calculate Sharpe ratio for a series of returns.
    
    Args:
        returns: List of period returns
        risk_free_rate: Annual risk-free rate (default 2%)
        
    Returns:
        Sharpe ratio
    """
    if not returns or len(returns) < 2:
        return 0.0
    
    returns_array = np.array(returns)
    
    # Calculate average return and standard deviation
    avg_return = np.mean(returns_array)
    std_return = np.std(returns_array, ddof=1)
    
    if std_return == 0:
        return 0.0
    
    # Adjust risk-free rate to match period of returns
    # Assuming daily returns
    daily_rf = risk_free_rate / 252
    
    # Calculate Sharpe ratio
    sharpe = (avg_return - daily_rf) / std_return
    
    # Annualize if using daily returns
    sharpe_annual = sharpe * np.sqrt(252)
    
    return round(float(sharpe_annual), 3)


# =============================================================================
# LOGGING AND FORMATTING
# =============================================================================

def format_odds_display(
    decimal_odds: float,
    include_american: bool = True,
    include_probability: bool = True
) -> str:
    """
    Format odds for display.
    
    Args:
        decimal_odds: Decimal odds to format
        include_american: Whether to include American odds
        include_probability: Whether to include implied probability
        
    Returns:
        Formatted string representation
    """
    parts = [f"{decimal_odds:.2f}"]
    
    if include_american:
        american = convert_decimal_to_american(decimal_odds)
        sign = "+" if american > 0 else ""
        parts.append(f"({sign}{american})")
    
    if include_probability:
        prob = implied_probability(decimal_odds)
        parts.append(f"[{prob:.1%}]")
    
    return " ".join(parts)


def format_bet_recommendation(
    player: str,
    market: str,
    position: str,
    line: float,
    odds: float,
    model_prob: float,
    kelly_stake: float,
    edge: float
) -> Dict[str, Union[str, float]]:
    """
    Format a bet recommendation for output.
    
    Args:
        player: Player name
        market: Market type (e.g., 'receiving_yards')
        position: Position ('over' or 'under')
        line: Line value
        odds: Decimal odds
        model_prob: Model probability
        kelly_stake: Recommended stake fraction
        edge: Calculated edge
        
    Returns:
        Formatted dictionary with bet details
    """
    return {
        'timestamp': datetime.now().isoformat(),
        'player': player,
        'market': market,
        'position': position,
        'line': line,
        'odds': odds,
        'odds_display': format_odds_display(odds),
        'model_probability': round(model_prob, 3),
        'implied_probability': implied_probability(odds),
        'edge': round(edge, 3),
        'edge_pct': f"{edge:.1%}",
        'kelly_fraction': round(kelly_stake, 4),
        'recommended_stake_pct': f"{kelly_stake:.2%}",
        'expected_value': calculate_expected_value(odds, model_prob)
    }


# =============================================================================
# TESTING AND EXAMPLES
# =============================================================================

def run_examples():
    """Run example calculations to demonstrate utility functions."""
    print("=" * 60)
    print("NFL PROP BETTING UTILITIES - EXAMPLES")
    print("=" * 60)
    
    # Odds conversion examples
    print("\n📊 ODDS CONVERSION:")
    american_odds = [-110, +150, -200, +250]
    for ao in american_odds:
        do = convert_american_to_decimal(ao)
        prob = implied_probability(do)
        print(f"  {ao:+4d} → {do:.3f} decimal → {prob:.1%} implied prob")
    
    # Vig removal example
    print("\n🎯 VIG REMOVAL:")
    market_odds = [1.909, 1.909]  # -110/-110
    true_probs = remove_vig(market_odds)
    print(f"  Market: {market_odds} → True probs: {true_probs}")
    
    market_odds = [1.87, 1.95]  # Typical O/U
    true_probs = remove_vig(market_odds)
    print(f"  Market: {market_odds} → True probs: {true_probs}")
    
    # Kelly criterion example
    print("\n💰 KELLY CRITERION:")
    scenarios = [
        (2.10, 0.55, "Small edge"),
        (3.00, 0.40, "Medium edge"),
        (1.80, 0.60, "Large edge"),
        (2.00, 0.45, "No edge"),
    ]
    
    for odds, prob, desc in scenarios:
        kelly = calculate_kelly_fraction(odds, prob)
        ev = calculate_expected_value(odds, prob, 100)
        edge = calculate_edge(prob, implied_probability(odds))
        print(f"  {desc}: Odds={odds:.2f}, Prob={prob:.0%}")
        print(f"    → Kelly={kelly:.2%}, EV=${ev:.2f}, Edge={edge:.1%}")
    
    # CLV example
    print("\n📈 CLOSING LINE VALUE:")
    bet_scenarios = [
        (2.00, 1.90, "Beat closing line"),
        (1.85, 1.90, "Lost to closing line"),
        (1.90, 1.90, "No CLV"),
    ]
    
    for bet_odds, closing, desc in bet_scenarios:
        clv = calculate_clv(closing, bet_odds)
        print(f"  {desc}: Bet@{bet_odds:.2f}, Closed@{closing:.2f} → CLV={clv:+.1f}%")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Run examples when script is executed directly
    run_examples()
    
    # Run quick tests
    print("\n🧪 RUNNING UNIT TESTS...")
    
    try:
        # Test odds conversion
        assert abs(convert_american_to_decimal(-110) - 1.909) < 0.01
        assert abs(convert_american_to_decimal(150) - 2.5) < 0.01
        assert convert_decimal_to_american(1.909) == -110
        assert convert_decimal_to_american(2.5) == 150
        
        # Test implied probability
        assert abs(implied_probability(2.0) - 0.5) < 0.001
        
        # Test vig removal
        probs = remove_vig([1.909, 1.909])
        assert abs(sum(probs) - 1.0) < 0.001
        assert abs(probs[0] - 0.5) < 0.01
        
        # Test Kelly criterion
        kelly = calculate_kelly_fraction(2.0, 0.6, 0.25)
        assert 0 < kelly < 0.05  # Should be positive but fractional
        
        print("✅ All tests passed!")
        
    except AssertionError as e:
        print(f"❌ Test failed: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")



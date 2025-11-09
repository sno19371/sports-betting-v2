import pytest

from utils import (
    convert_american_to_decimal,
    convert_decimal_to_american,
    implied_probability,
    remove_vig,
    calculate_kelly_fraction,
    calculate_expected_value,
    calculate_edge,
    calculate_zscore,
    calculate_percentile,
    calculate_trend,
    validate_odds_data,
    clean_player_name,
    calculate_roi,
    calculate_clv,
    calculate_sharpe_ratio,
    format_odds_display,
    format_bet_recommendation,
)


# -----------------------------------------------------------------------------
# ODDS CONVERSION AND PROBABILITY
# -----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "american, expected_decimal",
    [
        (-110, 1.909),
        (150, 2.50),
        (-200, 1.5),
        (300, 4.0),
    ],
)
def test_convert_american_to_decimal(american, expected_decimal):
    assert convert_american_to_decimal(american) == expected_decimal


def test_convert_american_to_decimal_errors():
    with pytest.raises(ValueError, match="cannot be 0"):
        convert_american_to_decimal(0)


@pytest.mark.parametrize(
    "decimal, expected_american",
    [
        (1.909, -110),
        (2.50, 150),
        (1.50, -200),
        (3.00, 200),
    ],
)
def test_convert_decimal_to_american(decimal, expected_american):
    assert convert_decimal_to_american(decimal) == expected_american


def test_convert_decimal_to_american_errors():
    with pytest.raises(ValueError, match="greater than 1"):
        convert_decimal_to_american(1.0)
    with pytest.raises(ValueError, match="greater than 1"):
        convert_decimal_to_american(0.99)


@pytest.mark.parametrize(
    "decimal, expected_prob",
    [
        (2.0, 0.5),
        (1.909, 0.524),  # 1/1.909 ≈ 0.5239
    ],
)
def test_implied_probability(decimal, expected_prob):
    assert implied_probability(decimal) == pytest.approx(expected_prob, rel=1e-3, abs=1e-3)


def test_implied_probability_errors():
    with pytest.raises(ValueError, match="greater than 1"):
        implied_probability(1.0)
    with pytest.raises(ValueError, match="greater than 1"):
        implied_probability(-2.0)


def test_remove_vig_two_way_fair():
    # -110/-110 market → normalized to 50/50
    probs = remove_vig([1.909, 1.909])
    assert sum(probs) == pytest.approx(1.0, abs=1e-6)
    assert probs[0] == pytest.approx(0.5, abs=1e-6)
    assert probs[1] == pytest.approx(0.5, abs=1e-6)


def test_remove_vig_two_way_typical():
    # Typical O/U type example, assert close normalization and sum to 1
    probs = remove_vig([1.87, 1.95])
    assert sum(probs) == pytest.approx(1.0, abs=1e-6)
    # First outcome should be slightly above 0.5, second slightly below
    assert probs[0] == pytest.approx(0.511, abs=0.005)
    assert probs[1] == pytest.approx(0.489, abs=0.005)


def test_remove_vig_multi_way():
    probs = remove_vig([2.5, 3.0, 10.0])
    assert sum(probs) == pytest.approx(1.0, abs=1e-6)
    assert all(p > 0 for p in probs)
    assert len(probs) == 3


def test_remove_vig_errors():
    with pytest.raises(ValueError, match="at least 2"):
        remove_vig([])
    with pytest.raises(ValueError, match="at least 2"):
        remove_vig([2.0])
    with pytest.raises(ValueError, match="greater than 1"):
        remove_vig([1.0, 2.0])


# -----------------------------------------------------------------------------
# KELLY, EV, EDGE
# -----------------------------------------------------------------------------
def test_calculate_kelly_fraction_large_edge_capped():
    # Large edge should be capped by MODEL_CONFIG max_bet_size (default 0.05)
    k = calculate_kelly_fraction(decimal_odds=3.0, model_prob=0.70)
    assert k == pytest.approx(0.05, abs=1e-6)  # capped at 5%


def test_calculate_kelly_fraction_small_edge():
    # Small but positive edge should produce a small positive fraction
    k = calculate_kelly_fraction(decimal_odds=2.2, model_prob=0.52)  # b=1.2
    # Full kelly ~ ((0.52*1.2 - 0.48) / 1.2) = 0.12/1.2 = 0.10; fractional 0.25 -> 0.025
    assert k == pytest.approx(0.025, abs=5e-3)


@pytest.mark.parametrize(
    "decimal_odds, prob",
    [
        (2.0, 0.50),  # zero edge
        (2.0, 0.45),  # negative edge
    ],
)
def test_calculate_kelly_fraction_no_edge(decimal_odds, prob):
    assert calculate_kelly_fraction(decimal_odds=decimal_odds, model_prob=prob) == 0.0


def test_calculate_kelly_fraction_errors():
    with pytest.raises(ValueError, match="greater than 1"):
        calculate_kelly_fraction(decimal_odds=1.0, model_prob=0.5)
    with pytest.raises(ValueError, match="between 0 and 1"):
        calculate_kelly_fraction(decimal_odds=2.0, model_prob=1.5)
    with pytest.raises(ValueError, match="between 0 and 1"):
        calculate_kelly_fraction(decimal_odds=2.0, model_prob=-0.1)
    with pytest.raises(ValueError, match="between 0 and 1"):
        calculate_kelly_fraction(decimal_odds=2.0, model_prob=0.5, fraction=0.0)
    with pytest.raises(ValueError, match="between 0 and 1"):
        calculate_kelly_fraction(decimal_odds=2.0, model_prob=0.5, fraction=1.1)


@pytest.mark.parametrize(
    "decimal_odds, prob, stake, expected",
    [
        (2.1, 0.55, 100, 15.50),  # 0.55*110 - 0.45*100 = 15.5
        (1.8, 0.60, 50, 4.00),    # (0.60 * 40 profit) - (0.40 * 50 stake) = 4
        (3.0, 0.30, 10, -1.00),   # 0.30*20 - 0.70*10 = -1
    ],
)
def test_calculate_expected_value(decimal_odds, prob, stake, expected):
    assert calculate_expected_value(decimal_odds, prob, stake) == pytest.approx(expected, abs=1e-6)


def test_calculate_expected_value_errors():
    with pytest.raises(ValueError, match="greater than 1"):
        calculate_expected_value(1.0, 0.5)
    with pytest.raises(ValueError, match="between 0 and 1"):
        calculate_expected_value(2.0, 1.2)


@pytest.mark.parametrize(
    "model_prob, implied_prob, expected",
    [
        (0.55, 0.50, 0.05),
        (0.50, 0.50, 0.0),
        (0.45, 0.50, -0.05),
    ],
)
def test_calculate_edge(model_prob, implied_prob, expected):
    assert calculate_edge(model_prob, implied_prob) == pytest.approx(expected, abs=1e-6)


# -----------------------------------------------------------------------------
# STATS HELPERS
# -----------------------------------------------------------------------------
def test_calculate_zscore_basic():
    assert calculate_zscore(12, mean=10, std=2) == pytest.approx(1.0, abs=1e-6)


def test_calculate_zscore_zero_std():
    assert calculate_zscore(10, mean=10, std=0) == 0.0


def test_calculate_percentile_basic():
    values = [1, 2, 3, 4, 5]
    # 3 values (1,2,3) are < 4 → 3/5 = 60.0
    assert calculate_percentile(values, 4) == pytest.approx(60.0, abs=1e-6)


def test_calculate_percentile_empty():
    # Default to 50 if list empty
    assert calculate_percentile([], 10) == 50.0


def test_calculate_trend_upward():
    # Last 3 points [3,4,5] slope ≈ 1.0
    assert calculate_trend([1, 2, 3, 4, 5]) == pytest.approx(1.0, abs=1e-6)


def test_calculate_trend_downward():
    assert calculate_trend([5, 4, 3, 2, 1]) == pytest.approx(-1.0, abs=1e-6)


def test_calculate_trend_flat():
    assert calculate_trend([3, 3, 3, 3]) == 0.0
    assert calculate_trend([3]) == 0.0
    assert calculate_trend([]) == 0.0


# -----------------------------------------------------------------------------
# DATA VALIDATION AND CLEANING
# -----------------------------------------------------------------------------
def test_validate_odds_data_valid_single():
    valid, errors = validate_odds_data({"decimal_odds": 1.91, "market_type": "h2h"})
    assert valid is True
    assert errors == []


def test_validate_odds_data_valid_list():
    valid, errors = validate_odds_data({"decimal_odds": [1.91, 2.05], "market_type": "h2h"})
    assert valid is True
    assert errors == []


def test_validate_odds_data_missing_and_invalid():
    valid, errors = validate_odds_data({})
    assert valid is False
    assert "Odds data is empty" in errors

    valid2, errors2 = validate_odds_data({"decimal_odds": 1.0})
    assert valid2 is False
    assert any("Invalid decimal odds" in e for e in errors2)
    assert any("Missing required field" in e for e in errors2)

    valid3, errors3 = validate_odds_data({"decimal_odds": [1.91, 1.0], "market_type": "totals"})
    assert valid3 is False
    assert any("Invalid decimal odds in list" in e for e in errors3)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("  Tyreek   Hill ", "Tyreek Hill"),
        ("Justin Jefferson Jr.", "Justin Jefferson Jr"),
        ("Tom Brady Sr.", "Tom Brady Sr"),
        ("", ""),
        (None, ""),  # type: ignore[arg-type]
    ],
)
def test_clean_player_name(raw, expected):
    assert clean_player_name(raw) == expected


# -----------------------------------------------------------------------------
# PERFORMANCE METRICS
# -----------------------------------------------------------------------------
def test_calculate_roi_basic():
    assert calculate_roi(100.0, 1000.0) == pytest.approx(10.0, abs=1e-6)  # 10%
    assert calculate_roi(-50.0, 200.0) == pytest.approx(-25.0, abs=1e-6)


def test_calculate_roi_zero_staked():
    assert calculate_roi(100.0, 0.0) == 0.0


@pytest.mark.parametrize(
    "closing, bet, expected",
    [
        (2.00, 1.90, +5.26),   # (2.00/1.90 - 1)*100 ≈ 5.263% → 5.26
        (1.85, 1.90, -2.63),   # ≈ -2.6316 → -2.63
        (1.90, 1.90, 0.00),
    ],
)
def test_calculate_clv(closing, bet, expected):
    assert calculate_clv(closing, bet) == pytest.approx(expected, abs=0.02)


def test_calculate_sharpe_ratio_basic():
    # Simple daily returns; set RF=0 to simplify; expect positive Sharpe
    r = [0.01, -0.005, 0.015, 0.0, 0.02, -0.01]
    s = calculate_sharpe_ratio(r, risk_free_rate=0.0)
    assert s > 0


def test_calculate_sharpe_ratio_edge_cases():
    assert calculate_sharpe_ratio([], risk_free_rate=0.0) == 0.0
    assert calculate_sharpe_ratio([0.01], risk_free_rate=0.0) == 0.0
    assert calculate_sharpe_ratio([0.01, 0.01, 0.01], risk_free_rate=0.0) == 0.0


# -----------------------------------------------------------------------------
# FORMATTING
# -----------------------------------------------------------------------------
def test_format_odds_display_defaults():
    s = format_odds_display(1.909, include_american=True, include_probability=True)
    # Decimal rounded to 2 places
    assert s.startswith("1.91")
    # American odds for 1.909 ≈ -110
    assert "(-110)" in s
    # Implied probability bracket
    assert "[52.4%]" in s  # 1/1.909 ≈ 52.39%


def test_format_bet_recommendation():
    rec = format_bet_recommendation(
        player="Tyreek Hill",
        market="receiving_yards",
        position="over",
        line=75.5,
        odds=1.909,
        model_prob=0.55,
        kelly_stake=0.025,
        edge=0.03,
    )
    # Keys exist
    for k in [
        "timestamp",
        "player",
        "market",
        "position",
        "line",
        "odds",
        "odds_display",
        "model_probability",
        "implied_probability",
        "edge",
        "edge_pct",
        "kelly_fraction",
        "recommended_stake_pct",
        "expected_value",
    ]:
        assert k in rec

    # Basic content checks
    assert rec["player"] == "Tyreek Hill"
    assert rec["market"] == "receiving_yards"
    assert rec["position"] == "over"
    assert rec["line"] == 75.5
    assert rec["odds"] == 1.909
    assert rec["odds_display"].startswith("1.91")
    assert "(-110)" in rec["odds_display"]
    assert rec["model_probability"] == pytest.approx(0.55, abs=1e-6)
    assert rec["edge"] == pytest.approx(0.03, abs=1e-6)



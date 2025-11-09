"""betting.py

Expected value, Kelly sizing, and simple bet logging utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import math


def american_to_decimal(odds: int) -> float:
    if odds > 0:
        return 1 + odds / 100
    else:
        return 1 + 100 / abs(odds)


def expected_value(prob_win: float, decimal_odds: float) -> float:
    lose_prob = max(0.0, 1.0 - prob_win)
    return prob_win * (decimal_odds - 1) - lose_prob


def kelly_fraction(prob_win: float, decimal_odds: float) -> float:
    b = decimal_odds - 1
    q = 1 - prob_win
    numerator = b * prob_win - q
    denominator = b
    if denominator == 0:
        return 0.0
    return max(0.0, numerator / denominator)


@dataclass
class Bet:
    selection: str
    american_odds: int
    prob_win: float
    bankroll: float
    kelly_multiplier: float = 0.25

    def stake(self) -> float:
        dec = american_to_decimal(self.american_odds)
        f = kelly_fraction(self.prob_win, dec)
        return round(self.bankroll * self.kelly_multiplier * f, 2)

    def ev(self) -> float:
        dec = american_to_decimal(self.american_odds)
        return expected_value(self.prob_win, dec)


__all__ = [
    'american_to_decimal',
    'expected_value',
    'kelly_fraction',
    'Bet',
]



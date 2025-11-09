"""main.py

Weekly workflow runner: fetch data, engineer features, train model, output bets.
"""

from __future__ import annotations

import logging
from datetime import datetime

from config import LOGGING_CONFIG
from utils import setup_logging
from data_sources import NFLDataSources
from feature_engineering import build_player_features
from modeling import train_baseline
from betting import Bet


def run_weekly_workflow() -> None:
    setup_logging(LOGGING_CONFIG)
    logger = logging.getLogger(__name__)

    logger.info("Starting weekly workflow")
    ds = NFLDataSources()

    weekly = ds.get_weekly_stats(years=[2024], save=False)
    features = build_player_features(weekly)

    model = train_baseline(features, target_column='receiving_yards')
    logger.info("Trained baseline model")

    # Example bet using a placeholder probability
    bet = Bet(selection="Tyreek Hill o75.5 receiving yards", american_odds=-110, prob_win=0.57, bankroll=1000)
    logger.info(f"Bet stake: ${bet.stake()} | EV: {bet.ev():.3f}")

    logger.info("Workflow complete")


if __name__ == "__main__":
    run_weekly_workflow()



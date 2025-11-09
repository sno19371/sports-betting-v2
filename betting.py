# betting.py
"""
Bet finding and portfolio management for NFL player props.
Identifies +EV betting opportunities by comparing model predictions to market odds.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from pathlib import Path
import logging
from pprint import pprint

from config import MODEL_CONFIG, LOG_DIR
from utils import (
    convert_american_to_decimal,
    calculate_expected_value,
    calculate_edge,
    implied_probability,
    calculate_kelly_fraction
)

logger = logging.getLogger(__name__)


class BetFinder:
    """
    Identifies positive expected value (EV) betting opportunities.
    
    Takes quantile predictions from the modeling system and compares them
    against market odds to find profitable bets.
    
    Example:
        >>> predictions = model_manager.predict_all(X_test)
        >>> market_odds = [{'player_name': 'Player A', 'prop_type': 'receiving_yards', ...}]
        >>> bet_finder = BetFinder(predictions, market_odds)
        >>> bets = bet_finder.find_bets()
    """
    
    def __init__(
        self,
        model_predictions: pd.DataFrame,
        market_odds: List[Dict]
    ):
        """
        Initialize the bet finder.
        
        Args:
            model_predictions: DataFrame with quantile predictions from ModelManager
                              Must have columns like: player_name, {prop}_q{quantile}
            market_odds: List of market dictionaries, each containing:
                        - player_name: str
                        - prop_type: str (e.g., 'receiving_yards')
                        - line: float (e.g., 75.5)
                        - over_odds: int (American odds, e.g., -110)
                        - under_odds: int (American odds, e.g., -110)
        """
        self.predictions = model_predictions
        self.market_odds = market_odds
        self.recommended_bets: List[Dict] = []
        
        logger.info(f"BetFinder initialized with {len(model_predictions)} predictions "
                   f"and {len(market_odds)} market lines")
    
    def _estimate_probability_from_quantiles(
        self,
        quantiles: pd.Series,
        line: float
    ) -> Tuple[float, float]:
        """
        Estimate probability of over/under by fitting a normal distribution to quantiles.
        
        This method translates our quantile predictions into probabilities for a specific
        betting line by:
        1. Fitting a normal distribution using the median and IQR
        2. Using the CDF to calculate probabilities
        
        Args:
            quantiles: Series with quantile predictions (e.g., prop_q0.1, prop_q0.25, etc.)
            line: The betting line to evaluate
            
        Returns:
            Tuple of (prob_over, prob_under)
        """
        # Extract key quantiles (median and IQR bounds)
        # Look for columns ending with the quantile values
        q50 = None
        q25 = None
        q75 = None
        
        for col_name, value in quantiles.items():
            if col_name.endswith('_q0.5'):
                q50 = value
            elif col_name.endswith('_q0.25'):
                q25 = value
            elif col_name.endswith('_q0.75'):
                q75 = value
        
        # Validate that we have the required quantiles
        if q50 is None or q25 is None or q75 is None:
            logger.warning("Missing required quantiles (0.25, 0.5, 0.75). "
                          "Returning neutral probabilities.")
            return (0.5, 0.5)
        
        # Check for NaN values
        if np.isnan(q50) or np.isnan(q25) or np.isnan(q75):
            logger.warning("NaN values in quantiles. Returning neutral probabilities.")
            return (0.5, 0.5)
        
        # Fit normal distribution
        # Mean = median (q0.5)
        mean = q50
        
        # Standard deviation from IQR
        # For a normal distribution: IQR ≈ 1.349 * σ
        # Therefore: σ ≈ IQR / 1.349
        iqr = q75 - q25
        
        if iqr <= 0:
            logger.warning(f"Invalid IQR ({iqr}). Returning neutral probabilities.")
            return (0.5, 0.5)
        
        std_dev = iqr / 1.349
        
        # Handle edge case of very small std dev
        if std_dev < 0.01:
            logger.warning(f"Very small std dev ({std_dev}). Adjusting to 0.01.")
            std_dev = 0.01
        
        # Calculate probabilities using normal CDF
        prob_under = stats.norm.cdf(line, loc=mean, scale=std_dev)
        prob_over = 1 - prob_under
        
        # Clip probabilities to reasonable bounds (avoid extreme values)
        prob_over = np.clip(prob_over, 0.01, 0.99)
        prob_under = np.clip(prob_under, 0.01, 0.99)
        
        logger.debug(f"Distribution fit: mean={mean:.2f}, std={std_dev:.2f}, "
                    f"line={line:.1f} -> P(over)={prob_over:.3f}")
        
        return (prob_over, prob_under)
    
    def _log_bet(
        self,
        player_name: str,
        prop_type: str,
        line: float,
        side: str,
        american_odds: int,
        decimal_odds: float,
        model_prob: float,
        market_prob: float,
        edge: float,
        ev: float,
        kelly_fraction: float
    ) -> None:
        """
        Log a recommended bet to the internal list.
        
        Args:
            player_name: Player name
            prop_type: Type of prop (e.g., 'receiving_yards')
            line: Betting line
            side: 'Over' or 'Under'
            american_odds: American odds format
            decimal_odds: Decimal odds format
            model_prob: Model's probability estimate
            market_prob: Market's implied probability
            edge: Edge (model prob - market prob)
            ev: Expected value per $1 bet
            kelly_fraction: Recommended stake as fraction of bankroll
        """
        bet = {
            'timestamp': datetime.now().isoformat(),
            'player_name': player_name,
            'prop_type': prop_type,
            'line': line,
            'side': side,
            'american_odds': american_odds,
            'decimal_odds': round(decimal_odds, 3),
            'model_probability': round(model_prob, 4),
            'market_probability': round(market_prob, 4),
            'edge': round(edge, 4),
            'expected_value': round(ev, 4),
            'kelly_fraction': round(kelly_fraction, 4),
            'recommended_stake_pct': f"{kelly_fraction:.2%}"
        }
        
        self.recommended_bets.append(bet)
        logger.info(f"Logged bet: {player_name} {prop_type} {side} {line} "
                   f"(Edge: {edge:+.1%}, Kelly: {kelly_fraction:.2%})")
    
    def save_log(self, filename: Optional[str] = None) -> None:
        """
        Save recommended bets to a CSV file.
        
        Args:
            filename: Optional custom filename. If not provided, uses timestamped default.
        """
        if not self.recommended_bets:
            logger.info("No bets to save - bet list is empty")
            return
        
        # Convert to DataFrame
        bets_df = pd.DataFrame(self.recommended_bets)
        
        # Create filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"bet_log_{timestamp}.csv"
        
        # Ensure LOG_DIR exists
        log_dir = Path(LOG_DIR)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Construct full path
        filepath = log_dir / filename
        
        # Save to CSV
        bets_df.to_csv(filepath, index=False)
        
        logger.info(f"Bet log saved: {filepath}")
        logger.info(f"  Total bets: {len(self.recommended_bets)}")
        logger.info(f"  Total edge: {bets_df['edge'].sum():.1%}")
        logger.info(f"  Average edge: {bets_df['edge'].mean():.1%}")
        logger.info(f"  Total Kelly allocation: {bets_df['kelly_fraction'].sum():.2%}")
    
    def find_bets(self) -> List[Dict]:
        """
        Find all +EV betting opportunities in the market.
        
        This method:
        1. Calculates EV for all market lines
        2. Filters based on edge threshold from config
        3. Sizes bets using Kelly criterion
        4. Logs recommended bets
        
        Returns:
            List of recommended bet dictionaries with all details
        """
        logger.info(f"Scanning {len(self.market_odds)} markets for +EV opportunities")
        
        # Reset recommended bets list
        self.recommended_bets = []
        
        # Get edge threshold from config
        edge_threshold = MODEL_CONFIG.get('prop_edge_threshold', 0.03)
        logger.info(f"Using edge threshold: {edge_threshold:.1%}")
        
        for market in self.market_odds:
            # Extract market details
            player_name = market.get('player_name')
            prop_type = market.get('prop_type')
            line = market.get('line')
            over_odds = market.get('over_odds')
            under_odds = market.get('under_odds')
            
            # Validate market data
            if not all([player_name, prop_type, line, over_odds, under_odds]):
                logger.warning(f"Incomplete market data: {market}")
                continue
            
            # Find matching predictions
            player_predictions = self.predictions[
                self.predictions['player_name'] == player_name
            ]
            
            if player_predictions.empty:
                logger.warning(f"No predictions found for player: {player_name}")
                continue
            
            # Extract quantile columns for this prop type
            quantile_cols = [col for col in player_predictions.columns 
                           if col.startswith(f'{prop_type}_q')]
            
            if not quantile_cols:
                logger.warning(f"No quantile predictions for {player_name} - {prop_type}")
                continue
            
            # Get the quantile values (assuming single row per player)
            quantiles = player_predictions[quantile_cols].iloc[0]
            
            # Estimate probabilities from quantiles
            prob_over, prob_under = self._estimate_probability_from_quantiles(
                quantiles, line
            )
            
            # Convert American odds to decimal
            decimal_over = convert_american_to_decimal(over_odds)
            decimal_under = convert_american_to_decimal(under_odds)
            
            # Calculate expected value for both sides
            ev_over = calculate_expected_value(decimal_over, prob_over, stake=1.0)
            ev_under = calculate_expected_value(decimal_under, prob_under, stake=1.0)
            
            # Calculate edge (model prob - market prob)
            market_prob_over = implied_probability(decimal_over)
            market_prob_under = implied_probability(decimal_under)
            
            edge_over = calculate_edge(prob_over, market_prob_over)
            edge_under = calculate_edge(prob_under, market_prob_under)
            
            # Phase 2: Filter and size +EV opportunities
            if ev_over > 0 and edge_over >= edge_threshold:
                # Calculate Kelly fraction for bet sizing
                kelly_fraction = calculate_kelly_fraction(
                    decimal_odds=decimal_over,
                    model_prob=prob_over
                )
                
                # Log the bet
                self._log_bet(
                    player_name=player_name,
                    prop_type=prop_type,
                    line=line,
                    side='Over',
                    american_odds=over_odds,
                    decimal_odds=decimal_over,
                    model_prob=prob_over,
                    market_prob=market_prob_over,
                    edge=edge_over,
                    ev=ev_over,
                    kelly_fraction=kelly_fraction
                )
            
            if ev_under > 0 and edge_under >= edge_threshold:
                # Calculate Kelly fraction for bet sizing
                kelly_fraction = calculate_kelly_fraction(
                    decimal_odds=decimal_under,
                    model_prob=prob_under
                )
                
                # Log the bet
                self._log_bet(
                    player_name=player_name,
                    prop_type=prop_type,
                    line=line,
                    side='Under',
                    american_odds=under_odds,
                    decimal_odds=decimal_under,
                    model_prob=prob_under,
                    market_prob=market_prob_under,
                    edge=edge_under,
                    ev=ev_under,
                    kelly_fraction=kelly_fraction
                )
        
        logger.info(f"Scan complete: {len(self.recommended_bets)} bets recommended")
        
        return self.recommended_bets


if __name__ == "__main__":
    """
    Example usage demonstrating the BetFinder functionality for Phase 2.
    """
    print("=" * 70)
    print("BET FINDER - PHASE 2 EXAMPLE")
    print("=" * 70)
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Create sample predictions DataFrame
    # This simulates output from ModelManager.predict_all()
    predictions_df = pd.DataFrame({
        'player_name': ['Tyreek Hill', 'Christian McCaffrey', 'Travis Kelce'],
        # Receiving yards quantiles
        'receiving_yards_q0.1': [65.0, 15.0, 45.0],
        'receiving_yards_q0.25': [80.0, 25.0, 55.0],
        'receiving_yards_q0.5': [100.0, 40.0, 70.0],
        'receiving_yards_q0.75': [120.0, 55.0, 85.0],
        'receiving_yards_q0.9': [135.0, 70.0, 100.0],
        # Rushing yards quantiles
        'rushing_yards_q0.1': [5.0, 60.0, 2.0],
        'rushing_yards_q0.25': [8.0, 75.0, 4.0],
        'rushing_yards_q0.5': [12.0, 95.0, 6.0],
        'rushing_yards_q0.75': [18.0, 115.0, 10.0],
        'rushing_yards_q0.9': [25.0, 135.0, 15.0]
    })
    
    print("\nModel Predictions:")
    print(predictions_df[['player_name', 'receiving_yards_q0.5', 'rushing_yards_q0.5']])
    
    # Create sample market odds
    # Design these to create both +EV and -EV scenarios
    market_odds = [
        {
            'player_name': 'Tyreek Hill',
            'prop_type': 'receiving_yards',
            'line': 85.5,
            'over_odds': -110,  # Our model says 100 median, market line at 85.5
            'under_odds': -110  # This should be +EV for OVER
        },
        {
            'player_name': 'Tyreek Hill',
            'prop_type': 'receiving_yards',
            'line': 110.5,
            'over_odds': +150,  # Our model says 100 median, line at 110.5
            'under_odds': -180  # This might be +EV for UNDER
        },
        {
            'player_name': 'Christian McCaffrey',
            'prop_type': 'rushing_yards',
            'line': 80.5,
            'over_odds': -105,  # Our model says 95 median, market line at 80.5
            'under_odds': -115  # This should be +EV for OVER
        },
        {
            'player_name': 'Travis Kelce',
            'prop_type': 'receiving_yards',
            'line': 75.5,
            'over_odds': -120,  # Our model says 70 median, line at 75.5
            'under_odds': +100  # This might be -EV for both (close to fair)
        },
        {
            'player_name': 'Christian McCaffrey',
            'prop_type': 'receiving_yards',
            'line': 35.5,
            'over_odds': +110,  # Our model says 40 median, line at 35.5
            'under_odds': -130  # This should be +EV for OVER
        }
    ]
    
    print(f"\nMarket Lines: {len(market_odds)} props to evaluate")
    print(f"Edge Threshold: {MODEL_CONFIG.get('prop_edge_threshold', 0.03):.1%}")
    
    # Initialize BetFinder
    bet_finder = BetFinder(
        model_predictions=predictions_df,
        market_odds=market_odds
    )
    
    # Find bets
    print("\n" + "=" * 70)
    print("SCANNING FOR +EV OPPORTUNITIES")
    print("=" * 70)
    
    recommended_bets = bet_finder.find_bets()
    
    # Display results
    print("\n" + "=" * 70)
    print("RECOMMENDED BETS")
    print("=" * 70)
    
    if recommended_bets:
        print(f"\nFound {len(recommended_bets)} bet(s) meeting criteria:\n")
        pprint(recommended_bets)
        
        # Display summary statistics
        bets_df = pd.DataFrame(recommended_bets)
        print("\n" + "=" * 70)
        print("SUMMARY STATISTICS")
        print("=" * 70)
        print(f"Total bets: {len(bets_df)}")
        print(f"Average edge: {bets_df['edge'].mean():.2%}")
        print(f"Total expected value: ${bets_df['expected_value'].sum():.2f}")
        print(f"Total Kelly allocation: {bets_df['kelly_fraction'].sum():.2%}")
        print(f"Average Kelly per bet: {bets_df['kelly_fraction'].mean():.2%}")
        
        # Save to log file
        print("\n" + "=" * 70)
        print("SAVING BET LOG")
        print("=" * 70)
        bet_finder.save_log()
        
    else:
        print("\nNo bets found meeting the criteria.")
        print(f"(Edge threshold: {MODEL_CONFIG.get('prop_edge_threshold', 0.03):.1%})")
    
    print("\n" + "=" * 70)
    print("Phase 2 example complete!")
    print("=" * 70)



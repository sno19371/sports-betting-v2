# feature_engineering.py
"""
Feature engineering for NFL prop betting system.
Transforms raw player data into model-ready features.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict
import logging
from config import MODEL_CONFIG

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering pipeline for NFL player prop betting.
    Transforms raw weekly player statistics into predictive features.
    """
    
    def __init__(self, min_games: int = None):
        """
        Initialize the feature engineer.
        
        Args:
            min_games: Minimum number of games required for feature calculation
        """
        self.min_games = min_games or MODEL_CONFIG.get('min_sample_size', 5)
        
        # Define key stats for rolling calculations
        self.core_stats = [
            'targets', 'carries', 'receiving_yards', 'rushing_yards',
            'receptions', 'receiving_tds', 'rushing_tds', 
            'passing_yards', 'passing_tds', 'completions', 'passing_att'
        ]
        
        logger.info(f"FeatureEngineer initialized (min_games={self.min_games})")
    
    # ==========================================================================
    # PHASE 1: FOUNDATIONAL PLAYER METRICS
    # ==========================================================================
    
    def add_rolling_averages(
        self, 
        df: pd.DataFrame, 
        window: int = 4,
        stats: List[str] = None
    ) -> pd.DataFrame:
        """
        Add rolling average features with proper shifting to prevent data leakage.
        
        Rolling averages capture recent performance trends over a fixed window.
        We shift(1) so that for week N, we only use data from weeks N-1, N-2, etc.
        
        Args:
            df: DataFrame with player stats (must be sorted by player, season, week)
            window: Rolling window size (default: 4 games)
            stats: List of stat columns to calculate rolling averages for
                   If None, uses self.core_stats
        
        Returns:
            DataFrame with added rolling average columns
        """
        if stats is None:
            stats = self.core_stats
        
        # Filter to only stats that exist in the dataframe
        available_stats = [stat for stat in stats if stat in df.columns]
        
        if not available_stats:
            logger.warning("No valid stats found for rolling averages")
            return df
        
        logger.info(f"Adding rolling averages (window={window}) for {len(available_stats)} stats")
        
        df = df.copy()
        
        # Group by player to ensure rolling windows don't cross players
        player_col = self._get_player_column(df)
        
        for stat in available_stats:
            # Calculate rolling mean with shift to prevent leakage
            # Use apply() to ensure shift, rolling, and mean all happen within each group
            df[f'{stat}_rolling_{window}g'] = df.groupby(player_col, group_keys=False).apply(
                lambda g: g[stat].shift(1).rolling(window=window, min_periods=1).mean()
            )
        
        return df
    
    def add_ewma_features(
        self,
        df: pd.DataFrame,
        span: int = 4,
        stats: List[str] = None
    ) -> pd.DataFrame:
        """
        Add Exponentially Weighted Moving Average (EWMA) features.
        
        EWMA gives more weight to recent games, making it more responsive to
        changes in form compared to simple rolling averages.
        
        Formula: EWMA_t = alpha * x_t + (1 - alpha) * EWMA_(t-1)
        where alpha = 2 / (span + 1)
        
        Args:
            df: DataFrame with player stats (must be sorted by player, season, week)
            span: Span parameter for EWMA (default: 4)
                  Higher span = more smoothing, less weight on recent values
            stats: List of stat columns to calculate EWMA for
                   If None, uses self.core_stats
        
        Returns:
            DataFrame with added EWMA columns
        """
        if stats is None:
            stats = self.core_stats
        
        # Filter to only stats that exist in the dataframe
        available_stats = [stat for stat in stats if stat in df.columns]
        
        if not available_stats:
            logger.warning("No valid stats found for EWMA")
            return df
        
        logger.info(f"Adding EWMA features (span={span}) for {len(available_stats)} stats")
        
        df = df.copy()
        
        # Group by player to ensure EWMA doesn't cross players
        player_col = self._get_player_column(df)
        
        for stat in available_stats:
            # Calculate EWMA with shift to prevent leakage
            # Use apply() to ensure shift, ewm, and mean all happen within each group
            df[f'{stat}_ewma_{span}'] = df.groupby(player_col, group_keys=False).apply(
                lambda g: g[stat].shift(1).ewm(span=span, adjust=False).mean()
            )
        
        return df
    
    # ==========================================================================
    # PHASE 2: GAME CONTEXT FEATURES
    # ==========================================================================
    
    def add_matchup_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Defense vs Position (DvP) matchup features.
        
        Calculates how difficult each opponent's defense is against the player's
        position, using percentile ranks. Higher percentile = easier matchup
        (defense allows more production).
        
        Args:
            df: DataFrame with player stats, must include 'opponent' and 'position' columns
            
        Returns:
            DataFrame with added matchup difficulty features
        """
        if 'opponent' not in df.columns or 'position' not in df.columns:
            logger.warning("Missing 'opponent' or 'position' columns, skipping matchup features")
            return df
        
        logger.info("Adding matchup features (DvP)")
        
        df = df.copy()
        
        # Define stat-to-position mappings for defensive rankings
        position_stats = {
            'QB': ['passing_yards', 'passing_tds'],
            'RB': ['rushing_yards', 'rushing_tds', 'receiving_yards'],
            'WR': ['receiving_yards', 'receiving_tds', 'receptions'],
            'TE': ['receiving_yards', 'receiving_tds', 'receptions']
        }
        
        # Track all position-specific columns we create
        all_position_specific_cols = []
        
        # Create defensive profile for each stat-position combination
        for pos, stats in position_stats.items():
            for stat in stats:
                if stat not in df.columns:
                    continue
                
                # Use position-specific column name to avoid merge conflicts
                rank_col_name = f'matchup_{pos}_{stat}_rank'
                all_position_specific_cols.append(rank_col_name)
                
                # Group by season, opponent, and position to calculate defense allowed
                defensive_profile = (
                    df[df['position'] == pos]
                    .groupby(['season', 'opponent'])[stat]
                    .mean()
                    .reset_index()
                    .rename(columns={stat: f'{stat}_allowed'})
                )
                
                # Convert to percentile ranks within each season
                # Higher percentile = defense allows MORE (easier matchup)
                defensive_profile[rank_col_name] = (
                    defensive_profile.groupby('season')[f'{stat}_allowed']
                    .rank(pct=True)
                )
                
                # Merge this position-specific rank back to main dataframe
                df = df.merge(
                    defensive_profile[['season', 'opponent', rank_col_name]],
                    on=['season', 'opponent'],
                    how='left'
                )
        
        # Coalesce position-specific columns into generic stat columns
        # For each stat, find the relevant rank based on player's position
        for stat in self.core_stats:
            # Find all position-specific rank columns for this stat
            pos_specific_cols = [
                f'matchup_{pos}_{stat}_rank' 
                for pos in position_stats 
                if stat in position_stats[pos] and f'matchup_{pos}_{stat}_rank' in df.columns
            ]
            
            if pos_specific_cols:
                # Use bfill to get the first non-null value across position-specific columns
                # Each row will only have one non-null value (matching their position)
                df[f'matchup_{stat}_rank'] = df[pos_specific_cols].bfill(axis=1).iloc[:, 0]
                
                # Drop the position-specific columns now that we have the generic one
                df = df.drop(columns=pos_specific_cols)
        
        # Fill NaN values (e.g., first game of season) with median = 0.5
        matchup_cols = [col for col in df.columns if col.startswith('matchup_') and col.endswith('_rank')]
        for col in matchup_cols:
            df[col] = df[col].fillna(0.5)
        
        logger.info(f"Added {len(matchup_cols)} matchup features")
        
        return df
    
    def add_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add weather impact features.
        
        Creates a passing penalty multiplier based on adverse weather conditions.
        Assumes game-level weather data has already been merged.
        
        Args:
            df: DataFrame with weather columns (wind_speed, temp, is_dome)
            
        Returns:
            DataFrame with weather impact features
        """
        logger.info("Adding weather features")
        
        df = df.copy()
        
        # Initialize passing penalty to 1.0 (no penalty)
        df['passing_penalty'] = 1.0
        
        # Check if required columns exist
        has_wind = 'wind_speed' in df.columns
        has_temp = 'temp' in df.columns
        has_dome = 'is_dome' in df.columns
        
        if not (has_wind or has_temp):
            logger.warning("No weather columns found, passing_penalty set to 1.0")
            return df
        
        # Apply weather penalties for outdoor games
        if has_dome:
            # Only apply penalties to outdoor games
            outdoor_mask = df['is_dome'] == False
        else:
            # Assume all outdoor if is_dome not present
            outdoor_mask = pd.Series([True] * len(df), index=df.index)
        
        # Apply penalty for adverse conditions
        if has_wind and has_temp:
            adverse_weather = outdoor_mask & ((df['wind_speed'] > 15) | (df['temp'] < 32))
        elif has_wind:
            adverse_weather = outdoor_mask & (df['wind_speed'] > 15)
        elif has_temp:
            adverse_weather = outdoor_mask & (df['temp'] < 32)
        else:
            adverse_weather = pd.Series([False] * len(df), index=df.index)
        
        df.loc[adverse_weather, 'passing_penalty'] = 0.85
        
        logger.info(f"Weather penalties applied to {adverse_weather.sum()} games")
        
        return df
    
    def add_game_script_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add game script features based on Vegas spread.
        
        Uses the point spread to predict likely game flow:
        - Heavy underdogs likely to pass more (trailing, playing catch-up)
        - Heavy favorites likely to run more (leading, running out clock)
        
        Args:
            df: DataFrame with spread_line column (from team's perspective)
            
        Returns:
            DataFrame with game script features
        """
        if 'spread_line' not in df.columns:
            logger.warning("Missing 'spread_line' column, skipping game script features")
            df['game_script_factor'] = 0
            return df
        
        logger.info("Adding game script features")
        
        df = df.copy()
        
        # Create game script factor
        # Positive spread = underdog (expected to pass more)
        # Negative spread = favorite (expected to run more)
        df['game_script_factor'] = np.where(
            df['spread_line'] >= 7,
            1,  # Heavy underdog - passing script
            np.where(
                df['spread_line'] <= -7,
                -1,  # Heavy favorite - running script
                0   # Neutral game script
            )
        )
        
        script_counts = df['game_script_factor'].value_counts()
        logger.info(f"Game scripts - Passing: {script_counts.get(1, 0)}, "
                   f"Neutral: {script_counts.get(0, 0)}, Running: {script_counts.get(-1, 0)}")
        
        return df
    
    # ==========================================================================
    # HELPER METHODS
    # ==========================================================================
    
    def _get_player_column(self, df: pd.DataFrame) -> str:
        """
        Determine which column contains player names.
        
        Args:
            df: DataFrame to check
            
        Returns:
            Name of the player identifier column
        """
        if 'player_display_name' in df.columns:
            return 'player_display_name'
        elif 'player_name' in df.columns:
            return 'player_name'
        elif 'player_id' in df.columns:
            return 'player_id'
        else:
            raise ValueError("No player identifier column found in DataFrame")
    
    def validate_dataframe(self, df: pd.DataFrame) -> bool:
        """
        Validate that DataFrame is properly formatted for feature engineering.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if valid, raises ValueError if invalid
        """
        required_cols = ['season', 'week']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check for player identifier
        try:
            self._get_player_column(df)
        except ValueError as e:
            raise ValueError(f"DataFrame validation failed: {e}")
        
        # Check if data is sorted
        player_col = self._get_player_column(df)
        if not df.groupby(player_col).apply(
            lambda x: x[['season', 'week']].equals(x[['season', 'week']].sort_values(['season', 'week']))
        ).all():
            logger.warning("DataFrame is not sorted by player, season, week. Sorting now...")
            df.sort_values([player_col, 'season', 'week'], inplace=True)
        
        return True
    
    # ==========================================================================
    # MAIN PIPELINE
    # ==========================================================================
    
    def create_features(
        self,
        weekly_df: pd.DataFrame,
        schedule_df: Optional[pd.DataFrame] = None,
        include_rolling: bool = True,
        include_ewma: bool = True,
        include_matchup: bool = True,
        include_weather: bool = True,
        include_game_script: bool = True
    ) -> pd.DataFrame:
        """
        Main feature engineering pipeline - Phases 1 & 2.
        
        Args:
            weekly_df: Raw weekly player statistics DataFrame
            schedule_df: Game-level data with home_team, away_team columns
                        Must have columns: season, week, home_team, away_team
                        Optional columns: spread_line, roof (or is_dome)
            include_rolling: Whether to add rolling average features
            include_ewma: Whether to add EWMA features
            include_matchup: Whether to add DvP matchup features
            include_weather: Whether to add weather features
            include_game_script: Whether to add game script features
        
        Returns:
            DataFrame with engineered features from all phases
        """
        # Validate input
        self.validate_dataframe(weekly_df)
        
        logger.info(f"Starting feature engineering pipeline on {len(weekly_df)} rows")
        
        result_df = weekly_df.copy()
        
        # --- Standardize team column and index to enable safe merges and operations ---
        if 'recent_team' in result_df.columns and 'team' not in result_df.columns:
            result_df = result_df.rename(columns={'recent_team': 'team'})
            logger.info("Standardized 'recent_team' to 'team' for merging.")
        
        # Ensure a unique, monotonic index to avoid reindex errors after groupby/apply
        result_df = result_df.reset_index(drop=True)
        
        # Merge schedule data if provided
        if schedule_df is not None and not schedule_df.empty:
            logger.info("Merging schedule data")
            
            # Reshape schedule_df to have one row per team per game
            # This prevents row explosion from many-to-many merges
            
            if 'home_team' in schedule_df.columns and 'away_team' in schedule_df.columns:
                logger.info("Reshaping schedule data (home/away to team/opponent)")
                
                # Select columns to keep (excluding team-specific ones for now)
                schedule_cols = ['season', 'week']
                optional_cols = ['spread_line', 'roof', 'wind', 'temp']
                
                for col in optional_cols:
                    if col in schedule_df.columns:
                        schedule_cols.append(col)
                
                # Create home team schedule
                schedule_home = schedule_df[schedule_cols + ['home_team', 'away_team']].copy()
                schedule_home = schedule_home.rename(columns={
                    'home_team': 'team',
                    'away_team': 'opponent'
                })
                schedule_home['is_home'] = True
                
                # For home team, spread_line stays as is (positive = underdog, negative = favorite)
                # Already correct from home team perspective
                
                # Create away team schedule
                schedule_away = schedule_df[schedule_cols + ['home_team', 'away_team']].copy()
                schedule_away = schedule_away.rename(columns={
                    'away_team': 'team',
                    'home_team': 'opponent'
                })
                schedule_away['is_home'] = False
                
                # For away team, flip the spread_line (what's -7 for home is +7 for away)
                if 'spread_line' in schedule_away.columns:
                    schedule_away['spread_line'] = -schedule_away['spread_line']
                
                # Combine into single schedule with one row per team per game
                schedule_for_merge = pd.concat([schedule_home, schedule_away], ignore_index=True)
                
                logger.info(f"Reshaped schedule: {len(schedule_df)} games -> {len(schedule_for_merge)} team-games")
                
            else:
                # If already in team/opponent format, use as is
                logger.info("Schedule already in team/opponent format")
                schedule_for_merge = schedule_df.copy()
            
            # Perform clean merge on season, week, team
            if 'team' in result_df.columns:
                merge_keys = ['season', 'week', 'team']
                
                result_df = result_df.merge(
                    schedule_for_merge,
                    on=merge_keys,
                    how='left',
                    suffixes=('', '_schedule')
                )
                
                logger.info(f"Merged schedule data: {result_df.shape}")
                
                # Convert roof to is_dome if needed
                if 'roof' in result_df.columns and 'is_dome' not in result_df.columns:
                    # roof types: 'outdoors', 'dome', 'open', 'closed'
                    result_df['is_dome'] = result_df['roof'].isin(['dome', 'closed'])
                    logger.info("Converted 'roof' column to 'is_dome'")
            else:
                logger.warning("No 'team' column in weekly_df, cannot merge schedule data")
        
        # Phase 1: Foundational Player Metrics
        if include_rolling:
            result_df = self.add_rolling_averages(result_df, window=4)
        
        if include_ewma:
            result_df = self.add_ewma_features(result_df, span=4)
        
        # Phase 2: Game Context Features
        if include_matchup:
            result_df = self.add_matchup_features(result_df)
        
        if include_weather:
            result_df = self.add_weather_features(result_df)
        
        if include_game_script:
            result_df = self.add_game_script_features(result_df)
        
        # Count features added
        original_cols = set(weekly_df.columns)
        new_cols = set(result_df.columns) - original_cols
        
        logger.info(f"Pipeline complete: Added {len(new_cols)} new features")
        logger.info(f"Final shape: {result_df.shape}")
        
        return result_df



# modeling.py
"""
Predictive modeling for NFL player props.
Implements quantile regression models for prop betting.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import logging
from typing import List, Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class PlayerPropModel:
    """
    Single-prop quantile regression model using LightGBM.
    
    This class trains separate quantile regression models for each specified
    quantile, allowing us to predict full probability distributions rather than
    just point estimates.
    
    Example:
        >>> model = PlayerPropModel('receiving_yards', [0.1, 0.5, 0.9])
        >>> model.train(X_train, y_train)
        >>> predictions = model.predict(X_test)
    """
    
    def __init__(self, target_prop: str, quantiles: List[float]):
        """
        Initialize the player prop model.
        
        Args:
            target_prop: Name of the target column to predict (e.g., 'receiving_yards')
            quantiles: List of quantiles to predict (e.g., [0.1, 0.25, 0.5, 0.75, 0.9])
        """
        self.target_prop = target_prop
        self.quantiles = sorted(quantiles)  # Ensure quantiles are sorted
        self.models: Dict[float, lgb.LGBMRegressor] = {}
        
        logger.info(f"Initialized PlayerPropModel for '{target_prop}' with quantiles: {quantiles}")
    
    def train(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        lgb_params: Optional[Dict] = None
    ) -> None:
        """
        Train quantile regression models for each specified quantile.
        
        Args:
            X: Feature matrix (pandas DataFrame)
            y: Target values (pandas Series)
            lgb_params: Optional dictionary of additional LightGBM parameters
        """
        if len(X) != len(y):
            raise ValueError(f"X and y must have same length. Got X={len(X)}, y={len(y)}")
        
        if len(X) == 0:
            raise ValueError("Cannot train on empty dataset")
        
        logger.info(f"Training {len(self.quantiles)} quantile models on {len(X)} samples")
        
        # Default LightGBM parameters
        default_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'n_jobs': -1,
            'random_state': 42,
            'verbose': -1
        }
        
        # Merge with user-provided parameters
        if lgb_params:
            default_params.update(lgb_params)
        
        # Train a separate model for each quantile
        for quantile in self.quantiles:
            logger.info(f"  Training quantile {quantile:.2f} model...")
            
            # Create quantile regressor
            model = lgb.LGBMRegressor(
                objective='quantile',
                alpha=quantile,
                **default_params
            )
            
            # Fit the model
            model.fit(X, y)
            
            # Store the trained model
            self.models[quantile] = model
            
            logger.info(f"  Quantile {quantile:.2f} model trained successfully")
        
        logger.info(f"All {len(self.quantiles)} models trained for '{self.target_prop}'")
    
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Generate quantile predictions for new data.
        
        Args:
            X: Feature matrix for prediction
            
        Returns:
            DataFrame with one column per quantile, formatted as '{target_prop}_q{quantile}'
        """
        if not self.models:
            raise ValueError("No trained models found. Call .train() first.")
        
        if len(X) == 0:
            raise ValueError("Cannot predict on empty dataset")
        
        logger.info(f"Generating predictions for {len(X)} samples")
        
        # Initialize predictions DataFrame
        predictions_df = pd.DataFrame(index=X.index)
        
        # Generate predictions for each quantile
        for quantile, model in self.models.items():
            col_name = f'{self.target_prop}_q{quantile}'
            predictions_df[col_name] = model.predict(X)
            logger.info(f"  Generated predictions for quantile {quantile:.2f}")
        
        logger.info(f"Predictions complete: {predictions_df.shape}")
        
        return predictions_df
    
    def save_model(self, path: str) -> None:
        """
        Save trained models to disk.
        
        Args:
            path: File path to save the model (will be saved as .pkl file)
        """
        if not self.models:
            raise ValueError("No trained models to save. Call .train() first.")
        
        # Ensure path has .pkl extension
        path = Path(path)
        if path.suffix != '.pkl':
            path = path.with_suffix('.pkl')
        
        # Create directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the models dictionary
        model_data = {
            'target_prop': self.target_prop,
            'quantiles': self.quantiles,
            'models': self.models
        }
        
        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """
        Load trained models from disk.
        
        Args:
            path: File path to load the model from
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        # Load the models dictionary
        model_data = joblib.load(path)
        
        self.target_prop = model_data['target_prop']
        self.quantiles = model_data['quantiles']
        self.models = model_data['models']
        
        logger.info(f"Model loaded from {path}")
        logger.info(f"  Target prop: {self.target_prop}")
        logger.info(f"  Quantiles: {self.quantiles}")
        logger.info(f"  Models loaded: {len(self.models)}")
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance from the median (0.5) quantile model.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature names and importance scores
        """
        if 0.5 not in self.models:
            # Use the first available model if 0.5 not present
            quantile = list(self.models.keys())[0]
            logger.warning(f"Quantile 0.5 not found, using {quantile} instead")
        else:
            quantile = 0.5
        
        model = self.models[quantile]
        
        # Get feature importance
        importance_df = pd.DataFrame({
            'feature': model.feature_name_,
            'importance': model.feature_importances_
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)


class ModelManager:
    """
    Manager class for handling multiple PlayerPropModel instances.
    
    This orchestrator trains and manages separate quantile models for each
    target prop (receiving_yards, rushing_yards, etc.), providing a unified
    interface for training and prediction across all props.
    
    Example:
        >>> manager = ModelManager(['receiving_yards', 'rushing_yards'], [0.1, 0.5, 0.9])
        >>> manager.train_all(X_train, y_train, positions)
        >>> predictions = manager.predict_all(X_test)
    """
    
    def __init__(self, target_props: List[str], quantiles: List[float]):
        """
        Initialize the model manager.
        
        Args:
            target_props: List of target prop names to model
                         (e.g., ['receiving_yards', 'rushing_yards'])
            quantiles: List of quantiles to predict for each prop
        """
        self.target_props = target_props
        self.quantiles = quantiles
        self.managers: Dict[str, PlayerPropModel] = {}
        self.feature_names: Optional[List[str]] = None
        
        logger.info(f"Initialized ModelManager for {len(target_props)} props: {target_props}")
        logger.info(f"Quantiles: {quantiles}")
    
    def train_all(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        positions: pd.Series,
        lgb_params: Optional[Dict] = None
    ) -> None:
        """
        Train models for all target props.
        
        For each prop, this method:
        1. Filters out rows where the target is NaN (not all players have all stats)
        2. Filters by relevant positions for each prop (e.g., only WR/TE/RB for receiving_yards)
        3. Trains a PlayerPropModel on the cleaned data
        4. Stores the trained model
        
        Args:
            X: Feature matrix (pandas DataFrame)
            y: DataFrame containing all target columns
            positions: Series with player positions (same index as X and y)
            lgb_params: Optional dictionary of LightGBM parameters
        """
        logger.info(f"Training models for {len(self.target_props)} props")
        
        # Capture feature names from training data
        self.feature_names = list(X.columns)
        logger.info(f"Captured {len(self.feature_names)} feature names for inference")
        
        # Define position filters for each prop
        # Only train on positions that actually produce meaningful values for each stat
        position_filter = {
            'receiving_yards': ['WR', 'TE', 'RB', 'FB'],
            'receiving_tds': ['WR', 'TE', 'RB', 'FB'],
            'receptions': ['WR', 'TE', 'RB', 'FB'],
            'targets': ['WR', 'TE', 'RB', 'FB'],
            'rushing_yards': ['RB', 'QB', 'WR', 'FB'],
            'rushing_tds': ['RB', 'QB', 'WR', 'FB'],
            'carries': ['RB', 'QB', 'WR', 'FB'],
            'passing_yards': ['QB'],
            'passing_tds': ['QB'],
            'completions': ['QB'],
            'passing_att': ['QB'],
            'interceptions': ['QB']
        }
        
        for prop in self.target_props:
            logger.info(f"\n{'='*60}")
            logger.info(f"Training model for: {prop}")
            logger.info(f"{'='*60}")
            
            # Check if prop exists in y
            if prop not in y.columns:
                logger.warning(f"Target prop '{prop}' not found in y DataFrame. Skipping.")
                continue
            
            # Filter out rows where target is NaN
            # This handles position-specific props (e.g., QBs don't have receiving_yards)
            valid_mask = y[prop].notna()
            
            # Apply position filter if defined for this prop
            if prop in position_filter:
                relevant_positions = position_filter[prop]
                position_mask = positions.isin(relevant_positions)
                
                # Combine both filters
                combined_mask = valid_mask & position_mask
                
                filtered_count = valid_mask.sum() - combined_mask.sum()
                logger.info(f"Position filter: keeping {relevant_positions}")
                logger.info(f"  Filtered out {filtered_count} samples from irrelevant positions")
            else:
                # No position filter for this prop
                combined_mask = valid_mask
                logger.info(f"No position filter defined for '{prop}'")
            
            X_train_prop = X[combined_mask]
            y_train_prop = y.loc[combined_mask, prop]
            
            logger.info(f"Training samples: {len(X_train_prop)} "
                       f"(filtered {(~combined_mask).sum()} total invalid/irrelevant)")
            
            if len(X_train_prop) == 0:
                logger.warning(f"No valid training data for '{prop}'. Skipping.")
                continue
            
            # Show position distribution in training data
            if prop in position_filter:
                pos_counts = positions[combined_mask].value_counts()
                logger.info(f"Position distribution: {pos_counts.to_dict()}")
            
            # Instantiate and train model for this prop
            prop_model = PlayerPropModel(
                target_prop=prop,
                quantiles=self.quantiles
            )
            
            prop_model.train(X_train_prop, y_train_prop, lgb_params)
            
            # Store the trained model
            self.managers[prop] = prop_model
            
            logger.info(f"Model for '{prop}' trained and stored successfully")
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Training complete: {len(self.managers)}/{len(self.target_props)} models trained")
        logger.info(f"{'='*60}")
    
    def predict_all(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions for all trained models.
        
        Args:
            X: Feature matrix for prediction
            
        Returns:
            Combined DataFrame with quantile predictions for all props
        """
        if not self.managers:
            raise ValueError("No trained models found. Call .train_all() first.")
        
        logger.info(f"Generating predictions for {len(self.managers)} props")
        
        prediction_dfs = []
        
        for prop, manager in self.managers.items():
            logger.info(f"  Predicting {prop}...")
            prop_predictions = manager.predict(X)
            prediction_dfs.append(prop_predictions)
        
        # Combine all predictions into single DataFrame
        combined_predictions = pd.concat(prediction_dfs, axis=1)
        
        logger.info(f"Combined predictions: {combined_predictions.shape}")
        
        return combined_predictions
    
    def save_manager(self, path: str) -> None:
        """
        Save the entire ModelManager instance to disk.
        
        Args:
            path: File path to save the manager
        """
        if not self.managers:
            raise ValueError("No trained models to save. Call .train_all() first.")
        
        # Ensure path has .pkl extension
        path = Path(path)
        if path.suffix != '.pkl':
            path = path.with_suffix('.pkl')
        
        # Create directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save entire manager instance
        joblib.dump(self, path)
        logger.info(f"ModelManager saved to {path}")
        logger.info(f"  Saved {len(self.managers)} trained models")
        if self.feature_names:
            logger.info(f"  Saved {len(self.feature_names)} feature names")
    
    @staticmethod
    def load_manager(path: str) -> 'ModelManager':
        """
        Load a ModelManager instance from disk.
        
        Args:
            path: File path to load the manager from
            
        Returns:
            Loaded ModelManager instance with all trained models
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Manager file not found: {path}")
        
        # Load the manager instance
        manager = joblib.load(path)
        
        logger.info(f"ModelManager loaded from {path}")
        logger.info(f"  Target props: {manager.target_props}")
        logger.info(f"  Quantiles: {manager.quantiles}")
        logger.info(f"  Loaded {len(manager.managers)} trained models")
        
        # Log feature names if available
        if hasattr(manager, 'feature_names') and manager.feature_names:
            logger.info(f"  Feature names: {len(manager.feature_names)} features")
        else:
            logger.warning("  No feature names saved (older model version)")
        
        return manager
    
    def get_all_feature_importances(self, top_n: int = 10) -> Dict[str, pd.DataFrame]:
        """
        Get feature importances for all trained models.
        
        Args:
            top_n: Number of top features to return per model
            
        Returns:
            Dictionary mapping prop names to their feature importance DataFrames
        """
        importances = {}
        
        for prop, manager in self.managers.items():
            importances[prop] = manager.get_feature_importance(top_n)
        
        return importances


if __name__ == "__main__":
    """
    Example usage demonstrating both PlayerPropModel and ModelManager functionality.
    """
    print("=" * 70)
    print("NFL PROP MODELING - PHASE 2 EXAMPLE")
    print("=" * 70)
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Create synthetic training data with multiple props
    np.random.seed(42)
    n_samples = 1000
    
    # Generate features
    X_train = pd.DataFrame({
        'targets_rolling_4g': np.random.uniform(5, 12, n_samples),
        'receiving_yards_rolling_4g': np.random.uniform(40, 120, n_samples),
        'rushing_yards_rolling_4g': np.random.uniform(20, 80, n_samples),
        'targets_ewma_4': np.random.uniform(5, 12, n_samples),
        'matchup_receiving_yards_rank': np.random.uniform(0.2, 0.8, n_samples),
        'matchup_rushing_yards_rank': np.random.uniform(0.2, 0.8, n_samples),
        'game_script_factor': np.random.choice([-1, 0, 1], n_samples),
        'passing_penalty': np.random.choice([0.85, 1.0], n_samples, p=[0.1, 0.9])
    })
    
    # Generate multiple targets with realistic correlations
    # Simulate that some players are WRs (have receiving_yards, no rushing_yards)
    # and some are RBs (have both, but more rushing)
    player_type = np.random.choice(['WR', 'RB'], n_samples, p=[0.6, 0.4])
    
    # Create positions Series
    positions = pd.Series(player_type, name='position')
    
    # Receiving yards target
    receiving_yards = (
        X_train['targets_rolling_4g'] * 8 +
        X_train['receiving_yards_rolling_4g'] * 0.3 +
        X_train['matchup_receiving_yards_rank'] * 20 +
        np.random.normal(0, 15, n_samples)
    ).clip(0, 200)
    
    # Rushing yards target
    rushing_yards = (
        X_train['rushing_yards_rolling_4g'] * 0.5 +
        X_train['matchup_rushing_yards_rank'] * 15 +
        X_train['game_script_factor'] * 10 +
        np.random.normal(0, 20, n_samples)
    ).clip(0, 150)
    
    # Create y_train DataFrame with NaN values for position-specific props
    y_train = pd.DataFrame({
        'receiving_yards': receiving_yards,
        'rushing_yards': rushing_yards
    })
    
    # WRs don't have rushing yards (set to NaN)
    y_train.loc[player_type == 'WR', 'rushing_yards'] = np.nan
    # Some RBs have minimal receiving yards
    y_train.loc[player_type == 'RB', 'receiving_yards'] = np.where(
        np.random.random(sum(player_type == 'RB')) < 0.3,
        np.nan,
        y_train.loc[player_type == 'RB', 'receiving_yards']
    )
    
    print(f"\nTraining data shape: {X_train.shape}")
    print(f"\nTarget stats:")
    print(y_train.describe())
    print(f"\nNaN counts:")
    print(y_train.isna().sum())
    print(f"\nPosition distribution:")
    print(positions.value_counts())
    
    # =========================================================================
    # DEMONSTRATE MODELMANAGER
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("TRAINING MODELMANAGER")
    print("=" * 70)
    
    # Initialize ModelManager for multiple props
    manager = ModelManager(
        target_props=['receiving_yards', 'rushing_yards'],
        quantiles=[0.1, 0.25, 0.5, 0.75, 0.9]
    )
    
    # Train all models
    manager.train_all(X_train, y_train, positions)
    
    # Create test data
    X_test = pd.DataFrame({
        'targets_rolling_4g': [8.5, 10.2, 6.8],
        'receiving_yards_rolling_4g': [85.0, 95.0, 65.0],
        'rushing_yards_rolling_4g': [45.0, 55.0, 35.0],
        'targets_ewma_4': [8.2, 10.5, 7.1],
        'matchup_receiving_yards_rank': [0.7, 0.4, 0.6],
        'matchup_rushing_yards_rank': [0.5, 0.6, 0.4],
        'game_script_factor': [1, 0, -1],
        'passing_penalty': [1.0, 1.0, 0.85]
    })
    
    print("\n" + "=" * 70)
    print("PREDICTIONS FROM MODELMANAGER")
    print("=" * 70)
    
    # Generate predictions for all props
    predictions = manager.predict_all(X_test)
    print("\nCombined predictions for all props:")
    print(predictions.round(1))
    
    # Display feature importances for all models
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCES")
    print("=" * 70)
    
    importances = manager.get_all_feature_importances(top_n=5)
    for prop, importance_df in importances.items():
        print(f"\nTop 5 features for {prop}:")
        print(importance_df)
    
    # Test save/load functionality
    print("\n" + "=" * 70)
    print("TESTING SAVE/LOAD MANAGER")
    print("=" * 70)
    
    manager_path = "models/test_model_manager.pkl"
    manager.save_manager(manager_path)
    
    # Load manager
    loaded_manager = ModelManager.load_manager(manager_path)
    
    # Verify loaded manager works
    loaded_predictions = loaded_manager.predict_all(X_test)
    predictions_match = predictions.equals(loaded_predictions)
    print(f"\nLoaded manager predictions match original: {predictions_match}")
    
    # Verify feature names were saved and loaded
    if hasattr(loaded_manager, 'feature_names') and loaded_manager.feature_names:
        print(f"\n✓ Feature names preserved: {len(loaded_manager.feature_names)} features")
        print(f"  First 5 features: {loaded_manager.feature_names[:5]}")
    
    # =========================================================================
    # DEMONSTRATE SINGLE PLAYERPROPMODEL (from Phase 1)
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("SINGLE PLAYERPROPMODEL EXAMPLE (Phase 1)")
    print("=" * 70)
    
    # Train single model for receiving_yards only
    single_model = PlayerPropModel('receiving_yards', [0.1, 0.5, 0.9])
    
    # Use only rows with valid receiving_yards
    valid_mask = y_train['receiving_yards'].notna()
    single_model.train(X_train[valid_mask], y_train.loc[valid_mask, 'receiving_yards'])
    
    # Predict
    single_predictions = single_model.predict(X_test)
    print("\nSingle model predictions (receiving_yards only):")
    print(single_predictions.round(1))
    
    print("\n" + "=" * 70)
    print("Phase 2 example complete!")
    print("=" * 70)


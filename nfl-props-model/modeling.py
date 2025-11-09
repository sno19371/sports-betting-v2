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


if __name__ == "__main__":
    """
    Example usage demonstrating the PlayerPropModel functionality.
    """
    print("=" * 70)
    print("PLAYER PROP MODEL - PHASE 1 EXAMPLE")
    print("=" * 70)
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create synthetic training data
    np.random.seed(42)
    n_samples = 1000
    
    # Generate features
    X_train = pd.DataFrame({
        'targets_rolling_4g': np.random.uniform(5, 12, n_samples),
        'receiving_yards_rolling_4g': np.random.uniform(40, 120, n_samples),
        'targets_ewma_4': np.random.uniform(5, 12, n_samples),
        'matchup_receiving_yards_rank': np.random.uniform(0.2, 0.8, n_samples),
        'game_script_factor': np.random.choice([-1, 0, 1], n_samples),
        'passing_penalty': np.random.choice([0.85, 1.0], n_samples, p=[0.1, 0.9])
    })
    
    # Generate target (receiving yards) with some correlation to features
    y_train = (
        X_train['targets_rolling_4g'] * 8 +
        X_train['receiving_yards_rolling_4g'] * 0.3 +
        X_train['matchup_receiving_yards_rank'] * 20 +
        np.random.normal(0, 15, n_samples)
    )
    y_train = y_train.clip(0, 200)  # Realistic bounds for receiving yards
    
    print(f"\nTraining data shape: {X_train.shape}")
    print(f"Target stats: mean={y_train.mean():.1f}, std={y_train.std():.1f}")
    
    # Initialize model
    model = PlayerPropModel(
        target_prop='receiving_yards',
        quantiles=[0.1, 0.25, 0.5, 0.75, 0.9]
    )
    
    # Train the model
    print("\nTraining models...")
    model.train(X_train, y_train)
    
    # Create synthetic test data
    X_test = pd.DataFrame({
        'targets_rolling_4g': [8.5, 10.2, 6.8],
        'receiving_yards_rolling_4g': [85.0, 95.0, 65.0],
        'targets_ewma_4': [8.2, 10.5, 7.1],
        'matchup_receiving_yards_rank': [0.7, 0.4, 0.6],
        'game_script_factor': [1, 0, -1],
        'passing_penalty': [1.0, 1.0, 0.85]
    })
    
    print(f"\nTest data shape: {X_test.shape}")
    
    # Generate predictions
    print("\nGenerating predictions...")
    predictions = model.predict(X_test)
    
    print("\n" + "=" * 70)
    print("PREDICTIONS")
    print("=" * 70)
    print(predictions.round(1))
    
    # Display feature importance
    print("\n" + "=" * 70)
    print("TOP 5 FEATURES")
    print("=" * 70)
    print(model.get_feature_importance(top_n=5))
    
    # Test save/load functionality
    print("\n" + "=" * 70)
    print("TESTING SAVE/LOAD")
    print("=" * 70)
    
    model_path = "models/test_receiving_yards_model.pkl"
    model.save_model(model_path)
    print(f"Model saved to: {model_path}")
    
    # Create new model instance and load
    loaded_model = PlayerPropModel('placeholder', [0.5])
    loaded_model.load_model(model_path)
    print(f"Model loaded successfully")
    
    # Verify loaded model works
    loaded_predictions = loaded_model.predict(X_test)
    print(f"Loaded model predictions match: {predictions.equals(loaded_predictions)}")
    
    print("\n" + "=" * 70)
    print("Phase 1 example complete!")
    print("=" * 70)



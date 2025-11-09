# tests/test_modeling.py

import pandas as pd
import numpy as np
import pytest
import joblib
import lightgbm as lgb
from pathlib import Path
from modeling import PlayerPropModel

# ---- Test Fixtures ----

@pytest.fixture
def synthetic_data() -> tuple[pd.DataFrame, pd.Series]:
    """Provides a simple, predictable synthetic dataset for training."""
    np.random.seed(42)
    n_samples = 100
    X = pd.DataFrame({
        'feature1': np.linspace(0, 100, n_samples),
        'feature2': np.random.uniform(-10, 10, n_samples)
    })
    # Create a simple linear relationship with some noise
    y = pd.Series(X['feature1'] * 1.5 + X['feature2'] * 2 + np.random.normal(0, 5, n_samples))
    return X, y

@pytest.fixture
def trained_model(synthetic_data) -> PlayerPropModel:
    """Provides a pre-trained model instance for testing predict, save, etc."""
    X_train, y_train = synthetic_data
    model = PlayerPropModel(
        target_prop='test_prop',
        quantiles=[0.1, 0.5, 0.9]
    )
    model.train(X_train, y_train)
    return model

# ---- Test Functions ----

def test_initialization():
    """Tests that the model initializes correctly."""
    model = PlayerPropModel('receiving_yards', [0.1, 0.5, 0.9])
    assert model.target_prop == 'receiving_yards'
    assert model.quantiles == [0.1, 0.5, 0.9]
    assert model.models == {}

def test_train(synthetic_data):
    """Tests the train method to ensure models are created and stored."""
    X_train, y_train = synthetic_data
    model = PlayerPropModel('test_prop', [0.25, 0.75])
    model.train(X_train, y_train)
    
    # Check that models were created
    assert len(model.models) == 2
    # Check that the keys match the quantiles
    assert set(model.models.keys()) == {0.25, 0.75}
    # Check that the models are trained LightGBM regressors
    assert isinstance(model.models[0.25], lgb.LGBMRegressor)


def test_train_edge_cases():
    """Tests that the train method raises errors on invalid input."""
    model = PlayerPropModel('test_prop', [0.5])
    
    # Test with empty data
    with pytest.raises(ValueError, match="Cannot train on empty dataset"):
        model.train(pd.DataFrame(), pd.Series(dtype='float64'))
        
    # Test with mismatched lengths
    with pytest.raises(ValueError, match="X and y must have same length"):
        model.train(pd.DataFrame({'a': [1, 2]}), pd.Series([1]))

def test_predict(trained_model):
    """Tests the predict method for correct output shape and column names."""
    X_test = pd.DataFrame({'feature1': [25, 50, 75], 'feature2': [5, -5, 0]})
    predictions = trained_model.predict(X_test)
    
    # Check that output is a DataFrame
    assert isinstance(predictions, pd.DataFrame)
    # Check that number of rows matches input
    assert len(predictions) == len(X_test)
    # Check that number of columns matches number of quantiles
    assert len(predictions.columns) == len(trained_model.quantiles)
    # Check column naming convention
    expected_cols = ['test_prop_q0.1', 'test_prop_q0.5', 'test_prop_q0.9']
    assert all(col in predictions.columns for col in expected_cols)

def test_predict_quantile_ordering(trained_model):
    """A crucial test to ensure quantile predictions are monotonic (q0.1 <= q0.5 <= q0.9)."""
    X_test = pd.DataFrame(np.random.rand(10, 2), columns=['feature1', 'feature2'])
    predictions = trained_model.predict(X_test)
    
    # Check that for every row, the quantile predictions are in increasing order
    q0_1 = predictions['test_prop_q0.1']
    q0_5 = predictions['test_prop_q0.5']
    q0_9 = predictions['test_prop_q0.9']
    
    assert (q0_1 <= q0_5).all()
    assert (q0_5 <= q0_9).all()

def test_save_and_load_model(trained_model, tmp_path: Path):
    """
    Tests that a model can be saved and loaded, and that the loaded model
    produces identical predictions.
    """
    model_path = tmp_path / "test_model.pkl"
    
    # 1. Save the original trained model
    trained_model.save_model(str(model_path))
    assert model_path.exists()
    
    # 2. Create a new, empty model instance and load the saved state
    new_model = PlayerPropModel('placeholder', [0.5])
    new_model.load_model(str(model_path))
    
    # 3. Verify that the loaded model has the correct attributes
    assert new_model.target_prop == trained_model.target_prop
    assert new_model.quantiles == trained_model.quantiles
    assert len(new_model.models) == len(trained_model.models)
    
    # 4. Verify that the loaded model produces the exact same predictions
    X_test = pd.DataFrame({'feature1': [30], 'feature2': [-3]})
    original_preds = trained_model.predict(X_test)
    loaded_preds = new_model.predict(X_test)
    
    pd.testing.assert_frame_equal(original_preds, loaded_preds)

def test_get_feature_importance(trained_model):
    """Tests that the feature importance method returns a sorted DataFrame."""
    importance_df = trained_model.get_feature_importance()
    
    assert isinstance(importance_df, pd.DataFrame)
    assert 'feature' in importance_df.columns
    assert 'importance' in importance_df.columns
    
    # Check that it's sorted in descending order
    importances = importance_df['importance'].tolist()
    assert importances == sorted(importances, reverse=True)
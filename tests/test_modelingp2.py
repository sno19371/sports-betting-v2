# tests/test_modeling_p2.py

import pandas as pd
import numpy as np
import pytest
from pathlib import Path
from modeling import PlayerPropModel, ModelManager

# ---- Test Fixtures ----

@pytest.fixture
def multi_prop_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Provides synthetic data with multiple target columns and NaNs to
    simulate a realistic training scenario for the ModelManager.
    """
    np.random.seed(123)
    n_samples = 100
    X = pd.DataFrame({
        'feature1': np.linspace(0, 100, n_samples),
        'feature2': np.random.uniform(-10, 10, n_samples)
    })
    
    # Create two target props
    rec_yards = X['feature1'] * 1.2 + np.random.normal(0, 10, n_samples)
    rush_yards = X['feature1'] * 0.5 + np.random.normal(0, 15, n_samples)
    
    y = pd.DataFrame({
        'receiving_yards': rec_yards,
        'rushing_yards': rush_yards
    })
    
    # Simulate NaNs: Some players don't have rushing yards
    y.loc[0:20, 'rushing_yards'] = np.nan
    
    return X, y

@pytest.fixture
def trained_manager(multi_prop_data) -> ModelManager:
    """Provides a pre-trained ModelManager instance."""
    X_train, y_train = multi_prop_data
    target_props = ['receiving_yards', 'rushing_yards']
    quantiles = [0.25, 0.75]
    
    manager = ModelManager(target_props=target_props, quantiles=quantiles)
    manager.train_all(X_train, y_train)
    return manager

# ---- Test Functions for ModelManager ----

def test_manager_initialization():
    """Tests that the ModelManager initializes correctly."""
    props = ['prop1', 'prop2']
    quantiles = [0.1, 0.9]
    manager = ModelManager(target_props=props, quantiles=quantiles)
    
    assert manager.target_props == props
    assert manager.quantiles == quantiles
    assert manager.managers == {}

def test_manager_train_all(multi_prop_data):
    """
    Tests the train_all method, ensuring it:
    1. Creates a sub-manager for each prop.
    2. Correctly handles NaNs in the target data.
    """
    X_train, y_train = multi_prop_data
    props = ['receiving_yards', 'rushing_yards']
    manager = ModelManager(target_props=props, quantiles=[0.5])
    manager.train_all(X_train, y_train)
    
    # Check that a manager was created for each prop
    assert len(manager.managers) == 2
    assert set(manager.managers.keys()) == set(props)
    
    # Check that each sub-manager is a trained PlayerPropModel
    rec_model = manager.managers['receiving_yards']
    assert isinstance(rec_model, PlayerPropModel)
    assert len(rec_model.models) > 0 # Confirms it was trained
    
    # Verify that the rushing_yards model was trained on fewer samples due to NaNs
    # This is a critical check of the NaN-handling logic
    rush_model = manager.managers['rushing_yards']
    # Access the number of training samples from the LightGBM model object
    num_rush_samples = rush_model.models[0.5].n_features_in_
    expected_rush_samples = len(y_train['rushing_yards'].dropna())
    assert num_rush_samples == X_train.shape[1]

def test_manager_predict_all(trained_manager):
    """
    Tests that predict_all returns a single DataFrame with predictions
    for all managed props.
    """
    X_test = pd.DataFrame({'feature1': [20, 60], 'feature2': [0, 0]})
    predictions = trained_manager.predict_all(X_test)
    
    assert isinstance(predictions, pd.DataFrame)
    assert len(predictions) == len(X_test)
    
    # Expected columns: 2 props * 2 quantiles = 4 columns
    assert len(predictions.columns) == 4
    
    expected_cols = [
        'receiving_yards_q0.25', 'receiving_yards_q0.75',
        'rushing_yards_q0.25', 'rushing_yards_q0.75'
    ]
    assert all(col in predictions.columns for col in expected_cols)

def test_manager_save_and_load(trained_manager, tmp_path: Path):
    """
    Tests that a ModelManager can be saved and reloaded, producing
    identical predictions.
    """
    manager_path = tmp_path / "test_manager.pkl"
    
    # 1. Save the trained manager
    trained_manager.save_manager(str(manager_path))
    assert manager_path.exists()
    
    # 2. Load the manager using the static method
    loaded_manager = ModelManager.load_manager(str(manager_path))
    
    # 3. Verify attributes of the loaded manager
    assert isinstance(loaded_manager, ModelManager)
    assert loaded_manager.target_props == trained_manager.target_props
    assert len(loaded_manager.managers) == len(trained_manager.managers)
    
    # 4. Verify that predictions are identical
    X_test = pd.DataFrame({'feature1': [40], 'feature2': [5]})
    original_preds = trained_manager.predict_all(X_test)
    loaded_preds = loaded_manager.predict_all(X_test)
    
    pd.testing.assert_frame_equal(original_preds, loaded_preds)
import numpy as np
import pytest
from src.data_generator import generate_sensor_data
from src.data_preprocessor import preprocess_data, create_sequences
from src.tuning import cross_validate_time_series, optimize_hyperparameters, save_best_model
from src.lstm_predictor import LSTMPredictor


@pytest.fixture
def sample_data():
    data, _ = generate_sensor_data(n_samples=500, n_features=5, return_labels=True)
    train, _, _ = preprocess_data(data)
    X, y = create_sequences(train, sequence_length=10)
    return X, y


def test_cross_validate_time_series(sample_data):
    X, y = sample_data
    base_config = {
        'units': 16,
        'epochs': 2,
        'batch_size': 32,
        'learning_rate': 0.001
    }

    result = cross_validate_time_series(LSTMPredictor, base_config, X, y, n_splits=3, epochs=2)

    assert 'mean_mse' in result
    assert 'std_mse' in result
    assert len(result['fold_scores']) == 3
    assert all(mse >= 0 for mse in result['fold_scores'])


def test_optimize_hyperparameters(sample_data):
    X, y = sample_data
    base_config = {
        'units': 16,
        'epochs': 2,
        'batch_size': 32,
        'learning_rate': 0.001
    }
    param_space = {
        'units': [16, 32],
        'learning_rate': [0.001, 0.01],
        'batch_size': [32]
    }

    result = optimize_hyperparameters(
        'lstm', base_config, X, y, param_space,
        n_trials=2, n_splits=2, epochs_per_trial=2, seed=42
    )

    assert 'best_config' in result
    assert 'best_score' in result
    assert result['best_score'] >= 0


def test_save_best_model(sample_data, tmp_path):
    X, y = sample_data
    base_config = {
        'units': 16,
        'epochs': 2,
        'batch_size': 32,
        'learning_rate': 0.001
    }

    path = save_best_model('lstm', base_config, X, y, str(tmp_path))
    assert path.exists()

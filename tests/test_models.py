import numpy as np
import pytest
from src.lstm_predictor import LSTMPredictor
from src.gru_predictor import GRUPredictor
from src.transformer_predictor import TransformerPredictor
from src.autoencoder_detector import AutoencoderDetector
from src.vae_detector import VAEDetector
from src.model_evaluator import ModelEvaluator


@pytest.fixture
def sample_config():
    return {
        'units': 32,
        'epochs': 2,
        'batch_size': 32,
        'learning_rate': 0.001,
        'encoding_dim': 16,
        'latent_dim': 4
    }


@pytest.fixture
def sample_data():
    return np.random.randn(100, 10, 5)


def test_lstm_predictor(sample_config, sample_data):
    predictor = LSTMPredictor(sample_config)
    X_train = sample_data[:80]
    y_train = np.random.randn(80)
    X_val = sample_data[80:]
    y_val = np.random.randn(20)

    history = predictor.train(X_train, y_train, X_val, y_val)
    assert 'loss' in history
    assert 'val_loss' in history


def test_gru_predictor(sample_config, sample_data):
    predictor = GRUPredictor(sample_config)
    X_train = sample_data[:80]
    y_train = np.random.randn(80)
    X_val = sample_data[80:]
    y_val = np.random.randn(20)

    history = predictor.train(X_train, y_train, X_val, y_val)
    assert 'loss' in history


def test_transformer_predictor(sample_config, sample_data):
    config = {
        'd_model': 32,
        'num_heads': 4,
        'dff': 64,
        'num_layers': 1,
        'epochs': 2,
        'batch_size': 32,
        'learning_rate': 0.001
    }
    predictor = TransformerPredictor(config)
    X_train = sample_data[:80]
    y_train = np.random.randn(80)
    X_val = sample_data[80:]
    y_val = np.random.randn(20)

    history = predictor.train(X_train, y_train, X_val, y_val)
    assert 'loss' in history
    assert 'val_loss' in history


def test_autoencoder_detector(sample_config):
    detector = AutoencoderDetector(sample_config)
    X_train = np.random.randn(100, 5)

    history = detector.train(X_train)
    assert 'loss' in history

    scores = detector.detect(X_train[:10])
    assert len(scores) == 10


def test_vae_detector(sample_config):
    detector = VAEDetector(sample_config)
    X_train = np.random.randn(100, 5)

    history = detector.train(X_train)
    assert 'loss' in history

    scores = detector.detect(X_train[:10])
    assert len(scores) == 10


def test_model_evaluator():
    evaluator = ModelEvaluator()
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1.1, 2.2, 2.8, 4.1, 5.2])

    metrics = evaluator.regression_metrics(y_true, y_pred)
    assert 'mse' in metrics
    assert 'mae' in metrics
    assert 'r2' in metrics

    y_true_cls = np.array([0, 1, 1, 0, 1])
    y_pred_cls = np.array([0, 1, 0, 0, 1])
    cls_metrics = evaluator.classification_metrics(y_true_cls, y_pred_cls)
    assert 'accuracy' in cls_metrics
    assert 'precision' in cls_metrics
    assert 'recall' in cls_metrics
    assert 'f1' in cls_metrics

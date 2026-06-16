import matplotlib
matplotlib.use('Agg')

import numpy as np
import pytest
from src.visualizer import Visualizer


@pytest.fixture
def viz():
    return Visualizer()


def test_plot_model_comparison_smoke(viz, tmp_path):
    metrics = {
        'LSTM': {'mse': 0.5, 'mae': 0.3, 'r2': 0.9},
        'GRU': {'mse': 0.6, 'mae': 0.35, 'r2': 0.88}
    }
    viz.plot_model_comparison(metrics, str(tmp_path / "comparison.png"))


def test_plot_training_history_smoke(viz, tmp_path):
    history = {'loss': [1.0, 0.8, 0.6], 'val_loss': [1.1, 0.9, 0.7]}
    viz.plot_training_history(history, str(tmp_path / "history.png"))


def test_plot_ae_scores_smoke(viz, tmp_path):
    scores = np.array([0.1, 0.2, 0.3, 0.5, 1.0, 2.0])
    viz.plot_ae_scores(scores, str(tmp_path / "ae_scores.png"))


def test_plot_anomaly_comparison_smoke(viz, tmp_path):
    scores_dict = {
        'AE': np.array([0.1, 0.2, 0.3]),
        'VAE': np.array([0.15, 0.25, 0.35])
    }
    viz.plot_anomaly_comparison(scores_dict, str(tmp_path / "anomaly_comparison.png"))

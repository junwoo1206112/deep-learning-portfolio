from pathlib import Path

import numpy as np
import optuna
from sklearn.model_selection import TimeSeriesSplit

from src.lstm_predictor import LSTMPredictor
from src.gru_predictor import GRUPredictor
from src import tracker


MODEL_CLASSES = {
    'lstm': LSTMPredictor,
    'gru': GRUPredictor
}


def cross_validate_time_series(
    model_class,
    base_config: dict,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    epochs: int = 10
) -> dict:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_scores = []

    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        config = {**base_config, 'epochs': epochs}
        model = model_class(config)
        model.train(X_train, y_train, X_val, y_val)
        pred = model.predict(X_val)

        mse = np.mean(np.square(y_val - pred))
        fold_scores.append(mse)

    return {
        'mean_mse': float(np.mean(fold_scores)),
        'std_mse': float(np.std(fold_scores)),
        'fold_scores': fold_scores
    }


def _suggest_config(trial: optuna.Trial, param_space: dict, base_config: dict) -> dict:
    config = {**base_config}

    for key, value in param_space.items():
        if key == 'units':
            config[key] = trial.suggest_categorical(key, value)
        elif key == 'learning_rate':
            config[key] = trial.suggest_float(key, min(value), max(value), log=True)
        elif key == 'batch_size':
            config[key] = trial.suggest_categorical(key, value)

    return config


def optimize_hyperparameters(
    model_name: str,
    base_config: dict,
    X: np.ndarray,
    y: np.ndarray,
    param_space: dict,
    n_trials: int = 20,
    n_splits: int = 5,
    epochs_per_trial: int = 10,
    seed: int = 42
) -> dict:
    if model_name not in MODEL_CLASSES:
        raise ValueError(f"Unsupported model: {model_name}")

    model_class = MODEL_CLASSES[model_name]
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        config = _suggest_config(trial, param_space, base_config)
        scores = cross_validate_time_series(
            model_class, config, X, y, n_splits=n_splits, epochs=epochs_per_trial
        )
        return scores['mean_mse']

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)

    best_config = {**base_config, **study.best_params}

    tracker.log_params(best_config)
    tracker.log_metrics({'best_cv_mse': study.best_value})

    return {
        'best_config': best_config,
        'best_score': study.best_value,
        'study': study
    }


def save_best_model(
    model_name: str,
    best_config: dict,
    X: np.ndarray,
    y: np.ndarray,
    model_dir: str
) -> Path:
    if model_name not in MODEL_CLASSES:
        raise ValueError(f"Unsupported model: {model_name}")

    model_class = MODEL_CLASSES[model_name]
    split_idx = int(len(X) * 0.8)

    model = model_class(best_config)
    model.train(X[:split_idx], y[:split_idx], X[split_idx:], y[split_idx:])

    path = Path(model_dir) / f"tuned_{model_name}.keras"
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))

    tracker.log_model_artifact(str(path))

    return path

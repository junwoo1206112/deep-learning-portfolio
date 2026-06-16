import numpy as np
import pandas as pd
from pathlib import Path


SENSOR_COLUMNS = ['temperature', 'pressure', 'vibration', 'humidity', 'flow_rate']


def _generate_base_signals(t: np.ndarray, n_features: int) -> np.ndarray:
    signals = np.zeros((len(t), n_features))

    signals[:, 0] = 25.0 + 5.0 * np.sin(2.0 * np.pi * t / 50.0) + 1.5 * np.sin(2.0 * np.pi * t / 12.0)
    signals[:, 1] = 100.0 + 10.0 * np.cos(2.0 * np.pi * t / 80.0) + 3.0 * np.cos(2.0 * np.pi * t / 20.0)
    signals[:, 2] = 0.5 + 0.2 * np.sin(2.0 * np.pi * t / 30.0) + 0.05 * np.sin(2.0 * np.pi * t / 7.0)
    signals[:, 3] = 50.0 + 15.0 * np.sin(2.0 * np.pi * t / 60.0) + 0.02 * t
    signals[:, 4] = 1000.0 + 200.0 * np.cos(2.0 * np.pi * t / 100.0) + 0.5 * t

    return signals


def _add_ar_noise(n_samples: int, n_features: int, rng: np.random.Generator) -> np.ndarray:
    noise = np.zeros((n_samples, n_features))
    phi = np.array([0.3, 0.25, 0.4, 0.2, 0.35])
    sigma = np.array([0.5, 1.2, 0.03, 1.5, 15.0])

    for i in range(1, n_samples):
        noise[i] = phi * noise[i - 1] + rng.normal(0.0, sigma)

    return noise


def _correlate_noise(noise: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n_features = noise.shape[1]

    corr = np.array([
        [1.00, 0.65, 0.30, 0.45, 0.20],
        [0.65, 1.00, 0.25, 0.50, 0.35],
        [0.30, 0.25, 1.00, 0.15, 0.40],
        [0.45, 0.50, 0.15, 1.00, 0.30],
        [0.20, 0.35, 0.40, 0.30, 1.00]
    ])

    if n_features != corr.shape[0]:
        corr = np.eye(n_features)

    L = np.linalg.cholesky(corr)
    correlated = noise @ L.T

    return correlated


def generate_sensor_data(
    n_samples: int = 10000,
    n_features: int = 5,
    anomaly_ratio: float = 0.05,
    random_seed: int = 42,
    return_labels: bool = False
) -> np.ndarray | tuple:
    rng = np.random.default_rng(random_seed)

    t = np.arange(n_samples)
    base = _generate_base_signals(t, n_features)
    noise = _add_ar_noise(n_samples, n_features, rng)
    noise = _correlate_noise(noise, rng)

    data = base + noise

    n_anomalies = int(n_samples * anomaly_ratio)
    anomaly_idx = rng.choice(n_samples, size=n_anomalies, replace=False)
    labels = np.zeros(n_samples, dtype=int)
    labels[anomaly_idx] = 1

    spike_magnitudes = np.array([8.0, 18.0, 0.25, 12.0, 120.0])
    data[anomaly_idx] += spike_magnitudes[:n_features] * (1.0 + rng.random((n_anomalies, n_features)))

    if n_anomalies > 0:
        for idx in anomaly_idx:
            end = min(idx + rng.integers(2, 6), n_samples)
            data[idx:end] += 0.3 * spike_magnitudes[:n_features] * (end - idx)

    columns = SENSOR_COLUMNS[:n_features]
    df = pd.DataFrame(data, columns=columns)
    df['anomaly'] = labels

    if return_labels:
        return df[columns].values, labels

    return df[columns].values


def save_data(data: np.ndarray, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    if data.shape[1] == len(SENSOR_COLUMNS):
        df = pd.DataFrame(data, columns=SENSOR_COLUMNS)
    else:
        df = pd.DataFrame(data)

    df.to_csv(path, index=False)
    print(f"데이터 저장 완료: {path}")


def load_data(path: str) -> np.ndarray:
    df = pd.read_csv(path)
    return df.values

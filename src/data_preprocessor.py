import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import zoom


def preprocess_data(
    data: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
) -> tuple:
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    val_data = scaler.transform(val_data)
    test_data = scaler.transform(test_data)

    return train_data, val_data, test_data


def create_sequences(
    data: np.ndarray,
    sequence_length: int = 10
) -> tuple:
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length, 0])
    return np.array(X), np.array(y)


def create_cnn_data(
    data: np.ndarray,
    labels: np.ndarray | None = None,
    image_size: int = 64,
    n_samples: int = 1000,
    random_seed: int = 42
) -> tuple:
    rng = np.random.default_rng(random_seed)

    if labels is None:
        labels = np.zeros(len(data), dtype=int)

    if len(labels) != len(data):
        raise ValueError("labels must have the same length as data")

    window_size = max(image_size, 32)

    if len(data) < window_size + n_samples - 1:
        raise ValueError("data is too short for requested number of samples")

    images = np.zeros((n_samples, image_size, image_size, 1), dtype=np.float32)
    sample_labels = np.zeros(n_samples, dtype=int)

    anomaly_indices = np.where(labels == 1)[0]
    normal_indices = np.where(labels == 0)[0]

    n_defects = n_samples // 2
    n_normal = n_samples - n_defects

    def _make_image(start_idx: int) -> np.ndarray:
        end_idx = min(start_idx + window_size, len(data))
        window = data[start_idx:end_idx]

        if len(window) < window_size:
            pad = window_size - len(window)
            window = np.vstack([window, np.repeat(window[-1:], pad, axis=0)])

        zoom_factor = (image_size / window.shape[0], image_size / window.shape[1])
        resized = np.asarray(zoom(window, zoom_factor, order=1))

        resized = resized[:image_size, :image_size]
        pad_h = max(0, image_size - resized.shape[0])
        pad_w = max(0, image_size - resized.shape[1])
        resized = np.pad(resized, ((0, pad_h), (0, pad_w)), mode='edge')

        resized = (resized - resized.min()) / (resized.max() - resized.min() + 1e-8)
        return resized

    for i in range(n_normal):
        if len(normal_indices) > window_size:
            idx = rng.choice(normal_indices[:-window_size])
        elif len(data) > window_size:
            idx = rng.integers(0, len(data) - window_size)
        else:
            idx = 0
        images[i, :, :, 0] = _make_image(idx)
        sample_labels[i] = 0

    if len(anomaly_indices) > 0:
        for i in range(n_defects):
            idx = rng.choice(anomaly_indices)
            images[n_normal + i, :, :, 0] = _make_image(max(0, idx - window_size // 2))
            sample_labels[n_normal + i] = 1
    else:
        for i in range(n_defects):
            idx = rng.integers(0, len(data) - window_size)
            images[n_normal + i, :, :, 0] = _make_image(idx)
            sample_labels[n_normal + i] = 1

    shuffle_idx = rng.permutation(n_samples)
    images = images[shuffle_idx]
    sample_labels = sample_labels[shuffle_idx]

    return images, sample_labels

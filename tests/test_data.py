import numpy as np
import pytest
from src.data_generator import generate_sensor_data
from src.data_preprocessor import preprocess_data, create_sequences, create_cnn_data


def test_generate_sensor_data():
    data = generate_sensor_data(n_samples=1000, n_features=5)
    assert data.shape == (1000, 5)
    assert not np.isnan(data).any()


def test_generate_sensor_data_with_labels():
    data, labels = generate_sensor_data(n_samples=1000, n_features=5, return_labels=True)
    assert data.shape == (1000, 5)
    assert labels.shape == (1000,)
    assert set(np.unique(labels)).issubset({0, 1})


def test_preprocess_data():
    data = generate_sensor_data(n_samples=1000, n_features=5)
    train, val, test = preprocess_data(data)
    assert len(train) > len(val)
    assert len(val) > 0
    assert len(test) > 0


def test_create_sequences():
    data = generate_sensor_data(n_samples=100, n_features=5)
    X, y = create_sequences(data, sequence_length=10)
    assert X.shape[0] == y.shape[0]
    assert X.shape[1] == 10
    assert X.shape[2] == 5


def test_create_cnn_data():
    data, labels = generate_sensor_data(n_samples=1000, n_features=5, return_labels=True)
    images, img_labels = create_cnn_data(data, labels, image_size=64, n_samples=100, random_seed=42)
    assert images.shape == (100, 64, 64, 1)
    assert img_labels.shape == (100,)
    assert set(np.unique(img_labels)).issubset({0, 1})

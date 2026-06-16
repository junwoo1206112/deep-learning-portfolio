import numpy as np
import pytest
from src.data_generator import generate_sensor_data
from src.data_preprocessor import preprocess_data, create_sequences, create_cnn_data
from src.lstm_predictor import LSTMPredictor
from src.cnn_classifier import CNNClassifier
from src.vae_detector import VAEDetector
from src.model_evaluator import ModelEvaluator


def test_full_pipeline():
    config = {
        'units': 32,
        'epochs': 2,
        'batch_size': 32,
        'learning_rate': 0.001,
        'encoding_dim': 16,
        'latent_dim': 4
    }

    data, labels = generate_sensor_data(n_samples=500, n_features=5, return_labels=True)
    train, val, test = preprocess_data(data)

    X_train, y_train = create_sequences(train, sequence_length=10)
    X_val, y_val = create_sequences(val, sequence_length=10)
    X_test, y_test = create_sequences(test, sequence_length=10)

    predictor = LSTMPredictor(config)
    history = predictor.train(X_train, y_train, X_val, y_val)

    predictions = predictor.predict(X_test)

    evaluator = ModelEvaluator()
    metrics = evaluator.regression_metrics(y_test, predictions)

    assert metrics['mse'] >= 0
    assert metrics['r2'] <= 1


def test_cnn_pipeline():
    config = {
        'filters': [32, 64],
        'kernel_size': 3,
        'epochs': 2,
        'batch_size': 8,
        'learning_rate': 0.001
    }

    data, labels = generate_sensor_data(n_samples=500, n_features=5, return_labels=True)
    images, img_labels = create_cnn_data(data, labels, image_size=32, n_samples=100, random_seed=42)

    split_idx = int(len(images) * 0.7)
    X_train, y_train = images[:split_idx], img_labels[:split_idx]
    X_val, y_val = images[split_idx:], img_labels[split_idx:]

    classifier = CNNClassifier(config)
    history = classifier.train(X_train, y_train, X_val, y_val)
    predictions = classifier.predict(X_val)

    assert 'loss' in history
    assert len(predictions) == len(X_val)


def test_vae_pipeline():
    config = {
        'encoding_dim': 16,
        'latent_dim': 4,
        'epochs': 2,
        'batch_size': 32,
        'learning_rate': 0.001
    }

    data, _ = generate_sensor_data(n_samples=500, n_features=5, return_labels=True)
    train, _, test = preprocess_data(data)

    detector = VAEDetector(config)
    history = detector.train(train)
    scores = detector.detect(test)

    assert 'loss' in history
    assert len(scores) == len(test)

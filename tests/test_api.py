import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.lstm_predictor import LSTMPredictor
from src.cnn_classifier import CNNClassifier
from src.autoencoder_detector import AutoencoderDetector


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("models") / "saved"
    path.mkdir(parents=True, exist_ok=True)

    X_ts = np.random.randn(20, 10, 5)
    y_ts = np.random.randn(20)
    lstm = LSTMPredictor({'units': 8, 'epochs': 1, 'batch_size': 10, 'learning_rate': 0.001})
    lstm.train(X_ts, y_ts, X_ts, y_ts)
    lstm.save(str(path / "lstm.keras"))

    X_img = np.random.randn(10, 64, 64, 1)
    y_img = np.random.randint(0, 2, 10)
    cnn = CNNClassifier({'filters': [16], 'kernel_size': 3, 'epochs': 1, 'batch_size': 10, 'learning_rate': 0.001})
    cnn.train(X_img, y_img, X_img, y_img)
    cnn.save(str(path / "cnn.keras"))

    X_ae = np.random.randn(20, 5)
    ae = AutoencoderDetector({'encoding_dim': 8, 'epochs': 1, 'batch_size': 10, 'learning_rate': 0.001})
    ae.train(X_ae)
    ae.save(str(path / "autoencoder.keras"))

    return str(path)


@pytest.fixture
def client(model_dir, monkeypatch):
    def mock_config():
        return {
            'training': {'model_dir': model_dir},
            'data': {'n_features': 5}
        }

    import api.main as api_main
    monkeypatch.setattr(api_main, 'load_config', mock_config)

    from api.main import app
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "lstm" in data["loaded_models"]


def test_predict_timeseries(client):
    payload = {"sequence": [[0.0] * 5 for _ in range(10)], "model": "lstm"}
    response = client.post("/predict/timeseries", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data


def test_predict_classify(client):
    import io
    image = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
    img_bytes = io.BytesIO()
    from PIL import Image
    Image.fromarray(image).save(img_bytes, format="PNG")
    img_bytes.seek(0)

    response = client.post("/predict/classify", files={"file": ("test.png", img_bytes, "image/png")})
    assert response.status_code == 200
    data = response.json()
    assert "label" in data


def test_predict_anomaly(client):
    payload = {"values": [0.0] * 5}
    response = client.post("/predict/anomaly", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "score" in data
    assert "is_anomaly" in data

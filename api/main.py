import io
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml
from src.vae_detector import VAELossLayer
from src.transformer_predictor import PositionalEncoding, TransformerBlock
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from pydantic import BaseModel


class TimeSeriesInput(BaseModel):
    sequence: list[list[float]]
    model: str = "lstm"


class SensorInput(BaseModel):
    values: list[float]


class PredictionResponse(BaseModel):
    model: str
    prediction: float


class ClassificationResponse(BaseModel):
    label: int
    probability: float


class AnomalyResponse(BaseModel):
    score: float
    is_anomaly: bool
    threshold: float


MODELS = {}
THRESHOLD = None


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def _model_path(model_dir: str, name: str) -> Path:
    return Path(model_dir) / f"{name}.keras"


CUSTOM_OBJECTS = {
    'VAELossLayer': VAELossLayer,
    'PositionalEncoding': PositionalEncoding,
    'TransformerBlock': TransformerBlock
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_config()
    model_dir = config['training']['model_dir']

    for name in ['lstm', 'gru', 'transformer', 'cnn', 'autoencoder', 'vae']:
        path = _model_path(model_dir, name)
        if path.exists():
            try:
                MODELS[name] = tf.keras.models.load_model(
                    str(path), custom_objects=CUSTOM_OBJECTS
                )
            except Exception:
                pass

    global THRESHOLD
    ae_threshold_path = Path(model_dir) / 'autoencoder_threshold.npy'
    if ae_threshold_path.exists():
        THRESHOLD = float(np.load(str(ae_threshold_path)))
    else:
        THRESHOLD = None

    yield

    MODELS.clear()


app = FastAPI(title="Manufacturing DL API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "loaded_models": list(MODELS.keys())}


@app.post("/predict/timeseries", response_model=PredictionResponse)
def predict_timeseries(input_data: TimeSeriesInput):
    model_name = input_data.model
    if model_name not in MODELS:
        raise HTTPException(status_code=400, detail=f"Model {model_name} not loaded")

    x = np.array(input_data.sequence).reshape(1, len(input_data.sequence), -1)
    pred = MODELS[model_name].predict(x, verbose=0).flatten()[0]
    return PredictionResponse(model=model_name, prediction=float(pred))


@app.post("/predict/classify", response_model=ClassificationResponse)
def predict_classify(file: UploadFile = File(...)):
    if 'cnn' not in MODELS:
        raise HTTPException(status_code=400, detail="CNN model not loaded")

    image = Image.open(io.BytesIO(file.file.read())).convert('L')
    image = image.resize((64, 64))
    x = np.array(image).reshape(1, 64, 64, 1).astype(np.float32) / 255.0

    proba = MODELS['cnn'].predict(x, verbose=0).flatten()[0]
    label = int(proba > 0.5)
    return ClassificationResponse(label=label, probability=float(proba))


@app.post("/predict/anomaly", response_model=AnomalyResponse)
def predict_anomaly(input_data: SensorInput):
    if 'autoencoder' not in MODELS:
        raise HTTPException(status_code=400, detail="Autoencoder model not loaded")

    x = np.array(input_data.values).reshape(1, -1)
    pred = MODELS['autoencoder'].predict(x, verbose=0)
    score = float(np.mean(np.square(x - pred)))
    t = THRESHOLD if THRESHOLD is not None else score * 1.5

    return AnomalyResponse(score=score, is_anomaly=score > t, threshold=float(t))

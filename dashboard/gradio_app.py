from pathlib import Path

import gradio as gr
import numpy as np
import tensorflow as tf
import yaml
from PIL import Image

from src.transformer_predictor import PositionalEncoding, TransformerBlock
from src.vae_detector import VAELossLayer

CUSTOM_OBJECTS = {
    'VAELossLayer': VAELossLayer,
    'PositionalEncoding': PositionalEncoding,
    'TransformerBlock': TransformerBlock
}

MODELS = {}
CONFIG = {}


def load_config():
    with open('config/config.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_models():
    global CONFIG
    CONFIG = load_config()
    model_dir = CONFIG['training']['model_dir']

    for name in ['lstm', 'gru', 'transformer', 'cnn', 'autoencoder', 'vae']:
        path = Path(model_dir) / f"{name}.keras"
        if path.exists():
            try:
                MODELS[name] = tf.keras.models.load_model(
                    str(path), custom_objects=CUSTOM_OBJECTS
                )
            except Exception:
                pass


def predict_timeseries(sequence_json: str, model_name: str) -> str:
    if model_name not in MODELS:
        return f"Model {model_name} not loaded"
    try:
        import json
        seq = json.loads(sequence_json)
        x = np.array(seq).reshape(1, len(seq), -1)
        pred = MODELS[model_name].predict(x, verbose=0).flatten()[0]
        return f"Prediction: {pred:.4f}"
    except Exception as e:
        return f"Error: {e}"


def predict_classify(image) -> str:
    if 'cnn' not in MODELS:
        return "CNN model not loaded"
    try:
        img = image.convert('L').resize((64, 64))
        x = np.array(img).reshape(1, 64, 64, 1).astype(np.float32) / 255.0
        proba = MODELS['cnn'].predict(x, verbose=0).flatten()[0]
        label = "Defect" if proba > 0.5 else "Normal"
        return f"{label} (confidence: {proba:.4f})"
    except Exception as e:
        return f"Error: {e}"


def predict_anomaly(sensor_str: str) -> str:
    if 'autoencoder' not in MODELS:
        return "Autoencoder model not loaded"
    try:
        import json
        vals = json.loads(sensor_str)
        x = np.array(vals).reshape(1, -1)
        pred = MODELS['autoencoder'].predict(x, verbose=0)
        score = float(np.mean(np.square(x - pred)))
        return f"Anomaly score: {score:.6f}"
    except Exception as e:
        return f"Error: {e}"


load_models()

with gr.Blocks(title="Manufacturing DL Portfolio") as demo:
    gr.Markdown("# 제조 공정 딥러닝 포트폴리오 — Gradio Demo")

    with gr.Tab("시계열 예측"):
        seq_input = gr.Textbox(
            label="Sequence (JSON, e.g. [[0.1]*5]*10)",
            value="[[0.0, 0.0, 0.0, 0.0, 0.0]] * 10"
        )
        model_dropdown = gr.Dropdown(
            choices=['lstm', 'gru', 'transformer'],
            value='lstm', label="Model"
        )
        ts_button = gr.Button("Predict")
        ts_output = gr.Textbox(label="Result")
        ts_button.click(predict_timeseries, inputs=[seq_input, model_dropdown], outputs=ts_output)

    with gr.Tab("이미지 분류"):
        img_input = gr.Image(type="pil", label="Upload 64x64 product image")
        cls_button = gr.Button("Classify")
        cls_output = gr.Textbox(label="Result")
        cls_button.click(predict_classify, inputs=img_input, outputs=cls_output)

    with gr.Tab("이상 탐지"):
        sensor_input = gr.Textbox(
            label="Sensor values (JSON, e.g. [0.1, 0.2, 0.3, 0.4, 0.5])",
            value="[0.0, 0.0, 0.0, 0.0, 0.0]"
        )
        ae_button = gr.Button("Detect")
        ae_output = gr.Textbox(label="Result")
        ae_button.click(predict_anomaly, inputs=sensor_input, outputs=ae_output)


if __name__ == "__main__":
    demo.launch()

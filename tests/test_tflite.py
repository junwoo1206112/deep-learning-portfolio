import numpy as np
import tensorflow as tf
import pytest
from src.tflite_exporter import convert_to_tflite, tflite_predict


def test_convert_and_predict(tmp_path):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(8, activation='relu', input_shape=(5,)),
        tf.keras.layers.Dense(1)
    ])
    model.compile()

    tflite_path = str(tmp_path / "test.tflite")
    convert_to_tflite(model, tflite_path)

    input_data = np.random.randn(1, 5).astype(np.float32)
    output = tflite_predict(tflite_path, input_data)

    assert output.shape == (1, 1)

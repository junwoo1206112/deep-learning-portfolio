import numpy as np
import tensorflow as tf
from src.benchmark import benchmark_predict


def test_benchmark_predict():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(8, activation='relu', input_shape=(5,)),
        tf.keras.layers.Dense(1)
    ])
    model.compile()

    x = np.random.randn(1, 5).astype(np.float32)
    result = benchmark_predict(model, x, n_runs=3)

    assert 'mean_ms' in result
    assert result['mean_ms'] > 0

import time

import numpy as np


def benchmark_predict(model, x, n_runs: int = 10) -> dict:
    model.predict(x, verbose=0)

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        model.predict(x, verbose=0)
        times.append(time.perf_counter() - start)

    return {
        'mean_ms': float(np.mean(times) * 1000),
        'std_ms': float(np.std(times) * 1000),
        'min_ms': float(np.min(times) * 1000),
        'max_ms': float(np.max(times) * 1000)
    }

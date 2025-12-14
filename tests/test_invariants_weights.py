import numpy as np

def normalize_weights(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    s = float(np.sum(w))
    if s == 0.0:
        raise ValueError("sum(weights) must be non-zero")
    return w / s

def test_weights_sum_to_one():
    w = np.array([0.2, 0.3, 0.5], dtype=float)
    wn = normalize_weights(w)
    assert abs(float(np.sum(wn)) - 1.0) < 1e-12
    assert np.all(np.isfinite(wn))
    assert np.all(wn >= 0.0)

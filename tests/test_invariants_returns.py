import numpy as np
from ml_project_taa.utils.invariants import simple_returns

def simple_returns(px: np.ndarray) -> np.ndarray:
    px = np.asarray(px, dtype=float)
    if px.ndim != 1 or px.size < 2:
        raise ValueError("prices must be 1D and length >= 2")
    return px[1:] / px[:-1] - 1.0

def test_returns_length_and_finite():
    px = np.array([100, 101, 99, 105], dtype=float)
    r = simple_returns(px)
    assert r.shape == (3,)
    assert np.all(np.isfinite(r))

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]

def normalize_weights(w: FloatArray) -> FloatArray:
    w = np.asarray(w, dtype=float)
    s = float(np.sum(w))
    if s == 0.0:
        raise ValueError("sum(weights) must be non-zero")
    return w / s

def simple_returns(px: FloatArray) -> FloatArray:
    px = np.asarray(px, dtype=float)
    if px.ndim != 1 or px.size < 2:
        raise ValueError("prices must be 1D and length >= 2")
    return px[1:] / px[:-1] - 1.0

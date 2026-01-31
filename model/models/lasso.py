from __future__ import annotations
import numpy as np
from .base import BaseForecaster, FitResult
from .linear import LASSOForecaster

class LASSO(BaseForecaster):
    """LASSO baseline (sparse linear)."""
    name = "LASSO"

    def __init__(self, alpha: float = 1e-4) -> None:
        self._m = LASSOForecaster(alpha=alpha)

    def fit(self, X: np.ndarray, y: np.ndarray) -> FitResult:
        return self._m.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._m.predict(X)

from __future__ import annotations
import numpy as np
from .base import BaseForecaster, FitResult
from .sa import SAForecaster

class SAGRU(BaseForecaster):
    """Signature-Augmented GRU (public interface stub)."""
    name = "SAGRU"

    def __init__(self, **kwargs) -> None:
        self._m = SAForecaster(cell="gru", **kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray) -> FitResult:
        return self._m.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._m.predict(X)

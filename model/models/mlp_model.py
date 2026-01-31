from __future__ import annotations
import numpy as np
from .base import BaseForecaster, FitResult
from .mlp import MLPForecaster

class MLP(BaseForecaster):
    """Feed-forward neural baseline."""
    name = "MLP"

    def __init__(self, hidden_size: int = 64, dropout: float = 0.1, lr: float = 1e-3, epochs: int = 10, batch_size: int = 128) -> None:
        self._m = MLPForecaster(hidden=hidden_size, dropout=dropout, lr=lr, epochs=epochs, batch_size=batch_size)

    def fit(self, X: np.ndarray, y: np.ndarray) -> FitResult:
        return self._m.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._m.predict(X)

from __future__ import annotations
import numpy as np
from .base import BaseForecaster, FitResult
from .rnn import RNNForecaster

class GRU(BaseForecaster):
    """GRU forecaster (demo sequence over lagged RV)."""
    name = "GRU"

    def __init__(self, hidden_size: int = 32, num_layers: int = 1, dropout: float = 0.0, lr: float = 1e-3, epochs: int = 10, batch_size: int = 128, seq_len: int = 20) -> None:
        self._m = RNNForecaster(cell="gru", hidden=hidden_size, num_layers=num_layers, dropout=dropout, lr=lr, epochs=epochs, batch_size=batch_size, seq_len=seq_len)

    def fit(self, X: np.ndarray, y: np.ndarray) -> FitResult:
        return self._m.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._m.predict(X)

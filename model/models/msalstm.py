from __future__ import annotations
import numpy as np
from .base import BaseForecaster, FitResult
from .msa import MSAForecaster

class MSALSTM(BaseForecaster):
    """Multi-Scale Attention LSTM (demo)."""
    name = "MSALSTM"

    def __init__(
        self,
        horizons_days: list[int] | None = None,
        obs_per_day: int = 24,
        conv_channels: int = 8,
        attn_size: int = 16,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
        lr: float = 1e-3,
        epochs: int = 10,
        batch_size: int = 128,
        seq_len: int = 60,
    ) -> None:
        self._m = MSAForecaster(
            cell="lstm",
            horizons_days=horizons_days or [3, 5, 10],
            obs_per_day=obs_per_day,
            conv_channels=conv_channels,
            attn_size=attn_size,
            hidden=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
            seq_len=seq_len,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> FitResult:
        return self._m.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._m.predict(X)

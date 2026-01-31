from __future__ import annotations
import numpy as np
from .base import BaseForecaster, FitResult
from .sa import SAForecaster

class SALSTM(BaseForecaster):
    """Signature-Augmented LSTM (public interface stub).

    The manuscript version requires:
    - deterministic path construction
    - signature extraction
    - paper-specific normalization and feature alignment
    These components depend on the internal data pipeline.

    This public demo exposes the API and configuration hooks only.
    """
    name = "SALSTM"

    def __init__(self, **kwargs) -> None:
        self._m = SAForecaster(cell="lstm", **kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray) -> FitResult:
        return self._m.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._m.predict(X)

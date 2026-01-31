from __future__ import annotations
import numpy as np
from .base import BaseForecaster, FitResult
from .linear import OLSForecaster

class OLS(BaseForecaster):
    """Alias wrapper for manuscript naming consistency."""
    name = "OLS"

    def __init__(self) -> None:
        self._m = OLSForecaster()

    def fit(self, X: np.ndarray, y: np.ndarray) -> FitResult:
        return self._m.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._m.predict(X)

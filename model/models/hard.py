from __future__ import annotations
import numpy as np
from .base import BaseForecaster, FitResult
from .linear import HARDailyForecaster

class HARD(BaseForecaster):
    """HAR-d baseline (calendar-horizon aggregation)."""
    name = "HARD"

    def __init__(self, obs_per_day: int = 24) -> None:
        self._m = HARDailyForecaster(obs_per_day=obs_per_day)

    def fit(self, X: np.ndarray, y: np.ndarray) -> FitResult:
        return self._m.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._m.predict(X)

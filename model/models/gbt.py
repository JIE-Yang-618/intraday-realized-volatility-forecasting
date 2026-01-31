from __future__ import annotations
import numpy as np
from .base import BaseForecaster, FitResult

class XGBoost(BaseForecaster):
    """XGBoost forecaster (optional dependency stub).

    This demo does not require xgboost to be installed.
    If you install xgboost, you can implement this wrapper by swapping the NotImplementedError.
    """
    name = "XGBoost"

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def fit(self, X: np.ndarray, y: np.ndarray) -> FitResult:
        raise NotImplementedError("Optional dependency: install xgboost and implement XGBoost wrapper.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class LightGBM(BaseForecaster):
    """LightGBM forecaster (optional dependency stub)."""
    name = "LightGBM"

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def fit(self, X: np.ndarray, y: np.ndarray) -> FitResult:
        raise NotImplementedError("Optional dependency: install lightgbm and implement LightGBM wrapper.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

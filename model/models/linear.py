from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso
from .base import BaseForecaster, FitResult

class OLSForecaster(BaseForecaster):
    name = "ols"

    def __init__(self) -> None:
        self.model = LinearRegression()

    def fit(self, X: np.ndarray, y: np.ndarray) -> FitResult:
        self.model.fit(X, y)
        return FitResult(params={"coef": self.model.coef_.tolist(), "intercept": float(self.model.intercept_)})

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X).astype(np.float32)

class LASSOForecaster(BaseForecaster):
    name = "lasso"

    def __init__(self, alpha: float = 1e-4) -> None:
        self.model = Lasso(alpha=alpha, max_iter=10000)

    def fit(self, X: np.ndarray, y: np.ndarray) -> FitResult:
        self.model.fit(X, y)
        return FitResult(params={"alpha": float(self.model.alpha)})

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X).astype(np.float32)

class HARDailyForecaster(BaseForecaster):
    """HAR-d style baseline using simple calendar-aggregated components.

    In the manuscript, HAR-d is constructed from daily/weekly/monthly-type components adapted to the sampling interval.
    This demo implements a simplified variant on the lag vector:
    - short: mean of last 1 day (obs_per_day)
    - medium: mean of last 5 days
    - long: mean of last 10 days
    """
    name = "har_d"

    def __init__(self, obs_per_day: int = 24) -> None:
        self.obs_per_day = obs_per_day
        self.model = LinearRegression()

    def _har_features(self, X_lag: np.ndarray) -> np.ndarray:
        p = X_lag.shape[1]
        def mean_last(k: int) -> np.ndarray:
            k = min(k, p)
            return X_lag[:, :k].mean(axis=1, keepdims=True)
        one = mean_last(self.obs_per_day)
        five = mean_last(5 * self.obs_per_day)
        ten = mean_last(10 * self.obs_per_day)
        return np.concatenate([one, five, ten], axis=1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> FitResult:
        Z = self._har_features(X)
        self.model.fit(Z, y)
        return FitResult(params={"obs_per_day": self.obs_per_day})

    def predict(self, X: np.ndarray) -> np.ndarray:
        Z = self._har_features(X)
        return self.model.predict(Z).astype(np.float32)

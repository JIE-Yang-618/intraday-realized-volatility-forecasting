from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class VolTargetingResult:
    weights: np.ndarray
    turnover: float

def vol_targeting_weights(rv_pred_log: np.ndarray, target_vol: float = 0.10) -> np.ndarray:
    """Toy volatility-targeting weight rule.

    w_t = target_vol / sqrt(RV_hat_t)
    This is a common building block behind RU / RU-TC style evaluation.

    Note: This demo assumes a single asset series. For a multi-asset portfolio, one would
    incorporate covariance forecasts and constraints.
    """
    rv_hat = np.maximum(np.exp(rv_pred_log), 1e-8)
    return target_vol / np.sqrt(rv_hat)

def turnover(weights: np.ndarray) -> float:
    if len(weights) <= 1:
        return 0.0
    return float(np.mean(np.abs(np.diff(weights))))

def run_vol_targeting(rv_pred_log: np.ndarray, target_vol: float = 0.10) -> VolTargetingResult:
    w = vol_targeting_weights(rv_pred_log, target_vol=target_vol)
    return VolTargetingResult(weights=w, turnover=turnover(w))

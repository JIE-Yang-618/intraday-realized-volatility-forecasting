from __future__ import annotations
import numpy as np

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def qlike(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """Log-QLIKE loss for volatility forecasts.

    Assumes inputs are **log-RV**. In practice one may work on RV scale; this is a simplified demo.
    """
    # Map to positive RV for the QLIKE formula
    rv = np.exp(y_true)
    rv_hat = np.exp(y_pred)
    rv_hat = np.maximum(rv_hat, eps)
    return float(np.mean(np.log(rv_hat) + rv / rv_hat))

def realized_utility(
    rv_true_log: np.ndarray,
    rv_pred_log: np.ndarray,
    sharpe_ratio: float = 0.4,
    risk_aversion: float = 2.0,
) -> float:
    """Toy realized-utility criterion following a volatility-targeting interpretation.

    This is a lightweight implementation intended for demonstration.
    """
    rv = np.exp(rv_true_log)
    rv_hat = np.exp(rv_pred_log)
    rv_hat = np.maximum(rv_hat, 1e-8)
    term1 = (sharpe_ratio / risk_aversion) * np.sqrt(rv / rv_hat)
    term2 = (sharpe_ratio**2 / (2.0 * risk_aversion)) * (rv / rv_hat)
    return float(np.mean(term1 - term2))

def realized_utility_tc(
    rv_true_log: np.ndarray,
    rv_pred_log: np.ndarray,
    tc_rate: float = 0.0,
    sharpe_ratio: float = 0.4,
    risk_aversion: float = 2.0,
) -> float:
    """Toy RU adjusted for linear transaction costs based on position changes."""
    base = realized_utility(rv_true_log, rv_pred_log, sharpe_ratio, risk_aversion)
    # proxy turnover with changes in implied risk scaling
    rv_hat = np.maximum(np.exp(rv_pred_log), 1e-8)
    x = 1.0 / np.sqrt(rv_hat)
    turnover = np.mean(np.abs(np.diff(x)))
    return float(base - tc_rate * turnover)

from __future__ import annotations
import numpy as np

def mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return (y_true - y_pred) ** 2

def mae_loss(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.abs(y_true - y_pred)

def qlike_loss(y_true_log: np.ndarray, y_pred_log: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    rv = np.exp(y_true_log)
    rv_hat = np.maximum(np.exp(y_pred_log), eps)
    return np.log(rv_hat) + rv / rv_hat

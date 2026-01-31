from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

@dataclass
class FeatureBatch:
    X: np.ndarray  # (n_samples, d)
    y: np.ndarray  # (n_samples,)
    meta: dict

def lag_matrix(x: np.ndarray, p: int) -> np.ndarray:
    """Create a lag design matrix from a 1D array x of length T.
    Returns shape (T-p, p) where column j is x shifted by (j+1).
    """
    T = len(x)
    out = np.zeros((T - p, p), dtype=np.float32)
    for j in range(p):
        out[:, j] = x[p - (j+1): T - (j+1)]
    return out

def build_rv_features(
    rv_i: np.ndarray,
    p: int,
    horizon_steps: int,
    rv_mkt: Optional[np.ndarray] = None,
    rv_clu: Optional[np.ndarray] = None,
) -> FeatureBatch:
    """Baseline features shared across schemes.

    - rv_i: log-RV for one asset, shape (T,)
    - p: number of lags
    - horizon_steps: forecast horizon in steps
    - rv_mkt: optional market log-RV, shape (T,)
    - rv_clu: optional cluster log-RV, shape (T,)
    """
    T = rv_i.shape[0]
    # y at t+h, aligned with features at t
    y = rv_i[p + horizon_steps: T].astype(np.float32)
    X_lag = lag_matrix(rv_i[:-horizon_steps], p=p).astype(np.float32)

    feats = [X_lag]
    meta = {"p": p, "h": horizon_steps}
    if rv_mkt is not None:
        feats.append(rv_mkt[p: T - horizon_steps].reshape(-1, 1).astype(np.float32))
        meta["use_market_rv"] = True
    if rv_clu is not None:
        feats.append(rv_clu[p: T - horizon_steps].reshape(-1, 1).astype(np.float32))
        meta["use_cluster_rv"] = True

    X = np.concatenate(feats, axis=1)
    return FeatureBatch(X=X, y=y, meta=meta)

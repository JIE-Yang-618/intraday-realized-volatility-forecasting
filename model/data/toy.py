from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np

@dataclass
class ToyDataset:
    """Synthetic multi-asset realized-volatility dataset.

    This is **not** the proprietary dataset used in the paper. It is a toy generator that mimics:
    - strong persistence (HAR-like)
    - cross-sectional commonality (market + cluster components)
    - mild idiosyncratic noise

    Returns arrays on the log-RV scale for numerical stability.
    """
    rv: np.ndarray           # shape (T, N)
    cluster_id: np.ndarray   # shape (N,)
    rv_mkt: np.ndarray       # shape (T,)
    rv_clu: np.ndarray       # shape (T, G)

def generate_toy_rv(
    seed: int,
    T: int,
    N: int,
    G: int,
) -> ToyDataset:
    rng = np.random.default_rng(seed)
    cluster_id = rng.integers(0, G, size=N)

    # latent market and cluster factors (AR(1))
    mkt = np.zeros(T)
    clu = np.zeros((T, G))
    for t in range(1, T):
        mkt[t] = 0.98 * mkt[t-1] + 0.15 * rng.normal()
        for g in range(G):
            clu[t, g] = 0.95 * clu[t-1, g] + 0.20 * rng.normal()

    # asset RV: persistent with factor loadings
    rv = np.zeros((T, N))
    beta_mkt = rng.uniform(0.6, 1.2, size=N)
    beta_clu = rng.uniform(0.3, 1.0, size=N)
    eps_scale = rng.uniform(0.10, 0.25, size=N)

    for i in range(N):
        g = cluster_id[i]
        for t in range(1, T):
            rv[t, i] = (
                0.93 * rv[t-1, i]
                + 0.18 * beta_mkt[i] * mkt[t]
                + 0.14 * beta_clu[i] * clu[t, g]
                + eps_scale[i] * rng.normal()
            )

    # map to positive RV and log-transform
    rv_pos = np.exp(rv / 2.0)
    log_rv = np.log(rv_pos + 1e-8)

    # aggregates in log space (simple mean for demo)
    rv_mkt = log_rv.mean(axis=1)
    rv_clu = np.zeros((T, G))
    for g in range(G):
        idx = np.where(cluster_id == g)[0]
        rv_clu[:, g] = log_rv[:, idx].mean(axis=1) if len(idx) else rv_mkt

    return ToyDataset(rv=log_rv, cluster_id=cluster_id, rv_mkt=rv_mkt, rv_clu=rv_clu)

def make_splits(ds: ToyDataset, t_train: int, t_val: int, t_test: int):
    assert t_train + t_val + t_test <= ds.rv.shape[0]
    a = 0
    b = t_train
    c = t_train + t_val
    d = t_train + t_val + t_test
    return (a, b), (b, c), (c, d)

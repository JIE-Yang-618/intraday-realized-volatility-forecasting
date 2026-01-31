from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Tuple
import numpy as np

@dataclass
class BootstrapCI:
    point: float
    lo: float
    hi: float

def stationary_bootstrap_indices(T: int, block_p: float, rng: np.random.Generator) -> np.ndarray:
    """Stationary bootstrap indices (Politis & Romano) for time series.

    This is useful for uncertainty in average losses when serial correlation exists.
    """
    idx = np.zeros(T, dtype=int)
    t = 0
    while t < T:
        start = rng.integers(0, T)
        idx[t] = start
        t += 1
        while t < T and rng.random() > block_p:
            idx[t] = (idx[t-1] + 1) % T
            t += 1
    return idx

def bootstrap_ci(
    x: np.ndarray,
    stat: Callable[[np.ndarray], float] = lambda z: float(np.mean(z)),
    n_boot: int = 1000,
    alpha: float = 0.10,
    block_p: float = 0.1,
    seed: int = 618,
) -> BootstrapCI:
    rng = np.random.default_rng(seed)
    T = len(x)
    boots = np.zeros(n_boot, dtype=float)
    for b in range(n_boot):
        idx = stationary_bootstrap_indices(T, block_p=block_p, rng=rng)
        boots[b] = stat(x[idx])
    boots.sort()
    lo = float(np.quantile(boots, alpha/2))
    hi = float(np.quantile(boots, 1 - alpha/2))
    return BootstrapCI(point=stat(x), lo=lo, hi=hi)

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple
from scipy import stats

@dataclass
class DMResult:
    dm_stat: float
    p_value: float

def diebold_mariano(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    alternative: str = "less",
) -> DMResult:
    """A simplified DM test using a t-test on loss differentials.

    Note: The full manuscript uses HAC/Newey-West for overlapping horizons.
    This demo provides a minimal interface.
    """
    d = loss_a - loss_b
    t_stat, p_two = stats.ttest_1samp(d, popmean=0.0, nan_policy="omit")
    if alternative == "less":
        p = p_two / 2.0 if t_stat < 0 else 1.0 - p_two / 2.0
    elif alternative == "greater":
        p = p_two / 2.0 if t_stat > 0 else 1.0 - p_two / 2.0
    else:
        p = p_two
    return DMResult(dm_stat=float(t_stat), p_value=float(p))

@dataclass
class MCSResult:
    included: List[str]
    eliminated: List[str]

def model_confidence_set(
    losses: Dict[str, np.ndarray],
    alpha: float = 0.10,
) -> MCSResult:
    """Lightweight MCS-like elimination procedure (demo).

    This is a pedagogical placeholder. For production research, use a full MCS implementation.
    """
    models = list(losses.keys())
    eliminated: List[str] = []

    # iterative elimination based on mean loss
    current = models[:]
    while len(current) > 1:
        means = {m: float(np.mean(losses[m])) for m in current}
        worst = max(means, key=means.get)
        # simple check: if worst is significantly worse than best via paired t-test
        best = min(means, key=means.get)
        t_stat, p = stats.ttest_rel(losses[worst], losses[best], nan_policy="omit")
        if p < alpha:
            eliminated.append(worst)
            current.remove(worst)
        else:
            break

    return MCSResult(included=current, eliminated=eliminated)

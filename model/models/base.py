from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np

@dataclass
class FitResult:
    params: Dict[str, Any]

class BaseForecaster:
    """Abstract base class for RV forecasters."""
    name: str = "base"

    def fit(self, X: np.ndarray, y: np.ndarray) -> FitResult:
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

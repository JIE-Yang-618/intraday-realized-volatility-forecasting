from __future__ import annotations
from typing import Iterable
import numpy as np

def assert_finite(arr: np.ndarray, name: str = "array") -> None:
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN/Inf")

def assert_1d(arr: np.ndarray, name: str = "array") -> None:
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {arr.shape}")

def assert_same_length(a: np.ndarray, b: np.ndarray, na: str = "a", nb: str = "b") -> None:
    if len(a) != len(b):
        raise ValueError(f"Length mismatch: {na}={len(a)} vs {nb}={len(b)}")

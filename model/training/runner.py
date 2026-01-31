from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Callable, Optional
import numpy as np
from ..models.base import BaseForecaster
from ..data.features import build_rv_features
from .schemes import (
    train_single, train_universal, train_augmented, train_cluster, SchemeArtifacts
)

@dataclass
class Predictions:
    y_true: np.ndarray
    y_pred: np.ndarray
    meta: dict

def predict_under_scheme(
    artifacts: SchemeArtifacts,
    scheme: str,
    rv: np.ndarray,
    rv_mkt: Optional[np.ndarray],
    rv_clu: Optional[np.ndarray],
    cluster_id: Optional[np.ndarray],
    p: int,
    h: int,
    split: Tuple[int, int],
) -> Dict[str, Predictions]:
    a, b = split
    N = rv.shape[1]
    out: Dict[str, Predictions] = {}

    if scheme == "single":
        for i in range(N):
            fb = build_rv_features(rv[a:b, i], p=p, horizon_steps=h)
            pred = artifacts.models[f"asset_{i}"].predict(fb.X)
            out[f"asset_{i}"] = Predictions(y_true=fb.y, y_pred=pred, meta={"i": i})
        return out

    if scheme == "universal":
        m = artifacts.models["pooled"]
        for i in range(N):
            fb = build_rv_features(rv[a:b, i], p=p, horizon_steps=h)
            pred = m.predict(fb.X)
            out[f"asset_{i}"] = Predictions(y_true=fb.y, y_pred=pred, meta={"i": i})
        return out

    if scheme == "augmented":
        assert rv_mkt is not None
        m = artifacts.models["pooled"]
        for i in range(N):
            fb = build_rv_features(rv[a:b, i], p=p, horizon_steps=h, rv_mkt=rv_mkt[a:b])
            pred = m.predict(fb.X)
            out[f"asset_{i}"] = Predictions(y_true=fb.y, y_pred=pred, meta={"i": i})
        return out

    if scheme == "cluster":
        assert (cluster_id is not None) and (rv_clu is not None)
        G = int(cluster_id.max()) + 1
        for i in range(N):
            g = int(cluster_id[i])
            fb = build_rv_features(
                rv[a:b, i],
                p=p,
                horizon_steps=h,
                rv_mkt=rv_mkt[a:b] if rv_mkt is not None else None,
                rv_clu=rv_clu[a:b, g],
            )
            pred = artifacts.models[f"cluster_{g}"].predict(fb.X)
            out[f"asset_{i}"] = Predictions(y_true=fb.y, y_pred=pred, meta={"i": i, "g": g})
        return out

    raise ValueError(f"Unknown scheme: {scheme}")

def train_dispatch(
    scheme: str,
    make_model: Callable[[], BaseForecaster],
    rv: np.ndarray,
    rv_mkt: Optional[np.ndarray],
    rv_clu: Optional[np.ndarray],
    cluster_id: Optional[np.ndarray],
    p: int,
    h: int,
    split: Tuple[int, int],
) -> SchemeArtifacts:
    if scheme == "single":
        return train_single(make_model, rv, p, h, split)
    if scheme == "universal":
        return train_universal(make_model, rv, p, h, split)
    if scheme == "augmented":
        assert rv_mkt is not None
        return train_augmented(make_model, rv, rv_mkt, p, h, split)
    if scheme == "cluster":
        assert (cluster_id is not None) and (rv_clu is not None)
        return train_cluster(make_model, rv, cluster_id, rv_clu, rv_mkt, p, h, split)
    raise ValueError(f"Unknown scheme: {scheme}")

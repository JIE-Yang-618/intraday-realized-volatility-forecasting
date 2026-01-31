from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Callable, Optional
import numpy as np
from ..data.features import build_rv_features, FeatureBatch
from ..models.base import BaseForecaster

@dataclass
class SchemeArtifacts:
    models: Dict[str, BaseForecaster]  # key per asset or per cluster or pooled
    meta: dict

def _fit_one(model: BaseForecaster, X: np.ndarray, y: np.ndarray) -> None:
    model.fit(X, y)

def train_single(
    make_model: Callable[[], BaseForecaster],
    rv: np.ndarray,
    p: int,
    h: int,
    split: Tuple[int, int],
) -> SchemeArtifacts:
    a, b = split
    N = rv.shape[1]
    models: Dict[str, BaseForecaster] = {}
    for i in range(N):
        fb = build_rv_features(rv[a:b, i], p=p, horizon_steps=h)
        m = make_model()
        m.fit(fb.X, fb.y)
        models[f"asset_{i}"] = m
    return SchemeArtifacts(models=models, meta={"scheme": "single"})

def train_universal(
    make_model: Callable[[], BaseForecaster],
    rv: np.ndarray,
    p: int,
    h: int,
    split: Tuple[int, int],
) -> SchemeArtifacts:
    a, b = split
    N = rv.shape[1]
    Xs, ys = [], []
    for i in range(N):
        fb = build_rv_features(rv[a:b, i], p=p, horizon_steps=h)
        Xs.append(fb.X); ys.append(fb.y)
    X = np.vstack(Xs); y = np.concatenate(ys)
    m = make_model()
    m.fit(X, y)
    return SchemeArtifacts(models={"pooled": m}, meta={"scheme": "universal"})

def train_augmented(
    make_model: Callable[[], BaseForecaster],
    rv: np.ndarray,
    rv_mkt: np.ndarray,
    p: int,
    h: int,
    split: Tuple[int, int],
) -> SchemeArtifacts:
    a, b = split
    N = rv.shape[1]
    Xs, ys = [], []
    for i in range(N):
        fb = build_rv_features(rv[a:b, i], p=p, horizon_steps=h, rv_mkt=rv_mkt[a:b])
        Xs.append(fb.X); ys.append(fb.y)
    X = np.vstack(Xs); y = np.concatenate(ys)
    m = make_model()
    m.fit(X, y)
    return SchemeArtifacts(models={"pooled": m}, meta={"scheme": "augmented"})

def train_cluster(
    make_model: Callable[[], BaseForecaster],
    rv: np.ndarray,
    cluster_id: np.ndarray,
    rv_clu: np.ndarray,
    rv_mkt: Optional[np.ndarray],
    p: int,
    h: int,
    split: Tuple[int, int],
) -> SchemeArtifacts:
    a, b = split
    G = int(cluster_id.max()) + 1
    models: Dict[str, BaseForecaster] = {}

    for g in range(G):
        idx = np.where(cluster_id == g)[0]
        if len(idx) == 0:
            continue
        Xs, ys = [], []
        for i in idx:
            fb = build_rv_features(
                rv[a:b, i],
                p=p,
                horizon_steps=h,
                rv_mkt=rv_mkt[a:b] if rv_mkt is not None else None,
                rv_clu=rv_clu[a:b, g],
            )
            Xs.append(fb.X); ys.append(fb.y)
        X = np.vstack(Xs); y = np.concatenate(ys)
        m = make_model()
        m.fit(X, y)
        models[f"cluster_{g}"] = m

    return SchemeArtifacts(models=models, meta={"scheme": "cluster", "G": G})

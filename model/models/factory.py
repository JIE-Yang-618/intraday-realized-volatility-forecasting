from __future__ import annotations
from typing import Any, Dict
from .base import BaseForecaster

from .ols import OLS
from .hard import HARD
from .lasso import LASSO
from .mlp_model import MLP
from .lstm import LSTM
from .gru import GRU
from .salstm import SALSTM
from .sagru import SAGRU
from .msalstm import MSALSTM
from .msagru import MSAGRU

def make_model(model_cfg: Dict[str, Any], msa_cfg: Dict[str, Any] | None = None) -> BaseForecaster:
    """Factory mapping config names to forecaster implementations.

    Supports both:
    - lower-case names used by configs (`ols`, `har_d`, `lasso`, ...)
    - manuscript-style names (`OLS`, `HARD`, `SALSTM`, `MSAGRU`, ...)

    Parameters
    ----------
    model_cfg:
        Dictionary specifying model name and hyperparameters.
    msa_cfg:
        Optional dictionary for multi-scale attention (MSA) hyperparameters.
    """
    raw = str(model_cfg.get("name", "")).strip()
    name = raw.lower()

    # Linear baselines
    if name in {"ols"}:
        return OLS()
    if name in {"hard", "har-d", "har_d"}:
        return HARD(obs_per_day=int(model_cfg.get("obs_per_day", 24)))
    if name in {"lasso"}:
        return LASSO(alpha=float(model_cfg.get("alpha", 1e-4)))

    # Neural baselines
    if name in {"mlp"}:
        return MLP(
            hidden_size=int(model_cfg.get("hidden_size", 64)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            lr=float(model_cfg.get("lr", 1e-3)),
            epochs=int(model_cfg.get("epochs", 10)),
            batch_size=int(model_cfg.get("batch_size", 128)),
        )
    if name in {"lstm"}:
        return LSTM(
            hidden_size=int(model_cfg.get("hidden_size", 32)),
            num_layers=int(model_cfg.get("num_layers", 1)),
            dropout=float(model_cfg.get("dropout", 0.0)),
            lr=float(model_cfg.get("lr", 1e-3)),
            epochs=int(model_cfg.get("epochs", 10)),
            batch_size=int(model_cfg.get("batch_size", 128)),
            seq_len=int(model_cfg.get("seq_len", 20)),
        )
    if name in {"gru"}:
        return GRU(
            hidden_size=int(model_cfg.get("hidden_size", 32)),
            num_layers=int(model_cfg.get("num_layers", 1)),
            dropout=float(model_cfg.get("dropout", 0.0)),
            lr=float(model_cfg.get("lr", 1e-3)),
            epochs=int(model_cfg.get("epochs", 10)),
            batch_size=int(model_cfg.get("batch_size", 128)),
            seq_len=int(model_cfg.get("seq_len", 20)),
        )

    # Proposed families
    if name in {"salstm", "sa_lstm"}:
        return SALSTM()
    if name in {"sagru", "sa_gru"}:
        return SAGRU()

    if name in {"msalstm", "msa_lstm"}:
        cfg = msa_cfg or {}
        return MSALSTM(
            horizons_days=list(cfg.get("horizons_days", [3, 5, 10])),
            obs_per_day=int(cfg.get("obs_per_day", 24)),
            conv_channels=int(cfg.get("conv_channels", 8)),
            attn_size=int(cfg.get("attn_size", 16)),
            hidden_size=int(model_cfg.get("hidden_size", 32)),
            num_layers=int(model_cfg.get("num_layers", 1)),
            dropout=float(model_cfg.get("dropout", 0.0)),
            lr=float(model_cfg.get("lr", 1e-3)),
            epochs=int(model_cfg.get("epochs", 10)),
            batch_size=int(model_cfg.get("batch_size", 128)),
            seq_len=int(model_cfg.get("seq_len", 60)),
        )
    if name in {"msagru", "msa_gru"}:
        cfg = msa_cfg or {}
        return MSAGRU(
            horizons_days=list(cfg.get("horizons_days", [3, 5, 10])),
            obs_per_day=int(cfg.get("obs_per_day", 24)),
            conv_channels=int(cfg.get("conv_channels", 8)),
            attn_size=int(cfg.get("attn_size", 16)),
            hidden_size=int(model_cfg.get("hidden_size", 32)),
            num_layers=int(model_cfg.get("num_layers", 1)),
            dropout=float(model_cfg.get("dropout", 0.0)),
            lr=float(model_cfg.get("lr", 1e-3)),
            epochs=int(model_cfg.get("epochs", 10)),
            batch_size=int(model_cfg.get("batch_size", 128)),
            seq_len=int(model_cfg.get("seq_len", 60)),
        )

    raise ValueError(f"Unknown model name: {raw}")

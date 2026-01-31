from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List
import json
import numpy as np
from .metrics import mse, mae, qlike, realized_utility, realized_utility_tc

@dataclass
class MetricReport:
    mse: float
    mae: float
    qlike: float
    ru: float
    ru_tc: float

def compute_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sharpe_ratio: float = 0.4,
    risk_aversion: float = 2.0,
    tc_rate: float = 0.0,
) -> MetricReport:
    return MetricReport(
        mse=mse(y_true, y_pred),
        mae=mae(y_true, y_pred),
        qlike=qlike(y_true, y_pred),
        ru=realized_utility(y_true, y_pred, sharpe_ratio, risk_aversion),
        ru_tc=realized_utility_tc(y_true, y_pred, tc_rate, sharpe_ratio, risk_aversion),
    )

def save_report(report: MetricReport, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2)

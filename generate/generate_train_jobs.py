from __future__ import annotations
"""Generate a job list for training (demo)."""
import json
from pathlib import Path

JOBS = [
    {"config": "configs/toy_10min_MSAGRU_cluster.yaml"},
    {"config": "configs/toy_10min_GRU_universal.yaml"},
    {"config": "configs/toy_10min_OLS_single.yaml"},
]

def main() -> None:
    Path("data_centre/jobs").mkdir(parents=True, exist_ok=True)
    Path("data_centre/jobs/train_jobs.json").write_text(json.dumps(JOBS, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()

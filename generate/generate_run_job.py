from __future__ import annotations
"""Run a single job spec (JSON)."""
import argparse, json, subprocess
from pathlib import Path

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True, help="JSON string or path to a .json file")
    args = ap.parse_args()

    if args.job.endswith(".json"):
        job = json.loads(Path(args.job).read_text(encoding="utf-8"))
    else:
        job = json.loads(args.job)

    subprocess.check_call(["python", "scripts/run_toy_experiment.py", "--config", job["config"]])

if __name__ == "__main__":
    main()

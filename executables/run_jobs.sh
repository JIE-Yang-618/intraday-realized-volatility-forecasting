#!/usr/bin/env bash
set -euo pipefail
python generate/generate_train_jobs.py
python - << 'PY'
import json, subprocess
from pathlib import Path
jobs = json.loads(Path("data_centre/jobs/train_jobs.json").read_text(encoding="utf-8"))
for job in jobs:
    subprocess.check_call(["python","generate/generate_run_job.py","--job",json.dumps(job)])
PY

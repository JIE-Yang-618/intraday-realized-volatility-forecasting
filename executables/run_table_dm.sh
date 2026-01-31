#!/usr/bin/env bash
set -euo pipefail
python generate/generate_dm_inputs.py
python generate/generate_dm_tests.py

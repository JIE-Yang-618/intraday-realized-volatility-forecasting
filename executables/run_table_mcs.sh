#!/usr/bin/env bash
set -euo pipefail
python generate/generate_mcs_inputs.py
python generate/generate_mcs.py

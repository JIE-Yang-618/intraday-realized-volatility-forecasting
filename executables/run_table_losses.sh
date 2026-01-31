#!/usr/bin/env bash
set -euo pipefail
python generate/generate_predictions.py
python generate/generate_losses.py
python generate/generate_table_losses.py

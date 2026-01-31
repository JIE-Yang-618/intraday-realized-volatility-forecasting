#!/usr/bin/env bash
set -euo pipefail
python generate/generate_figure_placeholders.py
python generate/generate_figures.py

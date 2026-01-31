#!/usr/bin/env bash
set -euo pipefail

# Main driver (demo).
# The workflow is paper-oriented: shell drivers in executables/ orchestrate scripts in generate/.
# Full replication scripts will be released upon acceptance (or provided for academic verification on request).

bash init.sh

bash executables/run_pipeline_prep.sh
bash executables/run_jobs.sh

bash executables/run_table_models.sh
bash executables/run_table_losses.sh
bash executables/run_table_dm.sh
bash executables/run_table_mcs.sh

bash executables/run_backtest.sh
bash executables/run_figure_bundle.sh

echo "[main] done."

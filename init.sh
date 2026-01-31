#!/usr/bin/env bash
set -euo pipefail

# Initialise directories required by the project.
# NOTE: This public repository is a DEMO framework. The full preprocessing + full replication
# scripts will be released upon acceptance, or provided for academic verification upon request.

mkdir -p outputs logs figures data_centre
mkdir -p data_centre/cache data_centre/features data_centre/commonality data_centre/jobs data_centre/dm data_centre/mcs
echo "[init] created outputs/, logs/, figures/, data_centre/*"

# Contact: jieyang020618@gmail.com

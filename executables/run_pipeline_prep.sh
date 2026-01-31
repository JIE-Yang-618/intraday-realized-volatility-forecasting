#!/usr/bin/env bash
set -euo pipefail
python generate/generate_data_cache.py
python generate/generate_feature_store.py
python generate/generate_commonality_features.py
python generate/generate_train_jobs.py

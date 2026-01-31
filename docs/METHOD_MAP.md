# Method-to-Code Map (Demo)

This document maps the main components of the manuscript to the demo repository.

## Model families
- **Baselines:** `src/models/linear.py`, `src/models/mlp.py`
  - OLS, HAR-d, LASSO, MLP
- **Recurrent backbones:** `src/models/rnn.py`
  - LSTM / GRU (sequence over lagged RV)
- **Multi-Scale Attention (MSA):** `src/models/msa.py`
  - Multi-scale causal convolutions with horizons H in {3,5,10} days
  - Attention across horizons to form a descriptor
  - Recurrent backbone + forecast head
- **Signature-Augmented (SA):** `src/models/sa.py`
  - Interface stub: full signature extraction depends on the paper-specific path construction

## Training schemes
Implemented in `src/training/schemes.py` and `src/training/runner.py`.

- **Single:** per-asset model fits
- **Universal:** pooled parameters across all assets
- **Augmented:** pooled parameters + market RV feature
- **Cluster:** pooled within clusters + cluster RV feature (with optional market RV)

## Evaluation
- Loss metrics: `src/eval/metrics.py`
- Statistical tests: `src/eval/stats.py` (demo-level)

## Public demo disclaimer
This repository is intended for illustrating the project structure and facilitating collaboration.
The exact preprocessing and full experiment reproduction scripts will be released upon acceptance.

See `docs/TABLE_FIGURE_INDEX.md` for a paper-style artifact map.

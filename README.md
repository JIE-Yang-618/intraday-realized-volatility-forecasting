# Intraday Realized Volatility Forecasting

This repository is a **public demo** framework accompanying an academic project.

- The public demo ships **project structure, interfaces, and a runnable toy pipeline**.
- The full replication code (including the proprietary preprocessing pipeline) is **forthcoming upon acceptance**.
- Data used in the manuscript are **not distributed** (licensed/proprietary).

For academic verification/collaboration requests, see `CONTACT.md`.

## Repository layout
- `init.sh`: initialise directories used by the pipeline
- `main.sh`: demo driver (runs a small toy sweep and produces demo artifacts)
- `executables/`: shell entrypoints (paper-oriented orchestration)
- `generate/`: generator scripts (tables/figures/artifacts)
- `configs/`: experiment configs mirroring the manuscript taxonomy (models/schemes/horizons)
- `src/`: reusable model / training / evaluation modules
- `data_centre/`: local cache + intermediate artifacts (demo placeholders)
- `sql_scripts/`: optional database helpers (placeholder)
- `figures/`: figure outputs (demo placeholders)
- `docs/`: documentation, including a table/figure index

## Notes
- SA-LSTM/SA-GRU are **public interface stubs** in the demo (full implementation depends on the internal pipeline).
- MSA-LSTM/MSA-GRU are included as **demo implementations**.
- DM/MCS routines are exposed as **demo interfaces**.

See `docs/TABLE_FIGURE_INDEX.md` for the paper-style artifact map.

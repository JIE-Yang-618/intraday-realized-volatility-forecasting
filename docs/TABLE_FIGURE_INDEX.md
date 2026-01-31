# Table/Figure Index (Demo)

This file documents how the repository *organises* table/figure generation in a paper-oriented workflow.
The public demo runs on **toy data** and therefore does **not** reproduce the manuscript numbers.

## Key idea
- Each table/figure has a thin shell entry in `executables/`
- Each entry calls one or more scripts in `generate/`
- The heavy lifting is delegated to reusable modules under `src/`

## Tables (example mapping)
- **Forecast losses (MSE / QLIKE / RU / RU-TC)**
  - `executables/run_table_losses.sh`
  - `generate/generate_predictions.py`
  - `generate/generate_losses.py`
  - `generate/generate_table_losses.py`
  - `src/eval/*`

- **DM test (pairwise)**
  - `executables/run_table_dm.sh`
  - `generate/generate_dm_inputs.py`
  - `generate/generate_dm_tests.py`

- **MCS**
  - `executables/run_table_mcs.sh`
  - `generate/generate_mcs_inputs.py`
  - `generate/generate_mcs.py`

- **Model list**
  - `executables/run_table_models.sh`
  - `generate/generate_table_model_list.py`

## Figures (example mapping)
- **Forecast comparison (toy)**
  - `executables/run_figure_bundle.sh`
  - `generate/generate_figure_placeholders.py`

## Notes
- The manuscript version uses proprietary preprocessing and a licensed data source.
- The public demo ships *interfaces, structure, and a runnable toy pipeline* only.

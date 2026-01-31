from __future__ import annotations
"""Emit a model-family list (demo)."""
import json
from pathlib import Path

MODELS = ["OLS","HARD","LASSO","MLP","LSTM","GRU","SALSTM","SAGRU","MSALSTM","MSAGRU"]

def main() -> None:
    out = Path("outputs/tables")
    out.mkdir(parents=True, exist_ok=True)
    (out / "table_models.json").write_text(json.dumps(MODELS, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()

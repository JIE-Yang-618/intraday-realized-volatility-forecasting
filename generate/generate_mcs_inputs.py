from __future__ import annotations
"""Prepare MCS inputs (demo stub)."""
from pathlib import Path
import json

def main() -> None:
    Path("data_centre/mcs").mkdir(parents=True, exist_ok=True)
    Path("data_centre/mcs/mcs_inputs.json").write_text(json.dumps([], indent=2), encoding="utf-8")
    Path("data_centre/mcs/README.txt").write_text(
        "Demo MCS inputs placeholder. Full MCS inputs released upon acceptance.",
        encoding="utf-8",
    )

if __name__ == "__main__":
    main()

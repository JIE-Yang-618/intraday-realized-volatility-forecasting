from __future__ import annotations
"""Prepare DM inputs (demo stub)."""
from pathlib import Path
import json

def main() -> None:
    Path("data_centre/dm").mkdir(parents=True, exist_ok=True)
    Path("data_centre/dm/dm_inputs.json").write_text(json.dumps([], indent=2), encoding="utf-8")
    Path("data_centre/dm/README.txt").write_text(
        "Demo DM inputs placeholder. Full DM inputs released upon acceptance.",
        encoding="utf-8",
    )

if __name__ == "__main__":
    main()

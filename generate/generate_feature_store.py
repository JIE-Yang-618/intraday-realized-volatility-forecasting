from __future__ import annotations
"""Initialise a feature store (demo placeholder)."""
from pathlib import Path

def main() -> None:
    Path("data_centre/features").mkdir(parents=True, exist_ok=True)
    Path("data_centre/features/README.txt").write_text(
        "Demo feature-store placeholder. Full feature engineering released upon acceptance.",
        encoding="utf-8",
    )

if __name__ == "__main__":
    main()

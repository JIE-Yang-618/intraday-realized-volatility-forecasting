from __future__ import annotations
"""Prepare a local cache under data_centre/.

Public demo: creates directory placeholders only.
"""
from pathlib import Path

def main() -> None:
    Path("data_centre/cache").mkdir(parents=True, exist_ok=True)
    Path("data_centre/cache/README.txt").write_text(
        "Demo cache placeholder. Full preprocessing released upon acceptance.",
        encoding="utf-8",
    )

if __name__ == "__main__":
    main()

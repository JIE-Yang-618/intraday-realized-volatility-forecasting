from __future__ import annotations
"""Create placeholder figure outputs (demo)."""
from pathlib import Path

def main() -> None:
    Path("figures").mkdir(parents=True, exist_ok=True)
    (Path("figures") / "NOTE.txt").write_text(
        "Demo repository: figures are placeholders. Full figure scripts released upon acceptance.",
        encoding="utf-8",
    )

if __name__ == "__main__":
    main()

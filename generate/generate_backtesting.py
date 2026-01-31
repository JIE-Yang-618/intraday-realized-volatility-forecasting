from __future__ import annotations
"""Generate backtesting artifacts (demo placeholder)."""
import json, glob
from pathlib import Path

def main() -> None:
    out_dir = Path("outputs/backtest")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [{"path": p, "note": "Toy backtest placeholder (reports only)."} for p in glob.glob("outputs/**/report.json", recursive=True)]
    (out_dir / "backtest_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()

from __future__ import annotations
"""Assemble a paper-style losses table (demo)."""
import json, glob
from pathlib import Path

def main() -> None:
    table = []
    for rp in glob.glob("outputs/**/report.json", recursive=True):
        d = json.loads(Path(rp).read_text(encoding="utf-8"))
        table.append({
            "model": d.get("model"),
            "scheme": d.get("scheme"),
            "mse": d.get("mse"),
            "mae": d.get("mae"),
            "qlike": d.get("qlike"),
            "ru": d.get("ru"),
            "ru_tc": d.get("ru_tc"),
            "config": d.get("config"),
        })
    out = Path("outputs/tables")
    out.mkdir(parents=True, exist_ok=True)
    (out / "table_losses.json").write_text(json.dumps(table, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()

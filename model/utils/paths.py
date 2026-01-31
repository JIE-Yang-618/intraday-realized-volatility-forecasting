from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass
class ProjectPaths:
    root: Path
    outputs: Path
    logs: Path
    figures: Path

def make_paths(root: str | Path) -> ProjectPaths:
    root = Path(root)
    outputs = root / "outputs"
    logs = outputs / "logs"
    figures = outputs / "figures"
    for p in (outputs, logs, figures):
        p.mkdir(parents=True, exist_ok=True)
    return ProjectPaths(root=root, outputs=outputs, logs=logs, figures=figures)

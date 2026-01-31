from __future__ import annotations
import logging
from pathlib import Path

def get_logger(name: str = "rv") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

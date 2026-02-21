# utils/logging.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


def make_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.propagate = False
    return logger

def setup_logger(run_dir: str, name: str = "run") -> logging.Logger:
    """
    Compatibility wrapper used by scripts that expect setup_logger(run_dir, name).
    Creates run_dir and logs to run_dir/{name}.log plus stdout.
    """
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    log_file = str(Path(run_dir) / f"{name}.log")
    return make_logger(name=name, log_file=log_file)
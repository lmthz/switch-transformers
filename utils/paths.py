from __future__ import annotations
from pathlib import Path
from typing import Dict, Any

def make_run_dir(cfg: Dict[str, Any]) -> str:
    out_dir = Path(cfg["run"]["out_dir"])
    name = cfg["run"]["name"]
    run_dir = out_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return str(run_dir)
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, List

import yaml


def _deep_update(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_yaml(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config not found: {p}")
    with p.open("r") as f:
        obj = yaml.safe_load(f)
    if obj is None:
        obj = {}
    if not isinstance(obj, dict):
        raise ValueError("config must be a yaml mapping")
    return obj


def load_config(primary: str, overrides: Optional[List[str]] = None) -> Dict[str, Any]:
    cfg = load_yaml(primary)
    if overrides:
        for o in overrides:
            cfg = _deep_update(cfg, load_yaml(o))
    validate_config(cfg)
    return cfg


def validate_config(cfg: Dict[str, Any]) -> None:
    required_top = ["run", "data", "transformer", "msar"]
    for k in required_top:
        if k not in cfg:
            raise ValueError(f"missing top-level key '{k}' in config")

    run_req = ["name", "out_dir", "seed", "device", "resume"]
    for k in run_req:
        if k not in cfg["run"]:
            raise ValueError(f"missing run.{k}")

    data_req = ["data_dir", "dataset", "context_len", "val_frac"]
    for k in data_req:
        if k not in cfg["data"]:
            raise ValueError(f"missing data.{k}")

    tr_req = ["steps", "batch_size", "lr", "grad_clip", "log_every", "eval_every", "save_every", "model"]
    for k in tr_req:
        if k not in cfg["transformer"]:
            raise ValueError(f"missing transformer.{k}")

    mreq = ["d_model", "n_heads", "n_layers", "dropout"]
    for k in mreq:
        if k not in cfg["transformer"]["model"]:
            raise ValueError(f"missing transformer.model.{k}")

    msar_req = ["candidate_orders", "maxiter", "em_iter"]
    for k in msar_req:
        if k not in cfg["msar"]:
            raise ValueError(f"missing msar.{k}")


def save_config_snapshot(cfg: Dict[str, Any], run_dir: str) -> None:
    p = Path(run_dir) / "config_snapshot.yaml"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
# tasks/base.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Dict, Any


@dataclass
class TaskBatch:
    x: Any
    y: Any
    state: Any
    t_idx: Any


class ForecastTask(Protocol):
    name: str

    def loss(self, yhat, y) -> Any: ...
    def format_batch(self, batch, device: str) -> TaskBatch: ...
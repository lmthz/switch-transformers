# tasks/one_step_forecast.py
from __future__ import annotations
from dataclasses import dataclass
import torch
from tasks.base import TaskBatch


@dataclass
class OneStepForecastTask:
    name: str = "one_step"
    loss_fn: torch.nn.Module = torch.nn.MSELoss()

    def loss(self, yhat, y):
        return self.loss_fn(yhat, y)

    def format_batch(self, batch, device: str) -> TaskBatch:
        x, y, s, t_idx = batch
        x = x.to(device)
        y = y.to(device)
        if s is not None:
            s = s.to(device)
        t_idx = t_idx.to(device)
        return TaskBatch(x=x, y=y, state=s, t_idx=t_idx)
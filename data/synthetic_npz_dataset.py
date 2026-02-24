# data/synthetic_npz_dataset.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class SeriesStandardizer:
    mean: float
    std: float

    def transform(self, y: np.ndarray) -> np.ndarray:
        s = self.std if self.std > 0 else 1.0
        return (y - self.mean) / s

    def inverse(self, y_std: np.ndarray) -> np.ndarray:
        s = self.std if self.std > 0 else 1.0
        return y_std * s + self.mean


def load_npz(path: Path) -> Dict[str, Any]:
    arr = np.load(path)
    return {k: arr[k] for k in arr.files}


class SlidingWindowDataset(Dataset):
    """
    (context -> next) pairs from a single 1d standardized series.

    x = [y[t-L], ..., y[t-1]] shaped (L, 1)
    y = y[t] shaped (1,)
    """

    def __init__(
        self,
        y_std: np.ndarray,
        states: Optional[np.ndarray],
        context_len: int,
        start_t: int,
        end_t: int,
    ):
        y_std = np.asarray(y_std, dtype=np.float32)
        if y_std.ndim != 1:
            raise ValueError("y_std must be 1d")

        self.y = y_std
        self.states = None if states is None else np.asarray(states, dtype=np.int64)
        self.L = int(context_len)

        # valid target t must satisfy t-L >= 0 and t < len(y)
        start_t = int(start_t)
        end_t = int(end_t)
        start_t = max(start_t, self.L)
        end_t = min(end_t, len(self.y))

        if end_t <= start_t:
            raise ValueError("empty dataset after applying bounds/context_len")

        self.start_t = start_t
        self.end_t = end_t
        self._len = self.end_t - self.start_t

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, i: int):
        t = self.start_t + int(i)
        ctx = self.y[t - self.L : t]  # (L,)
        tgt = self.y[t]              # scalar
        x = torch.from_numpy(ctx).unsqueeze(-1)  # (L, 1)
        y = torch.tensor([tgt], dtype=torch.float32)  # (1,)

        if self.states is None:
            s = None
        else:
            s = torch.tensor(int(self.states[t]), dtype=torch.long)
        return x, y, s


def make_train_val_datasets(
    npz_path: str,
    context_len: int,
    val_frac: float,
) -> Tuple[SlidingWindowDataset, SlidingWindowDataset, SeriesStandardizer, Dict[str, Any]]:
    """
    time split:
      train: [0, n_train)
      val:   [n_train, n)

    standardization uses train slice only.
    """
    meta = load_npz(Path(npz_path))
    y = np.asarray(meta["y"], dtype=float)
    states = np.asarray(meta["states"], dtype=int) if "states" in meta else None

    n = len(y)
    n_train = int((1.0 - float(val_frac)) * n)
    n_train = max(1, min(n_train, n - 1))

    mean = float(y[:n_train].mean())
    std = float(y[:n_train].std())
    if std <= 0:
        std = 1.0
    stdzr = SeriesStandardizer(mean=mean, std=std)

    y_std = stdzr.transform(y)

    ds_train = SlidingWindowDataset(
        y_std=y_std,
        states=states,
        context_len=context_len,
        start_t=0,
        end_t=n_train,
    )
    ds_val = SlidingWindowDataset(
        y_std=y_std,
        states=states,
        context_len=context_len,
        start_t=n_train,
        end_t=n,
    )

    return ds_train, ds_val, stdzr, meta
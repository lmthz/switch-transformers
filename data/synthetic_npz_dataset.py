from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from metrics import train_val_split_indices


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
    def __init__(
        self,
        y_std: np.ndarray,
        states: Optional[np.ndarray],
        context_len: int,
        start_idx: int,
        end_idx: int,
    ):
        y_std = np.asarray(y_std, dtype=np.float32)
        if y_std.ndim != 1:
            raise ValueError("y_std must be 1d")

        self.y = y_std
        self.states = None if states is None else np.asarray(states, dtype=np.int64)
        self.L = int(context_len)

        self.start = max(int(start_idx), self.L)
        self.end = min(int(end_idx), len(self.y))

        if self.end <= self.start:
            raise ValueError("empty dataset after bounds/context")

        self._len = self.end - self.start

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, i: int):
        t = self.start + i
        ctx = self.y[t - self.L : t]                 # (L,)
        tgt = self.y[t]                              # scalar
        st = None if self.states is None else self.states[t]

        x = torch.from_numpy(ctx).unsqueeze(-1)      # (L, 1)
        y = torch.tensor([tgt], dtype=torch.float32) # (1,)
        s = None if st is None else torch.tensor(st, dtype=torch.long)
        return x, y, s, t


def make_train_val_datasets(
    npz_path: str,
    context_len: int,
    val_frac: float,
) -> Tuple[SlidingWindowDataset, SlidingWindowDataset, SeriesStandardizer, Dict[str, Any], int]:
    path = Path(npz_path)
    meta = load_npz(path)

    y = np.asarray(meta["y"], dtype=float)
    states = np.asarray(meta["states"], dtype=int) if "states" in meta else None

    n = len(y)
    n_train, n_total = train_val_split_indices(n, val_frac)

    y_train = y[:n_train]
    mean = float(y_train.mean())
    std = float(y_train.std())
    stdzr = SeriesStandardizer(mean=mean, std=std if std > 0 else 1.0)

    y_std = stdzr.transform(y)

    ds_train = SlidingWindowDataset(
        y_std=y_std,
        states=states,
        context_len=context_len,
        start_idx=0,
        end_idx=n_train,
    )
    ds_val = SlidingWindowDataset(
        y_std=y_std,
        states=states,
        context_len=context_len,
        start_idx=n_train,
        end_idx=n_total,
    )
    return ds_train, ds_val, stdzr, meta, n_train
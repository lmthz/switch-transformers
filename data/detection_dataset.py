"""
SwitchDetectionDataset — sliding-window dataset with soft decayed detection targets.

For each timestep t the target is:
    detection_target[t] = exp(-alpha * (t - s))
where s is the most recent switch at or before t, and alpha=0.5.
If no switch has occurred yet in the series, target=0.

Returns (x, y, detection_target) where:
    x : (context_len, 1) float32 — context window
    y : (1,)             float32 — next value
    detection_target : ()        float32 scalar in [0, 1]
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional

from data.synthetic_npz_dataset import SlidingWindowDataset


class SwitchDetectionDataset(Dataset):
    """
    Wraps a standardised series and its state labels to produce
    soft-decayed detection targets for training a detection model.

    Parameters
    ----------
    y_std : (N,) float32 — standardised series
    states : (N,) int — ground-truth regime labels (0 or 1)
    context_len : int
    start_t : int — first target timestep (must be >= context_len)
    end_t : int — one past the last target timestep
    alpha : float — decay rate for the soft target (default 0.5)
    """

    def __init__(
        self,
        y_std: np.ndarray,
        states: np.ndarray,
        context_len: int,
        start_t: int,
        end_t: int,
        alpha: float = 0.5,
    ):
        self.L = context_len
        self.alpha = alpha
        self.y = y_std.astype(np.float32)
        self.states = states

        self.start_t = max(start_t, self.L)
        self.end_t = min(end_t, len(self.y))

        # Precompute soft targets for every timestep in [0, len(y))
        self._targets = self._compute_targets()

    def _compute_targets(self) -> np.ndarray:
        T = len(self.y)
        targets = np.zeros(T, dtype=np.float32)
        last_switch = -1
        for t in range(1, T):
            if self.states[t] != self.states[t - 1]:
                last_switch = t
            if last_switch >= 0:
                targets[t] = float(np.exp(-self.alpha * (t - last_switch)))
        return targets

    def __len__(self) -> int:
        return max(0, self.end_t - self.start_t)

    def __getitem__(self, i: int):
        t = self.start_t + int(i)
        ctx = self.y[t - self.L : t]
        tgt = self.y[t]
        x = torch.from_numpy(ctx).unsqueeze(-1)          # (L, 1)
        y = torch.tensor([tgt], dtype=torch.float32)     # (1,)
        det = torch.tensor(self._targets[t], dtype=torch.float32)  # scalar
        return x, y, det


def make_detection_datasets(
    npz_path: str,
    context_len: int,
    val_frac: float = 0.3,
    alpha: float = 0.5,
):
    """
    Load an NPZ file and return (train_dataset, val_dataset, standardiser_info).
    Standardises y using train-split mean/std.

    Returns
    -------
    ds_train : SwitchDetectionDataset
    ds_val   : SwitchDetectionDataset
    info     : dict with keys 'mean', 'std', 'n_train', 'n_val'
    """
    meta = np.load(npz_path, allow_pickle=True)
    y = np.asarray(meta["y"], dtype=np.float32)
    states = np.asarray(meta["states"], dtype=np.int64)

    n = len(y)
    n_train = int(n * (1.0 - val_frac))
    n_val = n - n_train

    mean = float(y[:n_train].mean())
    std  = float(y[:n_train].std()) + 1e-8
    y_std = (y - mean) / std

    ds_train = SwitchDetectionDataset(y_std, states, context_len,
                                      start_t=context_len, end_t=n_train,
                                      alpha=alpha)
    ds_val   = SwitchDetectionDataset(y_std, states, context_len,
                                      start_t=n_train, end_t=n,
                                      alpha=alpha)
    info = {"mean": mean, "std": std, "n_train": n_train, "n_val": n_val}
    return ds_train, ds_val, info

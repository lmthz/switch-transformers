# metrics.py
from __future__ import annotations

from typing import Dict, Any, List, Tuple
import numpy as np


def train_val_split_indices(n: int, val_frac: float) -> Tuple[int, int]:
    n_train = int((1.0 - float(val_frac)) * int(n))
    n_train = max(1, min(n_train, n - 1))
    return n_train, n - n_train


def mse_rmse(err: np.ndarray) -> Tuple[float, float]:
    err = np.asarray(err, dtype=float)
    if err.size == 0:
        return float("nan"), float("nan")
    mse = float(np.mean(err ** 2))
    return mse, float(np.sqrt(mse))


def per_regime_rmse(err: np.ndarray, states: np.ndarray, k_regimes: int) -> List[Dict[str, Any]]:
    err = np.asarray(err, dtype=float)
    states = np.asarray(states, dtype=int)
    out: List[Dict[str, Any]] = []
    for k in range(int(k_regimes)):
        m = states == k
        if np.sum(m) == 0:
            continue
        mse_k = float(np.mean(err[m] ** 2))
        out.append({"regime": int(k), "n": int(np.sum(m)), "rmse": float(np.sqrt(mse_k))})
    return out


def label_corrected_accuracy(decoded: np.ndarray, true_states: np.ndarray, k_regimes: int) -> Dict[str, float]:
    """
    for k=2: return best of (no swap, swapped). for k>2: raw accuracy only.
    """
    decoded = np.asarray(decoded, dtype=int)
    true_states = np.asarray(true_states, dtype=int)
    m = min(decoded.size, true_states.size)
    if m <= 0:
        return {"acc": float("nan"), "acc_no_swap": float("nan"), "acc_swap": float("nan")}

    decoded = decoded[:m]
    true_states = true_states[:m]

    acc_no = float(np.mean(decoded == true_states))
    if int(k_regimes) != 2:
        return {"acc": acc_no, "acc_no_swap": acc_no, "acc_swap": float("nan")}

    acc_sw = float(np.mean((1 - decoded) == true_states))
    return {"acc": max(acc_no, acc_sw), "acc_no_swap": acc_no, "acc_swap": acc_sw}
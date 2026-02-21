from __future__ import annotations
from typing import Dict, Any, Tuple, Optional, List
import numpy as np

def mse_rmse(errors: np.ndarray) -> Tuple[float, float]:
    errors = np.asarray(errors, dtype=float)
    if errors.size == 0:
        return float("nan"), float("nan")
    mse = float(np.mean(errors ** 2))
    rmse = float(np.sqrt(mse))
    return mse, rmse

def per_regime_rmse(errors: np.ndarray, states: np.ndarray, k_regimes: int) -> List[Dict[str, Any]]:
    errors = np.asarray(errors, dtype=float)
    states = np.asarray(states, dtype=int)
    out: List[Dict[str, Any]] = []
    for k in range(k_regimes):
        mask = states == k
        if mask.sum() == 0:
            continue
        mse_k, rmse_k = mse_rmse(errors[mask])
        out.append({"regime": int(k), "n": int(mask.sum()), "mse": mse_k, "rmse": rmse_k})
    return out

def label_corrected_accuracy(decoded: np.ndarray, true_states: np.ndarray, k_regimes: int) -> Dict[str, float]:
    decoded = np.asarray(decoded, dtype=int)
    true_states = np.asarray(true_states, dtype=int)

    if decoded.size == 0 or true_states.size == 0:
        return {"acc": float("nan"), "acc_no_swap": float("nan"), "acc_swap": float("nan")}

    m = min(len(decoded), len(true_states))
    decoded = decoded[:m]
    true_states = true_states[:m]

    if k_regimes != 2:
        acc = float(np.mean(decoded == true_states))
        return {"acc": acc, "acc_no_swap": acc, "acc_swap": float("nan")}

    acc_no = float(np.mean(decoded == true_states))
    acc_sw = float(np.mean((1 - decoded) == true_states))
    return {"acc": max(acc_no, acc_sw), "acc_no_swap": acc_no, "acc_swap": acc_sw}

def train_val_split_indices(n: int, val_frac: float) -> Tuple[int, int]:
    n_train = int((1.0 - val_frac) * n)
    return n_train, n
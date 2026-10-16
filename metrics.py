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


# ── Switch-aware metrics ──────────────────────────────────────────

def near_switch_mask(states: np.ndarray, window: int = 20) -> np.ndarray:
    """Boolean mask: True for timesteps within `window` steps of any regime transition."""
    states = np.asarray(states, dtype=int)
    switch_pts = np.where(np.diff(states) != 0)[0] + 1
    mask = np.zeros(len(states), dtype=bool)
    for s in switch_pts:
        mask[max(0, s - window):min(len(states), s + window)] = True
    return mask


def post_switch_spike_ratio(sq_err: np.ndarray, states: np.ndarray, window: int = 10) -> float:
    """
    Mean squared error in the first `window` steps after each regime switch, divided
    by the global mean squared error. Values > 1 indicate error spikes at transitions.
    Returns nan if the series has no switches.
    """
    states = np.asarray(states, dtype=int)
    sq_err = np.asarray(sq_err, dtype=float)
    switch_pts = np.where(np.diff(states) != 0)[0] + 1
    if len(switch_pts) == 0:
        return float("nan")
    global_mse = sq_err.mean()
    if global_mse == 0:
        return float("nan")
    n = len(sq_err)
    post_vals = [sq_err[s:min(s + window, n)].mean() for s in switch_pts]
    return float(np.mean(post_vals) / global_mse)


def recovery_time(sq_err: np.ndarray, states: np.ndarray,
                  threshold_factor: float = 1.5, max_window: int = 50) -> float:
    """
    Mean steps after each regime switch until error falls below
    threshold_factor * steady_state_mse. Returns max_window if never recovered.
    Returns nan if no switches. Steady-state MSE excludes the 10 steps around switches.
    """
    states = np.asarray(states, dtype=int)
    sq_err = np.asarray(sq_err, dtype=float)
    switch_pts = np.where(np.diff(states) != 0)[0] + 1
    if len(switch_pts) == 0:
        return float("nan")
    n = len(sq_err)
    near = np.zeros(n, dtype=bool)
    for s in switch_pts:
        near[max(0, s - 5):min(n, s + 10)] = True
    steady = sq_err[~near]
    threshold = threshold_factor * (steady.mean() if len(steady) else sq_err.mean())
    times = []
    for s in switch_pts:
        t = 0
        while t < max_window and s + t < n:
            if sq_err[s + t] <= threshold:
                break
            t += 1
        times.append(t)
    return float(np.mean(times))


def _regime_id_predicted(sq_err: np.ndarray, rolling_window: int = 10) -> np.ndarray:
    """Threshold rolling MSE at the series median to produce a binary regime prediction."""
    rolling = np.convolve(sq_err, np.ones(rolling_window) / rolling_window, mode="same")
    return (rolling > np.median(rolling)).astype(int)


def regime_id_accuracy_from_errors(sq_err: np.ndarray, states: np.ndarray,
                                   rolling_window: int = 10) -> float:
    """
    Global regime ID accuracy: infer regime from rolling error magnitude across all
    timesteps, compare to true states. Returns nan if only one regime present.
    """
    states = np.asarray(states, dtype=int)
    if len(np.unique(states)) < 2:
        return float("nan")
    predicted = _regime_id_predicted(sq_err, rolling_window)
    return float(label_corrected_accuracy(predicted, states, k_regimes=2)["acc_no_swap"])


def regime_id_accuracy_near_switch(sq_err: np.ndarray, states: np.ndarray,
                                   window: int = 20, rolling_window: int = 10) -> float:
    """
    Near-switch regime ID accuracy: same error-threshold classifier evaluated only on
    timesteps within `window` steps of a regime transition.
    Returns nan if no switches or fewer than 5 near-switch timesteps.
    """
    states = np.asarray(states, dtype=int)
    if len(np.unique(states)) < 2:
        return float("nan")
    mask = near_switch_mask(states, window)
    if mask.sum() < 5:
        return float("nan")
    predicted = _regime_id_predicted(sq_err, rolling_window)
    return float(label_corrected_accuracy(predicted[mask], states[mask], k_regimes=2)["acc_no_swap"])


def regime_id_accuracy_steady_state(sq_err: np.ndarray, states: np.ndarray,
                                    window: int = 20, rolling_window: int = 10) -> float:
    """
    Steady-state regime ID accuracy: same classifier evaluated only on timesteps
    at least `window` steps from any regime transition.
    Returns nan if no switches or fewer than 5 steady-state timesteps.
    """
    states = np.asarray(states, dtype=int)
    if len(np.unique(states)) < 2:
        return float("nan")
    mask = ~near_switch_mask(states, window)
    if mask.sum() < 5:
        return float("nan")
    predicted = _regime_id_predicted(sq_err, rolling_window)
    return float(label_corrected_accuracy(predicted[mask], states[mask], k_regimes=2)["acc_no_swap"])


# ── Linear probe for hidden-state regime identification ──────────────

class LinearProbe:
    """
    Logistic regression probe: tests whether regime identity is linearly
    decodable from transformer hidden states.
    """

    def __init__(self) -> None:
        self._clf = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearProbe":
        """X: (N, d_model)  y: (N,) int binary"""
        try:
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
            clf.fit(X, y)
            self._clf = clf
        except ImportError:
            # Fallback: least-squares binary classifier
            X_bias = np.hstack([X, np.ones((len(X), 1))])
            w, _, _, _ = np.linalg.lstsq(X_bias, y.astype(float), rcond=None)
            self._clf = ("lstsq", w)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns (N,) int binary predictions."""
        if self._clf is None:
            raise RuntimeError("Call fit() before predict()")
        if isinstance(self._clf, tuple):
            w = self._clf[1]
            X_bias = np.hstack([X, np.ones((len(X), 1))])
            return (X_bias @ w >= 0.5).astype(int)
        return self._clf.predict(X).astype(int)


def regime_id_accuracy_from_probe(
    hidden: np.ndarray, states: np.ndarray, probe: LinearProbe
) -> float:
    """Overall regime ID accuracy using a fitted linear probe on hidden states."""
    states = np.asarray(states, dtype=int)
    if len(np.unique(states)) < 2:
        return float("nan")
    predicted = probe.predict(hidden)
    return float(label_corrected_accuracy(predicted, states, k_regimes=2)["acc_no_swap"])


def regime_id_accuracy_near_switch_probe(
    hidden: np.ndarray, states: np.ndarray, probe: LinearProbe, window: int = 20
) -> float:
    """Near-switch regime ID accuracy using a fitted linear probe."""
    states = np.asarray(states, dtype=int)
    if len(np.unique(states)) < 2:
        return float("nan")
    mask = near_switch_mask(states, window)
    if mask.sum() < 5:
        return float("nan")
    predicted = probe.predict(hidden)
    return float(label_corrected_accuracy(predicted[mask], states[mask], k_regimes=2)["acc_no_swap"])


def regime_id_accuracy_steady_state_probe(
    hidden: np.ndarray, states: np.ndarray, probe: LinearProbe, window: int = 20
) -> float:
    """Steady-state regime ID accuracy using a fitted linear probe."""
    states = np.asarray(states, dtype=int)
    if len(np.unique(states)) < 2:
        return float("nan")
    mask = ~near_switch_mask(states, window)
    if mask.sum() < 5:
        return float("nan")
    predicted = probe.predict(hidden)
    return float(label_corrected_accuracy(predicted[mask], states[mask], k_regimes=2)["acc_no_swap"])


# ── Per-switch-in-context analysis ───────────────────────────────────

def per_switch_metrics(
    sq_err: np.ndarray,
    states: np.ndarray,
    context_len: int = 64,
    spike_window: int = 10,
    recovery_threshold: float = 1.5,
    recovery_max: int = 50,
    k_values: List[int] = None,
) -> List[Dict[str, Any]]:
    """
    For each switch point, returns metrics broken down by how many prior
    switches were visible in the model's context window at that moment.

    switches_in_context counts prior switch points in [position-context_len, position).
    This is the relevant quantity for a transformer with fixed context_len — switches
    further back than context_len are invisible to the model.

    Returns a list of dicts (one per switch), empty if no switches.
    """
    if k_values is None:
        k_values = [1, 5, 10, 20]

    states = np.asarray(states, dtype=int)
    sq_err = np.asarray(sq_err, dtype=float)
    switch_pts = list(np.where(np.diff(states) != 0)[0] + 1)
    if not switch_pts:
        return []

    n = len(sq_err)
    global_mse = float(sq_err.mean()) if n > 0 else float("nan")

    # Steady-state MSE: exclude 10 steps around each switch
    near = np.zeros(n, dtype=bool)
    for s in switch_pts:
        near[max(0, s - 5):min(n, s + 10)] = True
    steady_vals = sq_err[~near]
    steady_mse = float(steady_vals.mean()) if len(steady_vals) > 0 else global_mse
    threshold = recovery_threshold * steady_mse

    results = []
    for ordinal, s in enumerate(switch_pts):
        # Count prior switches visible in context window
        window_start = s - context_len
        switches_in_context = sum(
            1 for prev in switch_pts[:ordinal] if prev >= window_start
        )

        # Spike ratio
        post = sq_err[s:min(s + spike_window, n)]
        spike = float(post.mean() / global_mse) if global_mse > 0 and len(post) > 0 else float("nan")

        # Post-switch MSE (raw)
        post_mse = float(post.mean()) if len(post) > 0 else float("nan")

        # Recovery time
        t = 0
        while t < recovery_max and s + t < n:
            if sq_err[s + t] <= threshold:
                break
            t += 1
        rec = float(t)

        # Per-k post-switch MSE: sq_err at exactly k steps after switch
        k_mse = {}
        for k in k_values:
            idx = s + k - 1
            k_mse[k] = float(sq_err[idx]) if idx < n else float("nan")

        # Cumulative excess loss over recovery_max horizon
        post_horizon = sq_err[s:min(s + recovery_max, n)]
        excess = float(np.sum(post_horizon) - steady_mse * len(post_horizon)) if len(post_horizon) > 0 else float("nan")

        row = {
            "switch_ordinal":      ordinal,
            "position":            int(s),
            "switches_in_context": switches_in_context,
            "spike_ratio":         spike,
            "recovery_time":       rec,
            "post_switch_mse":     post_mse,
            "cumulative_excess_loss": excess,
        }
        for k in k_values:
            row[f"post_switch_mse_k{k}"] = k_mse[k]

        results.append(row)

    return results


# ── Additional metrics for adaptation method evaluation ──────────────


def post_switch_error_curve(
    sq_err: np.ndarray,
    states: np.ndarray,
    k_values: List[int] = None,
) -> Dict[int, float]:
    """
    Mean squared error at exactly k steps after each regime switch,
    averaged across all switch events.
    Returns {k: mean_sq_err_at_k} for each k in k_values.
    """
    if k_values is None:
        k_values = [1, 5, 10, 20]
    sq_err = np.asarray(sq_err, dtype=float)
    states = np.asarray(states, dtype=int)
    switch_pts = list(np.where(np.diff(states) != 0)[0] + 1)
    if not switch_pts:
        return {k: float("nan") for k in k_values}
    n = len(sq_err)
    result = {}
    for k in k_values:
        vals = [float(sq_err[s + k - 1]) for s in switch_pts if s + k - 1 < n]
        result[k] = float(np.mean(vals)) if vals else float("nan")
    return result


def cumulative_excess_loss(
    sq_err: np.ndarray,
    states: np.ndarray,
    horizon: int = 50,
) -> float:
    """
    Total prediction penalty during adaptation, summed over `horizon` steps
    after each switch and averaged across switch events.
    Excess = sum(sq_err[t0:t0+horizon]) - steady_state_mse * horizon.
    """
    sq_err = np.asarray(sq_err, dtype=float)
    states = np.asarray(states, dtype=int)
    switch_pts = list(np.where(np.diff(states) != 0)[0] + 1)
    if not switch_pts:
        return float("nan")
    n = len(sq_err)
    near = np.zeros(n, dtype=bool)
    for s in switch_pts:
        near[max(0, s - 5):min(n, s + 10)] = True
    steady_vals = sq_err[~near]
    steady_mse = float(steady_vals.mean()) if len(steady_vals) > 0 else float(sq_err.mean())
    excesses = []
    for s in switch_pts:
        post = sq_err[s:min(s + horizon, n)]
        if len(post) > 0:
            excesses.append(float(np.sum(post) - steady_mse * len(post)))
    return float(np.mean(excesses)) if excesses else float("nan")


def false_switch_rate(
    fired_times: List[int],
    true_switch_pts: List[int],
    tolerance: int = 5,
) -> float:
    """
    Fraction of CUSUM firings that are NOT within `tolerance` steps of
    any true regime switch. A false alarm rate for the change detector.
    Returns nan if no firings.
    """
    if not fired_times:
        return float("nan")
    true_set = set(true_switch_pts)
    false_alarms = 0
    for f in fired_times:
        near_any = any(abs(f - s) <= tolerance for s in true_set)
        if not near_any:
            false_alarms += 1
    return float(false_alarms) / len(fired_times)
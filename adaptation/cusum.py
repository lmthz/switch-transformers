from __future__ import annotations

from typing import Dict, List, Any

import numpy as np


class CUSUM:
    """
    One-sided CUSUM change-point detector.

    Tracks S_t = max(0, S_{t-1} + residual - drift).
    Fires (declares a change) when S_t > threshold, then resets S to 0.

    residual should be (observed_value - expected_value) for upward shifts,
    or squared_error - baseline_mse when applied to prediction residuals.
    """

    def __init__(self, threshold: float = 5.0, drift: float = 0.5):
        self.threshold = threshold
        self.drift = drift
        self._S = 0.0

    @property
    def statistic(self) -> float:
        return self._S

    def update(self, residual: float) -> tuple:
        """Update with one new residual. Returns (S_t, fired)."""
        self._S = max(0.0, self._S + float(residual) - self.drift)
        fired = self._S > self.threshold
        if fired:
            self._S = 0.0
        return self._S, fired

    def reset(self):
        self._S = 0.0


def cusum_detection_delays(
    sq_err: np.ndarray,
    states: np.ndarray,
    threshold: float = 5.0,
    drift: float = 0.5,
    max_delay: int = 50,
) -> List[Dict[str, Any]]:
    """
    Run CUSUM on transformer squared errors and record detection delay
    for each true regime switch.

    The baseline MSE is estimated from the first 20% of the series
    (assumed pre-switch). residual_t = sq_err[t] - baseline_mse.

    Returns a list of dicts (one per switch):
        switch_position, fire_position, delay_steps
    delay_steps = max_delay if CUSUM never fires before the next switch.
    """
    sq_err = np.asarray(sq_err, dtype=float)
    states = np.asarray(states, dtype=int)
    switch_pts = list(np.where(np.diff(states) != 0)[0] + 1)
    if not switch_pts:
        return []

    n = len(sq_err)
    baseline_mse = float(sq_err[:max(1, n // 5)].mean())

    results = []
    cusum = CUSUM(threshold=threshold, drift=drift)
    fire_times: List[int] = []

    # Single pass: record all firing times
    for t in range(n):
        _, fired = cusum.update(sq_err[t] - baseline_mse)
        if fired:
            fire_times.append(t)

    # For each switch, find first fire after it (before next switch)
    for i, s in enumerate(switch_pts):
        next_switch = switch_pts[i + 1] if i + 1 < len(switch_pts) else n
        fires_after = [f for f in fire_times if s <= f < next_switch]
        if fires_after:
            fire_pos = fires_after[0]
            delay = fire_pos - s
        else:
            fire_pos = -1
            delay = max_delay
        results.append({
            "switch_position": int(s),
            "fire_position":   fire_pos,
            "delay_steps":     delay,
        })

    return results


def cusum_on_raw(
    y: np.ndarray,
    states: np.ndarray,
    threshold: float = 5.0,
    drift: float = 0.5,
    max_delay: int = 50,
    baseline_window: int = 20,
) -> List[Dict[str, Any]]:
    """
    Run CUSUM on the raw observation series (not prediction errors) and
    record detection delay for each true regime switch.

    residual_t = (y_t - mu)^2 where mu is a rolling mean over the last
    `baseline_window` observations (pre-switch baseline).

    Returns same format as cusum_detection_delays.
    """
    y = np.asarray(y, dtype=float)
    states = np.asarray(states, dtype=int)
    switch_pts = list(np.where(np.diff(states) != 0)[0] + 1)
    if not switch_pts:
        return []

    n = len(y)
    results = []
    cusum = CUSUM(threshold=threshold, drift=drift)
    fire_times: List[int] = []

    # Rolling mean as local baseline
    mu = float(y[:baseline_window].mean()) if n >= baseline_window else float(y.mean())
    baseline_var = float(np.var(y[:max(1, n // 5)])) + 1e-8

    for t in range(n):
        if t >= baseline_window:
            mu = float(y[t - baseline_window:t].mean())
        dev = float((y[t] - mu) ** 2 / baseline_var)
        _, fired = cusum.update(dev - 1.0)  # residual relative to unit variance
        if fired:
            fire_times.append(t)

    for i, s in enumerate(switch_pts):
        next_switch = switch_pts[i + 1] if i + 1 < len(switch_pts) else n
        fires_after = [f for f in fire_times if s <= f < next_switch]
        if fires_after:
            fire_pos = fires_after[0]
            delay = fire_pos - s
        else:
            fire_pos = -1
            delay = max_delay
        results.append({
            "switch_position": int(s),
            "fire_position":   fire_pos,
            "delay_steps":     delay,
        })

    return results


def cusum_fire_times(
    sq_err: np.ndarray,
    baseline_mse: float,
    threshold: float = 5.0,
    drift: float = 0.5,
) -> List[int]:
    """Return list of timesteps where CUSUM fires on sq_err series."""
    cusum = CUSUM(threshold=threshold, drift=drift)
    fires = []
    for t, e in enumerate(sq_err):
        _, fired = cusum.update(float(e) - baseline_mse)
        if fired:
            fires.append(t)
    return fires

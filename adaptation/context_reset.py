from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch

from adaptation.cusum import CUSUM


class ContextResetModel:
    """
    Inference-time wrapper around a frozen CausalTransformerForecaster.

    Runs CUSUM on the transformer's squared prediction errors. When CUSUM
    fires (declaring a likely regime change), the observation buffer is
    truncated to only post-switch observations so the transformer is not
    anchored to pre-switch dynamics.

    Drop-in replacement for CausalTransformerForecaster at eval time.
    Exposes the same predict_next() and get_hidden() interface.

    Usage (sequential, one observation at a time):
        wrap = ContextResetModel(model, threshold=5.0)
        for t in range(T):
            x = wrap.get_context_tensor(device)   # (1, L, 1)
            y_hat = wrap.predict_next(x).item()
            wrap.observe(y_obs[t], y_hat)

    Usage (batch / pre-built context):
        # Pass x directly — context reset not applied, plain forward pass.
        y_hat = wrap.predict_next(x)
    """

    def __init__(
        self,
        model,
        cusum_threshold: float = 5.0,
        cusum_drift: float = 0.5,
        min_context: int = 8,
        baseline_window: int = 50,
    ):
        self.model = model
        self.context_len = model.cfg.context_len
        self.min_context = min_context
        self.baseline_window = baseline_window

        self.cusum = CUSUM(threshold=cusum_threshold, drift=cusum_drift)
        self._buffer: List[float] = []
        self._sq_errs: List[float] = []
        self.reset_times: List[int] = []
        self._t = 0

    # ── public interface ───────────────────────────────────────────

    def observe(self, y_true: float, y_hat: float) -> bool:
        """
        Record a new observation and update CUSUM.
        Returns True if a context reset was triggered.
        """
        self._buffer.append(float(y_true))
        sq_err = (float(y_true) - float(y_hat)) ** 2
        self._sq_errs.append(sq_err)

        baseline_mse = self._baseline_mse()
        _, fired = self.cusum.update(sq_err - baseline_mse)

        if fired:
            # Keep min_context most recent observations
            self._buffer = self._buffer[-self.min_context:]
            self.reset_times.append(self._t)

        self._t += 1
        return fired

    def get_context_tensor(self, device) -> torch.Tensor:
        """
        Build a (1, L, 1) context tensor from the current buffer,
        truncated/padded to context_len.
        """
        buf = self._buffer[-self.context_len:]
        arr = np.array(buf, dtype=np.float32).reshape(1, -1, 1)
        return torch.tensor(arr, device=device)

    def predict_next(self, x: torch.Tensor) -> torch.Tensor:
        """(B, L, 1) -> (B, 1) next-step forecast. Plain forward pass."""
        return self.model.predict_next(x)

    def get_hidden(self, x: torch.Tensor) -> torch.Tensor:
        """(B, L, 1) -> (B, d_model) final-position hidden state."""
        return self.model.get_hidden(x)

    def reset(self):
        """Reset buffer and CUSUM state."""
        self._buffer = []
        self._sq_errs = []
        self.cusum.reset()
        self.reset_times = []
        self._t = 0

    # ── internal ──────────────────────────────────────────────────

    def _baseline_mse(self) -> float:
        if len(self._sq_errs) < 2:
            return 0.0
        window = self._sq_errs[-self.baseline_window:]
        return float(np.mean(window))


def eval_series_context_reset(
    base_model,
    npz_path: str,
    context_len: int,
    val_frac: float,
    device,
    cusum_threshold: float = 5.0,
    cusum_drift: float = 0.5,
    min_context: int = 8,
    dataset_name: str = "",
    instance: int = 0,
):
    """
    Evaluate the CUSUM context-reset model on one .npz series.

    Steps sequentially through the val split one observation at a time,
    maintaining the rolling buffer and CUSUM state. Records per-step
    predictions, then computes per-switch metrics.

    Returns a dict with keys:
        rmse, spike_ratio, recovery_time, n_resets, false_switch_rate,
        post_switch_mse_k1/5/10/20, cumulative_excess_loss,
        per_switch (list of dicts)
    """
    from metrics import (
        train_val_split_indices, post_switch_spike_ratio, recovery_time,
        per_switch_metrics, cumulative_excess_loss, false_switch_rate,
    )

    data = np.load(npz_path)
    y_full = data["y"].astype(np.float32).flatten()
    states_full = data["states"].astype(int).flatten()

    n_total = len(y_full)
    n_train, n_val = train_val_split_indices(n_total, val_frac)
    y_val = y_full[n_train:]
    states_val = states_full[n_train:]
    n = len(y_val)

    if n < 2:
        return {}

    wrap = ContextResetModel(
        base_model,
        cusum_threshold=cusum_threshold,
        cusum_drift=cusum_drift,
        min_context=min_context,
    )

    # Warm up buffer with training observations (no CUSUM update)
    for y in y_full[:n_train]:
        wrap._buffer.append(float(y))

    preds = []
    for t in range(n):
        x = wrap.get_context_tensor(device)
        with torch.no_grad():
            y_hat = float(wrap.predict_next(x)[0, 0].cpu())
        preds.append(y_hat)
        wrap.observe(y_val[t], y_hat)

    preds = np.array(preds, dtype=float)
    sq_err = (y_val - preds) ** 2

    switch_pts = list(np.where(np.diff(states_val) != 0)[0] + 1)
    true_switch_positions = [int(s) for s in switch_pts]

    rmse = float(np.sqrt(sq_err.mean()))
    spike = post_switch_spike_ratio(sq_err, states_val)
    rec = recovery_time(sq_err, states_val)
    cel = cumulative_excess_loss(sq_err, states_val)
    fsr = false_switch_rate(wrap.reset_times, true_switch_positions)
    per_sw = per_switch_metrics(sq_err, states_val, context_len=context_len)

    return {
        "dataset":               dataset_name,
        "instance":              instance,
        "rmse":                  rmse,
        "spike_ratio":           spike,
        "recovery_time":         rec,
        "n_resets":              len(wrap.reset_times),
        "false_switch_rate":     fsr,
        "cumulative_excess_loss": cel,
        "per_switch":            per_sw,
    }

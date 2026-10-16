from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch

from adaptation.cusum import CUSUM
from adaptation.rls_head import RLSHead


class RLSTransformerModel:
    """
    Frozen transformer backbone with an online RLS prediction head.

    The transformer provides d_model=256 hidden state features at each
    timestep. The RLS head maps those features to a scalar prediction and
    updates its weights after every observation — no gradient descent.

    Three variants (controlled by forgetting_factor and use_cusum):
        - RLS fixed λ=0.95   (fast adaptation)
        - RLS fixed λ=0.99   (slow adaptation)
        - RLS + CUSUM-adaptive λ: λ=λ_stable normally; drops to λ_fast
          when CUSUM fires on the residuals.

    Usage (sequential, one observation at a time):
        rls = RLSTransformerModel(model, forgetting_factor=0.95)
        for t in range(T):
            x = build_context(y_obs[:t], context_len)  # (1, L, 1)
            y_hat, fired = rls.predict_and_update(x, y_obs[t])
    """

    def __init__(
        self,
        model,
        forgetting_factor: float = 0.95,
        use_cusum_adaptive_lambda: bool = False,
        lambda_stable: float = 0.99,
        lambda_fast: float = 0.80,
        cusum_threshold: float = 5.0,
        cusum_drift: float = 0.5,
    ):
        self.model = model
        d_model = model.cfg.d_model

        # If use_cusum_adaptive_lambda, forgetting_factor is ignored in favour
        # of lambda_stable/lambda_fast schedule
        self._use_adaptive = use_cusum_adaptive_lambda
        self._lambda_stable = lambda_stable
        self._lambda_fast = lambda_fast
        lam = lambda_stable if use_cusum_adaptive_lambda else forgetting_factor
        self.rls = RLSHead(d_features=d_model, forgetting_factor=lam)

        self.cusum: Optional[CUSUM] = None
        if use_cusum_adaptive_lambda:
            self.cusum = CUSUM(threshold=cusum_threshold, drift=cusum_drift)

        self._sq_errs: List[float] = []
        self.reset_times: List[int] = []
        self._t = 0

    # ── main interface ─────────────────────────────────────────────

    def predict_and_update(
        self, x: torch.Tensor, y_true: float
    ) -> tuple:
        """
        x: (1, L, 1) context tensor (built externally from observation buffer).
        y_true: scalar — the observation for the current timestep.

        Returns (y_hat, cusum_fired):
            y_hat       — scalar prediction made *before* seeing y_true
            cusum_fired — True if CUSUM detected a switch this step
        """
        with torch.no_grad():
            h = self.model.get_hidden(x)  # (1, d_model)
        h_np = h[0].cpu().numpy()         # (d_model,)

        y_hat = self.rls.update(h_np, float(y_true))

        sq_err = (float(y_true) - y_hat) ** 2
        self._sq_errs.append(sq_err)

        fired = False
        if self.cusum is not None:
            baseline = self._baseline_mse()
            _, fired = self.cusum.update(sq_err - baseline)
            if fired:
                # Increase uncertainty — adapt faster for next observations
                self.rls.reset_uncertainty()
                self.rls.lam = self._lambda_fast
                self.reset_times.append(self._t)
            else:
                # Gradually recover forgetting factor toward stable value
                self.rls.lam = min(self._lambda_stable, self.rls.lam * 1.05)

        self._t += 1
        return y_hat, fired

    def reset(self):
        self.rls.reset()
        if self.cusum is not None:
            self.cusum.reset()
        self._sq_errs = []
        self.reset_times = []
        self._t = 0

    # ── internal ──────────────────────────────────────────────────

    def _baseline_mse(self, window: int = 50) -> float:
        if not self._sq_errs:
            return 0.0
        return float(np.mean(self._sq_errs[-window:]))


def eval_series_rls(
    rls_model: RLSTransformerModel,
    npz_path: str,
    context_len: int,
    val_frac: float,
    device,
    dataset_name: str = "",
    instance: int = 0,
) -> dict:
    """
    Evaluate an RLSTransformerModel on one .npz series.

    Steps sequentially through the val split one observation at a time.
    Returns a dict with rmse, spike_ratio, recovery_time,
    cumulative_excess_loss, post_switch_mse_k{1,5,10,20},
    n_resets, per_switch (list of dicts).
    """
    from metrics import (
        train_val_split_indices, post_switch_spike_ratio, recovery_time,
        per_switch_metrics, cumulative_excess_loss,
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

    rls_model.reset()

    # Build a rolling observation buffer, pre-seeded with training data
    buffer: List[float] = list(y_full[:n_train].astype(float))

    preds = []
    for t in range(n):
        ctx = np.array(buffer[-context_len:], dtype=np.float32).reshape(1, -1, 1)
        x = torch.tensor(ctx, device=device)
        y_hat, _ = rls_model.predict_and_update(x, float(y_val[t]))
        preds.append(y_hat)
        buffer.append(float(y_val[t]))

    preds_arr = np.array(preds, dtype=float)
    sq_err = (y_val.astype(float) - preds_arr) ** 2

    rmse = float(np.sqrt(sq_err.mean()))
    spike = post_switch_spike_ratio(sq_err, states_val)
    rec = recovery_time(sq_err, states_val)
    cel = cumulative_excess_loss(sq_err, states_val)
    per_sw = per_switch_metrics(sq_err, states_val, context_len=context_len)

    return {
        "dataset":                dataset_name,
        "instance":               instance,
        "rmse":                   rmse,
        "spike_ratio":            spike,
        "recovery_time":          rec,
        "cumulative_excess_loss": cel,
        "n_resets":               len(rls_model.reset_times),
        "per_switch":             per_sw,
    }


def eval_suite_rls(
    base_model,
    data_dir: str,
    datasets: List[str],
    n_instances: int,
    context_len: int,
    val_frac: float,
    device,
    forgetting_factors: List[float] = None,
    use_cusum_variants: bool = True,
    out_csv: str = "results_rls.csv",
):
    """
    Evaluate all RLS variants on a suite of datasets.

    Runs three variants per dataset:
        - RLS λ=0.95 (fixed fast)
        - RLS λ=0.99 (fixed slow)
        - RLS + CUSUM-adaptive λ (if use_cusum_variants=True)

    Returns and saves a DataFrame.
    """
    import os
    import pandas as pd
    from pathlib import Path

    if forgetting_factors is None:
        forgetting_factors = [0.95, 0.99]

    variants = {f"rls_lam{lam}": {"forgetting_factor": lam, "use_cusum_adaptive_lambda": False}
                for lam in forgetting_factors}
    if use_cusum_variants:
        variants["rls_cusum_adaptive"] = {
            "forgetting_factor": 0.99,
            "use_cusum_adaptive_lambda": True,
        }

    rows = []
    for ds in datasets:
        for ri in range(n_instances):
            npz_path = str(Path(data_dir) / f"{ds}_r{ri}.npz")
            if not os.path.exists(npz_path):
                continue
            for variant_name, kwargs in variants.items():
                rls = RLSTransformerModel(base_model, **kwargs)
                result = eval_series_rls(
                    rls, npz_path, context_len, val_frac, device,
                    dataset_name=ds, instance=ri,
                )
                if result:
                    result["variant"] = variant_name
                    rows.append(result)

    if not rows:
        return pd.DataFrame()

    # Flatten per_switch list into separate rows for per-switch analysis
    flat_rows = []
    for r in rows:
        per_sw = r.pop("per_switch", [])
        for sw in per_sw:
            flat_rows.append({**r, **sw})

    df = pd.DataFrame(flat_rows)
    df.to_csv(out_csv, index=False)
    print(f"Saved {len(df)} per-switch rows to {out_csv}")
    return df

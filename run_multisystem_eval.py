"""
run_multisystem_eval.py

Evaluates the thesis switch-filter architecture against classical baselines.

Methods:
  oracle         — exact AR likelihoods + Bayesian switch filter (needs ar/sigma in NPZ)
  xformer_filter — transformer likelihoods + Bayesian switch filter (per-regime checkpoints)
  bocpd          — Bayesian Online Changepoint Detection (classical baseline)
  cusum          — CUSUM on raw observations (classical baseline)

For oracle/xformer_filter: feeds per-regime log-likelihoods into a Bayesian HMM
forward recursion using the stored transition matrix T from the NPZ. The filter
outputs P(z_t=k | x_{1:t}); detection score = 1 - max(posterior).

Output CSV: results_multisystem_eval.csv
  Columns: dataset, method, threshold, mean_latency, miss_rate, far_per100, arl0, f1

Usage:
    python run_multisystem_eval.py --dataset A1_ar2_coeffs_easy --methods all
    python run_multisystem_eval.py --all_datasets --methods all
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch

from adaptation.bocpd import bocpd_score_series
from metrics import (
    detection_events, mean_detection_latency,
    false_alarm_rate_per100, arl0 as compute_arl0,
    f1_at_tolerance,
)
from models.transformer_forecaster import CausalTransformerForecaster, TransformerConfig
from models.switch_filter import (
    SwitchFilter, build_log_T, ar_log_lik, transformer_log_lik,
    filter_detection_scores,
)

# ── Constants ──────────────────────────────────────────────────────────────────

DATASETS = [
    "A1_ar2_coeffs_easy", "A2_ar2_coeffs_hard",
    "A3_ar2_coeffs_plus_var",
    "B1_ar2_variance",    "B2_ar2_variance_big",
    "C1_arma21_coeffs_var",
    "D1_arima211",        "D2_arima221",         "D3_arima210",
    "E1_drift_only",      "E2_level_shift",
    "F1_seasonal_sarimax","F2_seasonal_exog",
    "G1_exogenous_only",
    "H1_ar10_coeffs",     "H2_ar1_near_unit_root",
    "S1_sparse_switching","S2_frequent_switching",
]

ALL_METHODS = ["oracle", "xformer_filter", "bocpd", "cusum"]

# For K=2 regimes, 1 - max(posterior) ≤ 0.5; cap threshold grid at 0.49
THRESHOLDS_FILTER = np.linspace(0.05, 0.49, 19)
THRESHOLDS_BOCPD  = np.concatenate([
    np.linspace(0.5, 3.0, 10),
    np.linspace(3.0, 8.0, 10)[1:],
])
THRESHOLDS_CUSUM  = np.concatenate([
    np.linspace(0.1, 2.0, 10),
    np.linspace(2.0, 10.0, 10)[1:],
])

# ── Data loading ───────────────────────────────────────────────────────────────

def load_val_split(npz_path: str, val_frac: float = 0.3):
    """Load standardised val split, returning y_val, states_val, mean, std."""
    data = np.load(npz_path)
    y      = np.asarray(data["y"],      dtype=np.float64).flatten()
    states = np.asarray(data["states"], dtype=int).flatten()
    n      = len(y)
    n_train = int(n * (1.0 - val_frac))
    mean = float(y[:n_train].mean())
    std  = float(y[:n_train].std()) + 1e-8
    return (y - mean)[n_train:] / std, states[n_train:], mean, std


def find_switch_positions(states: np.ndarray) -> List[int]:
    return [t for t in range(1, len(states)) if states[t] != states[t - 1]]


# ── Score series helpers ───────────────────────────────────────────────────────

def scores_cusum(y_val: np.ndarray, drift: float = 0.5,
                 baseline_window: int = 20) -> np.ndarray:
    """Return per-step CUSUM statistic (standardised squared-deviation residual)."""
    n = len(y_val)
    baseline_var = float(np.var(y_val[:max(1, n // 5)])) + 1e-8
    scores = np.zeros(n, dtype=np.float64)
    S  = 0.0
    mu = float(y_val[:min(baseline_window, n)].mean())
    for t in range(n):
        if t >= baseline_window:
            mu = float(y_val[t - baseline_window : t].mean())
        dev = float((y_val[t] - mu) ** 2 / baseline_var)
        S = max(0.0, S + dev - 1.0 - drift)
        scores[t] = S
    return scores


def scores_oracle_filter(
    y_val: np.ndarray,
    npz_data: dict,
    stdzr_std: float,
    context_len: int,
) -> Optional[np.ndarray]:
    """
    Exact AR(p) log-likelihoods for each regime → switch filter scores.
    Returns None if the dataset does not have plain AR coefficients (ARMA/ARIMA).
    For ARIMA datasets the AR oracle is applied to the DIFFERENCED series.
    """
    ar     = npz_data.get("ar")      # (K, p) or None
    sigma  = npz_data.get("sigma")   # (K,)   or None
    T_mat  = npz_data.get("T")       # (K, K)

    if ar is None or sigma is None or T_mat is None:
        return None

    # Skip ARIMA (stored ar applies to differenced series z, not y)
    # and ARMA (MA component ignored → biased likelihood).
    if "z" in npz_data or "ma" in npz_data:
        return None

    ar    = np.asarray(ar,    dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    K = ar.shape[0]
    p = ar.shape[1]

    # In standardised space: AR coefficients are scale-invariant.
    # Noise std scales as sigma[k] / stdzr_std.
    sigma_std = sigma / stdzr_std

    log_T = build_log_T(np.asarray(T_mat, dtype=np.float64))
    filt  = SwitchFilter(log_T)
    T_val = len(y_val)

    posteriors = np.zeros((T_val, K))
    for t in range(T_val):
        if t < p:
            # Not enough history — use uniform likelihoods
            log_liks = np.full(K, -np.log(K))
        else:
            log_liks = np.array([
                ar_log_lik(y_val[t], y_val[max(0, t - p):t], ar[k], sigma_std[k])
                for k in range(K)
            ])
        posteriors[t] = filt.update(log_liks)

    return filter_detection_scores(posteriors)


def scores_transformer_filter(
    y_val: np.ndarray,
    context_len: int,
    dataset: str,
    ckpt_dir: str,
    npz_data: dict,
    device: torch.device,
) -> Optional[np.ndarray]:
    """
    Per-regime transformer log-likelihoods → switch filter scores.
    Returns None if checkpoints are missing for any regime.
    """
    K = 2   # always 2 regimes for existing datasets
    T_mat = npz_data.get("T")
    if T_mat is None:
        return None

    # Load per-regime transformers
    models: List[CausalTransformerForecaster] = []
    sigma_hats: List[float] = []

    for k in range(K):
        path = str(Path(ckpt_dir) / f"{dataset}_regime{k}_final.pt")
        if not os.path.exists(path):
            return None
        saved = torch.load(path, map_location=device)
        cfg   = TransformerConfig(**saved["cfg"])
        m     = CausalTransformerForecaster(cfg).to(device)
        m.load_state_dict(saved["model_state"])
        m.eval()
        models.append(m)
        sigma_hats.append(float(saved["sigma_hat"]))

    log_T = build_log_T(np.asarray(T_mat, dtype=np.float64))
    filt  = SwitchFilter(log_T)
    T_val = len(y_val)
    posteriors = np.zeros((T_val, K))

    with torch.no_grad():
        for t in range(T_val):
            if t < context_len:
                log_liks = np.full(K, -np.log(K))
            else:
                ctx = y_val[t - context_len : t].reshape(1, -1, 1).astype(np.float32)
                x   = torch.tensor(ctx, device=device)
                log_liks = np.zeros(K)
                for k, (m, sh) in enumerate(zip(models, sigma_hats)):
                    pred = float(m(x)[0, -1, 0].cpu())
                    log_liks[k] = transformer_log_lik(float(y_val[t]), pred, sh)
            posteriors[t] = filt.update(log_liks)

    return filter_detection_scores(posteriors)


# ── Threshold → fire times ─────────────────────────────────────────────────────

def scores_to_fire_times(scores: np.ndarray, threshold: float) -> List[int]:
    """Edge-triggered: fire once per upward crossing of threshold."""
    fire_times = []
    above = False
    for t, s in enumerate(scores):
        if s >= threshold and not above:
            fire_times.append(t)
            above = True
        elif s < threshold:
            above = False
    return fire_times


# ── Metric computation ─────────────────────────────────────────────────────────

def compute_metrics_at_threshold(
    scores: np.ndarray,
    switch_positions: List[int],
    threshold: float,
    n_steps: int,
    max_delay: int = 50,
) -> dict:
    fire_times = scores_to_fire_times(scores, threshold)
    events     = detection_events(fire_times, switch_positions, max_delay=max_delay)
    n_switches = len(switch_positions)
    miss_rate  = sum(1 for e in events if not e["detected"]) / max(n_switches, 1)
    return {
        "threshold":    threshold,
        "mean_latency": mean_detection_latency(events),
        "miss_rate":    miss_rate,
        "far_per100":   false_alarm_rate_per100(fire_times, switch_positions, n_steps),
        "arl0":         compute_arl0(fire_times, switch_positions, n_steps),
        "f1":           f1_at_tolerance(fire_times, switch_positions),
    }


# ── Per-dataset evaluation ─────────────────────────────────────────────────────

def eval_dataset(
    dataset: str,
    npz_path: str,
    methods: List[str],
    ckpt_dir: str,
    context_len: int,
    val_frac: float,
    bocpd_lambda: float,
    cusum_drift: float,
    device: torch.device,
    max_delay: int = 50,
) -> List[dict]:
    if not os.path.exists(npz_path):
        print(f"  [skip] {dataset} — {npz_path} not found")
        return []

    y_val, states_val, stdzr_mean, stdzr_std = load_val_split(npz_path, val_frac)
    switch_positions = find_switch_positions(states_val)
    n_steps = len(y_val)
    npz_data = dict(np.load(npz_path))

    if not switch_positions:
        print(f"  [skip] {dataset} — no switches in val split")
        return []

    rows = []

    for method in methods:
        # Compute score series
        if method == "oracle":
            scores = scores_oracle_filter(y_val, npz_data, stdzr_std, context_len)
            thresholds = THRESHOLDS_FILTER
        elif method == "xformer_filter":
            scores = scores_transformer_filter(
                y_val, context_len, dataset, ckpt_dir, npz_data, device)
            thresholds = THRESHOLDS_FILTER
        elif method == "bocpd":
            scores = bocpd_score_series(y_val, hazard_lambda=bocpd_lambda)
            thresholds = THRESHOLDS_BOCPD
        elif method == "cusum":
            scores = scores_cusum(y_val, drift=cusum_drift)
            thresholds = THRESHOLDS_CUSUM
        else:
            continue

        if scores is None:
            print(f"  [{dataset}] {method}: skipped (no data/checkpoints)")
            continue

        for thr in thresholds:
            m = compute_metrics_at_threshold(
                scores, switch_positions, thr, n_steps, max_delay)
            rows.append({"dataset": dataset, "method": method, **m})

        # Best-F1 summary row
        best_row = max(
            (r for r in rows if r["dataset"] == dataset and r["method"] == method),
            key=lambda r: r["f1"],
            default=None,
        )
        if best_row:
            print(
                f"  [{dataset}] {method}: "
                f"latency={best_row['mean_latency']:.1f}  "
                f"f1={best_row['f1']:.3f}  "
                f"far={best_row['far_per100']:.2f}"
            )

    return rows


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()

    tgt = ap.add_mutually_exclusive_group()
    tgt.add_argument("--dataset",      type=str)
    tgt.add_argument("--all_datasets", action="store_true")

    ap.add_argument("--methods",   nargs="+", default=["all"])
    ap.add_argument("--data_dir",  type=str, default="generated_data")
    ap.add_argument("--instance",  type=int, default=0)
    ap.add_argument("--ckpt_dir",  type=str, default="runs/per_regime")
    ap.add_argument("--context_len",  type=int,   default=64)
    ap.add_argument("--val_frac",     type=float, default=0.3)
    ap.add_argument("--bocpd_lambda", type=float, default=100.0)
    ap.add_argument("--cusum_drift",  type=float, default=0.5)
    ap.add_argument("--max_delay",    type=int,   default=50)
    ap.add_argument("--out",    type=str, default="results_multisystem_eval.csv")
    ap.add_argument("--device", type=str, default="cuda")

    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)

    datasets = DATASETS if args.all_datasets else [args.dataset]
    if not args.all_datasets and args.dataset is None:
        ap.error("Supply --dataset <name> or --all_datasets")

    methods = ALL_METHODS if "all" in args.methods else args.methods

    all_rows = []
    for ds in datasets:
        npz_path = str(Path(args.data_dir) / f"{ds}_r{args.instance}.npz")
        if not os.path.exists(npz_path):
            npz_path = str(Path(args.data_dir) / f"{ds}.npz")
        print(f"\n=== {ds} ===")
        rows = eval_dataset(
            dataset=ds,
            npz_path=npz_path,
            methods=methods,
            ckpt_dir=args.ckpt_dir,
            context_len=args.context_len,
            val_frac=args.val_frac,
            bocpd_lambda=args.bocpd_lambda,
            cusum_drift=args.cusum_drift,
            device=device,
            max_delay=args.max_delay,
        )
        all_rows.extend(rows)

    if all_rows:
        df = pd.DataFrame(all_rows)
        df.to_csv(args.out, index=False)
        print(f"\nSaved {len(df)} rows to {args.out}")
    else:
        print("No results generated.")


if __name__ == "__main__":
    main()

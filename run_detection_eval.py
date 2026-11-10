"""
run_detection_eval.py

Unified evaluation of all detection methods on the held-out val split of each dataset.
Sweeps thresholds to generate latency/FAR trade-off curves.

Methods evaluated:
  cusum      — CUSUM on raw observations (adaptation/cusum.py)
  bocpd      — BOCPD with Normal-Gamma conjugate prior (adaptation/bocpd.py)
  probe      — Linear probe on frozen base transformer hidden states (metrics.py)
  det_only   — ComparisonDetectionTransformer trained with detection_only loss
  multitask  — ComparisonDetectionTransformer trained with multitask loss

Output CSV columns:
  dataset, method, threshold, mean_latency, far_per100, arl0, f1_tol3, f1_tol5

Usage:
    python run_detection_eval.py --methods cusum bocpd --dataset A1_ar2_coeffs_easy
    python run_detection_eval.py --all_datasets --methods all
    python run_detection_eval.py --all_datasets --methods all --out results_detection_eval.csv
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from metrics import (
    train_val_split_indices, LinearProbe,
    detection_events, mean_detection_latency,
    false_alarm_rate_per100, arl0 as compute_arl0,
    f1_at_tolerance, detection_roc,
)
from adaptation.cusum import cusum_on_raw, cusum_fire_times
from adaptation.bocpd import bocpd_score_series
from data.detection_dataset import make_detection_datasets
from models.transformer_forecaster import CausalTransformerForecaster, TransformerConfig
from models.detection_transformer import ComparisonDetectionTransformer

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

ALL_METHODS = ["cusum", "bocpd", "probe", "det_only", "multitask"]

# Threshold sweep grids (different scales for different method families)
# Neural/probe methods output sigmoid in [0,1]
THRESHOLDS_PROB = np.linspace(0.05, 0.95, 19)
# CUSUM outputs cumulative sum statistic (non-negative, units of std-devs²)
THRESHOLDS_CUSUM = np.concatenate([
    np.linspace(0.1, 2.0, 10),
    np.linspace(2.0, 10.0, 10)[1:],
])
# BOCPD outputs -log P(x_t | x_{1:t-1}); typical background ≈1–2, spikes ≈3–10
THRESHOLDS_BOCPD = np.concatenate([
    np.linspace(0.5, 3.0, 10),
    np.linspace(3.0, 8.0, 10)[1:],
])

# ── Data loading helpers ───────────────────────────────────────────────────────

def load_val_split(npz_path: str, val_frac: float = 0.3):
    """Load the val portion of a dataset, standardised on the train split."""
    data = np.load(npz_path)
    y      = np.asarray(data["y"], dtype=np.float32).flatten()
    states = np.asarray(data["states"], dtype=np.int64).flatten()
    n      = len(y)
    n_train, _ = train_val_split_indices(n, val_frac)

    mean = float(y[:n_train].mean())
    std  = float(y[:n_train].std()) + 1e-8
    y_std = (y - mean) / std

    return y_std[n_train:], states[n_train:], mean, std


def find_switch_positions(states: np.ndarray) -> List[int]:
    return [t for t in range(1, len(states)) if states[t] != states[t - 1]]


# ── Score series helpers ───────────────────────────────────────────────────────

def scores_cusum(
    y_val: np.ndarray,
    cusum_drift: float = 0.5,
    baseline_window: int = 20,
) -> np.ndarray:
    """
    Return per-step CUSUM statistic (higher = more evidence of change).
    Residual: (y_t - rolling_mean)^2 / baseline_var - 1  (same as cusum_on_raw).
    """
    n = len(y_val)
    baseline_var = float(np.var(y_val[:max(1, n // 5)])) + 1e-8
    scores = np.zeros(n, dtype=np.float64)
    S = 0.0
    mu = float(y_val[:min(baseline_window, n)].mean())
    for t in range(n):
        if t >= baseline_window:
            mu = float(y_val[t - baseline_window : t].mean())
        dev = float((y_val[t] - mu) ** 2 / baseline_var)
        S = max(0.0, S + dev - 1.0 - cusum_drift)
        scores[t] = S
    return scores


def scores_bocpd(y_val: np.ndarray, hazard_lambda: float = 100.0) -> np.ndarray:
    """Return per-step -log P(x_t | x_{1:t-1}): high after switches, low during stable segments."""
    return bocpd_score_series(y_val, hazard_lambda=hazard_lambda)


def scores_probe(
    y_val: np.ndarray,
    states_val: np.ndarray,
    context_len: int,
    base_ckpt_dir: str,
    dataset: str,
    device,
) -> np.ndarray | None:
    """
    Return per-step probe detection score using a frozen base transformer.
    Scores are logistic regression probabilities for 'switch recently'.
    Returns None if no base checkpoint is found for this dataset.
    """
    base_path = str(Path(base_ckpt_dir) / f"{dataset}_final.pt")
    if not os.path.exists(base_path):
        return None

    ckpt = torch.load(base_path, map_location=device)
    cfg  = TransformerConfig(**ckpt["cfg"])
    base_model = CausalTransformerForecaster(cfg).to(device)
    base_model.load_state_dict(ckpt["model_state"])
    base_model.eval()

    T = len(y_val)
    scores = np.zeros(T, dtype=np.float64)

    # Build hidden-state features and labels for probe training (use first 50%)
    n_probe_train = max(context_len + 1, int(T * 0.5))

    hidden_list, label_list = [], []
    switch_set = set(find_switch_positions(states_val))

    with torch.no_grad():
        for t in range(context_len, T):
            ctx = y_val[t - context_len : t].reshape(1, -1, 1)
            x   = torch.tensor(ctx, dtype=torch.float32, device=device)
            h   = base_model.get_hidden(x)                # (1, L, d)
            feat = h[0, -1, :].cpu().numpy()              # (d,)
            hidden_list.append(feat)
            # label: switch in recent 5 steps?
            lbl = int(any((t - k) in switch_set for k in range(1, 6)))
            label_list.append(lbl)

    if len(hidden_list) < 20:
        return None

    H = np.stack(hidden_list, axis=0)   # (T-L, d)
    y_probe = np.array(label_list)

    probe = LinearProbe()
    probe.fit(H[:n_probe_train - context_len],
              y_probe[:n_probe_train - context_len])

    # predict() returns binary; use decision_function/raw output when available
    clf = probe._clf
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(H)[:, 1]
    elif hasattr(clf, "decision_function"):
        raw = clf.decision_function(H)
        probs = 1.0 / (1.0 + np.exp(-raw))   # sigmoid
    elif isinstance(clf, tuple):
        # lstsq fallback: raw score before threshold
        w = clf[1]
        H_bias = np.hstack([H, np.ones((len(H), 1))])
        probs = np.clip(H_bias @ w, 0.0, 1.0)
    else:
        probs = probe.predict(H).astype(np.float64)

    scores[context_len:] = probs
    return scores


def scores_comparison_transformer(
    y_val: np.ndarray,
    context_len: int,
    ckpt_path: str,
    device,
) -> np.ndarray | None:
    """Return per-step sigmoid detection score from a ComparisonDetectionTransformer."""
    if not os.path.exists(ckpt_path):
        return None

    ckpt   = torch.load(ckpt_path, map_location=device)
    cfg    = TransformerConfig(**ckpt["cfg"])
    split_k = ckpt.get("split_k", 8)
    model  = ComparisonDetectionTransformer(cfg, split_k=split_k).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # y_val is already standardised by load_val_split — use it directly
    T = len(y_val)
    scores = np.zeros(T, dtype=np.float64)

    with torch.no_grad():
        for t in range(context_len, T):
            ctx = y_val[t - context_len : t].reshape(1, -1, 1)
            x   = torch.tensor(ctx, dtype=torch.float32, device=device)
            score = model.detection_score(x)   # (1, 1)
            scores[t] = float(score[0, 0].cpu())

    return scores


# ── Metric rows from a score array ────────────────────────────────────────────

def rows_from_scores(
    method: str,
    dataset: str,
    scores: np.ndarray,
    switch_positions: List[int],
    thresholds: np.ndarray,
    tolerance: int = 5,
    max_delay: int = 50,
) -> List[Dict[str, Any]]:
    """Sweep thresholds and return one metric-row dict per threshold."""
    roc = detection_roc(
        scores, switch_positions,
        n_steps=len(scores),
        thresholds=thresholds,
        tolerance=tolerance,
        max_delay=max_delay,
    )
    rows = []
    for entry in roc:
        rows.append({
            "dataset":      dataset,
            "method":       method,
            "threshold":    entry["threshold"],
            "mean_latency": entry["mean_latency"],
            "far_per100":   entry["far_per100"],
            "f1_tol5":      entry["f1"],
            # f1_tol3 uses a tighter tolerance
        })

    # Second sweep for f1_tol3
    roc3 = detection_roc(
        scores, switch_positions,
        n_steps=len(scores),
        thresholds=thresholds,
        tolerance=3,
        max_delay=max_delay,
    )
    for i, entry3 in enumerate(roc3):
        rows[i]["f1_tol3"] = entry3["f1"]

    return rows


# ── Per-dataset evaluation ─────────────────────────────────────────────────────

def eval_dataset(
    dataset: str,
    data_dir: str,
    methods: List[str],
    det_only_ckpt_dir: str,
    multitask_ckpt_dir: str,
    base_ckpt_dir: str,
    context_len: int,
    val_frac: float,
    device,
    instance: int = 0,
    cusum_drift: float = 0.5,
    bocpd_lambda: float = 100.0,
) -> List[Dict[str, Any]]:
    npz_path = str(Path(data_dir) / f"{dataset}_r{instance}.npz")
    if not os.path.exists(npz_path):
        print(f"  [skip] {dataset} — npz not found")
        return []

    y_val, states_val, mean, std = load_val_split(npz_path, val_frac)
    switch_positions = find_switch_positions(states_val)

    if not switch_positions:
        print(f"  [skip] {dataset} — no switches in val split")
        return []

    rows = []

    for method in methods:
        scores = None

        if method == "cusum":
            scores = scores_cusum(y_val, cusum_drift=cusum_drift)
            thresholds = THRESHOLDS_CUSUM

        elif method == "bocpd":
            scores = scores_bocpd(y_val, hazard_lambda=bocpd_lambda)
            thresholds = THRESHOLDS_BOCPD

        elif method == "probe":
            scores = scores_probe(
                y_val, states_val, context_len,
                base_ckpt_dir, dataset, device,
            )
            thresholds = THRESHOLDS_PROB

        elif method == "det_only":
            ckpt_path = str(Path(det_only_ckpt_dir) / f"{dataset}_final.pt")
            scores = scores_comparison_transformer(y_val, context_len, ckpt_path, device)
            thresholds = THRESHOLDS_PROB

        elif method == "multitask":
            ckpt_path = str(Path(multitask_ckpt_dir) / f"{dataset}_final.pt")
            scores = scores_comparison_transformer(y_val, context_len, ckpt_path, device)
            thresholds = THRESHOLDS_PROB

        if scores is None:
            print(f"  [skip] {dataset}/{method} — model/scores not available")
            continue

        method_rows = rows_from_scores(
            method, dataset, scores, switch_positions, thresholds
        )
        rows.extend(method_rows)
        if method_rows:
            best = max(method_rows, key=lambda r: r["f1_tol5"])
            print(f"    {dataset:30s}  {method:12s}  "
                  f"best_f1={best['f1_tol5']:.3f}  "
                  f"latency={best['mean_latency']:.1f}  "
                  f"FAR={best['far_per100']:.2f}")

    return rows


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()

    tgt = ap.add_mutually_exclusive_group()
    tgt.add_argument("--dataset",      type=str)
    tgt.add_argument("--all_datasets", action="store_true")

    ap.add_argument("--methods",   nargs="+", default=["all"],
                    help="Methods: cusum bocpd probe det_only multitask (or 'all')")
    ap.add_argument("--data_dir",  type=str, default="generated_data")
    ap.add_argument("--instance",  type=int, default=0,
                    help="Dataset instance suffix (default 0 → uses <dataset>_r0.npz)")
    ap.add_argument("--out",       type=str, default="results_detection_eval.csv")

    ap.add_argument("--det_only_ckpt_dir",  type=str,
                    default="runs/detection_detection_only")
    ap.add_argument("--multitask_ckpt_dir", type=str,
                    default="runs/detection_multitask")
    ap.add_argument("--base_ckpt_dir",      type=str,
                    default="runs/base_transformer",
                    help="Base transformer checkpoints for linear probe")

    ap.add_argument("--context_len",   type=int,   default=64)
    ap.add_argument("--val_frac",      type=float, default=0.3)
    ap.add_argument("--cusum_drift",   type=float, default=0.5)
    ap.add_argument("--bocpd_lambda",  type=float, default=100.0,
                    help="BOCPD hazard lambda (expected steps between switches)")
    ap.add_argument("--device",        type=str,   default="cuda")

    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)

    methods = ALL_METHODS if "all" in args.methods else args.methods
    for m in methods:
        if m not in ALL_METHODS:
            ap.error(f"Unknown method '{m}'. Choose from: {ALL_METHODS}")

    if args.all_datasets:
        datasets = DATASETS
    elif args.dataset is not None:
        datasets = [args.dataset]
    else:
        ap.error("Supply --dataset <name> or --all_datasets")

    print(f"Evaluating {len(datasets)} dataset(s) × {len(methods)} method(s)")

    all_rows = []
    for ds in datasets:
        print(f"\n{ds}")
        rows = eval_dataset(
            dataset=ds,
            data_dir=args.data_dir,
            methods=methods,
            det_only_ckpt_dir=args.det_only_ckpt_dir,
            multitask_ckpt_dir=args.multitask_ckpt_dir,
            base_ckpt_dir=args.base_ckpt_dir,
            context_len=args.context_len,
            val_frac=args.val_frac,
            device=device,
            instance=args.instance,
            cusum_drift=args.cusum_drift,
            bocpd_lambda=args.bocpd_lambda,
        )

        all_rows.extend(rows)

    if not all_rows:
        print("No results generated.")
        return

    df = pd.DataFrame(all_rows)
    df.to_csv(args.out, index=False)
    print(f"\nSaved {len(df)} rows to {args.out}")

    # Quick summary: best F1 per (dataset, method)
    if "f1_tol5" in df.columns:
        summary = (df.groupby(["dataset", "method"])
                     .apply(lambda g: g.loc[g["f1_tol5"].idxmax()])
                     .reset_index(drop=True)
                   [["dataset", "method", "threshold", "mean_latency",
                     "far_per100", "f1_tol5"]])
        print("\n=== Best F1 per (dataset, method) ===")
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

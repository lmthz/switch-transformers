"""
run_detection_latency.py

Three-way detection latency comparison for each regime switch event:
  1. CUSUM on raw observations           -> cusum_raw_delay
  2. CUSUM on transformer residuals      -> cusum_residual_delay
  3. Linear probe on hidden states       -> probe_delay

Also evaluates:
  4. CUSUM-gated context reset           -> context reset RMSE / recovery time
  5. RLS adaptive head (3 variants)      -> RLS RMSE / recovery time

All methods use the same trained model (no retraining).
Outputs:
  results_detection_latency.csv   — per-switch delays for all three methods
  results_context_reset.csv       — per-switch metrics for context reset
  results_rls.csv                 — per-switch metrics for RLS variants

Usage:
    python run_detection_latency.py --pool_path series_pool.npz
    python run_detection_latency.py --train_steps 1000   # smoke test
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import torch

from train_transformer import train_iid, resolve_device
from run_density_experiment import (
    DATASETS_B1, FAMILY_PRESETS,
    build_model, build_sampler, get_val_monitor_loader,
)
from metrics import (
    train_val_split_indices, LinearProbe,
    per_switch_metrics,
)
from adaptation.cusum import cusum_detection_delays, cusum_on_raw
from adaptation.context_reset import eval_series_context_reset
from adaptation.rls_model import RLSTransformerModel, eval_series_rls, eval_suite_rls


def probe_detection_delays(
    model,
    npz_path: str,
    context_len: int,
    val_frac: float,
    batch_size: int,
    device,
    max_delay: int = 50,
) -> List[Dict[str, Any]]:
    """
    Fit a linear probe on the first 50% of the val split, then for each
    switch event in the second 50%, record how many steps after the switch
    the probe first correctly classifies the new regime.

    Returns list of {switch_position, probe_fire_position, delay_steps}.
    """
    data = np.load(npz_path)
    y_full = data["y"].astype(np.float32).flatten()
    states_full = data["states"].astype(int).flatten()
    n_total = len(y_full)
    n_train, n_val = train_val_split_indices(n_total, val_frac)

    y_val = y_full[n_train:]
    states_val = states_full[n_train:]
    n = len(y_val)
    if n < 10:
        return []

    # Build rolling context windows for val split (sliding window over combined series)
    windows, state_labels = [], []
    for t in range(n):
        full = np.concatenate([y_full[:n_train], y_val[:t + 1]])
        win = full[-context_len:]
        if len(win) < context_len:
            win = np.pad(win, (context_len - len(win), 0))
        windows.append(win)
        state_labels.append(states_val[t])
    windows = np.array(windows, dtype=np.float32)
    state_labels = np.array(state_labels, dtype=int)

    # Collect hidden states in batches
    all_hidden = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(windows), batch_size):
            x = torch.tensor(windows[i:i + batch_size, :, None], device=device)
            h = model.get_hidden(x)
            all_hidden.append(h.cpu().numpy())
    H = np.concatenate(all_hidden, axis=0)  # (n_val, d_model)

    # Fit probe on first half
    mid = n // 2
    probe = LinearProbe().fit(H[:mid], state_labels[:mid])

    # Per-timestep probe predictions on second half
    pred_labels = probe.predict(H[mid:])
    true_labels = state_labels[mid:]

    # Switch points in the second half (offset by mid)
    switch_pts_full = list(np.where(np.diff(states_val) != 0)[0] + 1)
    switch_pts_second = [s - mid for s in switch_pts_full if s >= mid]

    results = []
    n_second = len(pred_labels)
    for i, s in enumerate(switch_pts_second):
        if s < 0 or s >= n_second:
            continue
        correct_label = true_labels[s]
        # Find first timestep >= s where probe is correct
        fire_pos = -1
        for t in range(s, min(s + max_delay, n_second)):
            if pred_labels[t] == correct_label:
                fire_pos = t
                break
        delay = (fire_pos - s) if fire_pos >= 0 else max_delay
        results.append({
            "switch_position":     int(s + mid),  # offset back to full val index
            "probe_fire_position": int(fire_pos + mid) if fire_pos >= 0 else -1,
            "delay_steps":         int(delay),
        })

    return results


def run_detection_latency_suite(
    model,
    data_dir: str,
    datasets: List[str],
    n_instances: int,
    context_len: int,
    val_frac: float,
    batch_size: int,
    device,
    cusum_threshold: float = 5.0,
    cusum_drift: float = 0.5,
) -> pd.DataFrame:
    """
    Three-way detection latency comparison across all datasets and instances.
    Returns DataFrame with one row per switch event.
    """
    rows = []
    for ds in datasets:
        for ri in range(n_instances):
            npz_path = str(Path(data_dir) / f"{ds}_r{ri}.npz")
            if not Path(npz_path).exists():
                continue

            data = np.load(npz_path)
            y_full = data["y"].astype(np.float32).flatten()
            states_full = data["states"].astype(int).flatten()
            n_total = len(y_full)
            n_train, n_val = train_val_split_indices(n_total, val_frac)
            y_val = y_full[n_train:]
            states_val = states_full[n_train:]

            if len(y_val) < 10:
                continue

            # Get transformer predictions for residual CUSUM
            model.eval()
            preds = []
            with torch.no_grad():
                for t in range(len(y_val)):
                    full = np.concatenate([y_full[:n_train], y_val[:t + 1]])
                    win = full[-context_len:]
                    if len(win) < context_len:
                        win = np.pad(win, (context_len - len(win), 0))
                    x = torch.tensor(win[None, :, None], dtype=torch.float32, device=device)
                    y_hat = float(model.predict_next(x)[0, 0].cpu())
                    preds.append(y_hat)
            preds_arr = np.array(preds, dtype=float)
            sq_err = (y_val.astype(float) - preds_arr) ** 2

            # 1. CUSUM on raw observations
            raw_delays = cusum_on_raw(y_val, states_val, cusum_threshold, cusum_drift)
            # 2. CUSUM on residuals
            res_delays = cusum_detection_delays(sq_err, states_val, cusum_threshold, cusum_drift)
            # 3. Probe detection delay — only covers second-half val switches;
            #    key by switch_position so the merge below is position-safe
            probe_delays_raw = probe_detection_delays(
                model, npz_path, context_len, val_frac, batch_size, device
            )
            probe_delay_map = {r["switch_position"]: r["delay_steps"] for r in probe_delays_raw}

            # Compute per-switch metrics once (not once per switch)
            sw_metrics = per_switch_metrics(sq_err, states_val, context_len=context_len)

            # Merge by switch_position
            switch_pts = list(np.where(np.diff(states_val) != 0)[0] + 1)
            for i, s in enumerate(switch_pts):
                row = {
                    "dataset": ds,
                    "instance": ri,
                    "switch_ordinal": i,
                    "switch_position": int(s),
                    "cusum_raw_delay":      raw_delays[i]["delay_steps"] if i < len(raw_delays) else float("nan"),
                    "cusum_residual_delay": res_delays[i]["delay_steps"] if i < len(res_delays) else float("nan"),
                    "probe_delay":          probe_delay_map.get(int(s), float("nan")),
                    "recovery_time":        float("nan"),
                }
                if i < len(sw_metrics):
                    row["recovery_time"] = sw_metrics[i]["recovery_time"]
                    row["spike_ratio"] = sw_metrics[i]["spike_ratio"]
                    row["cumulative_excess_loss"] = sw_metrics[i]["cumulative_excess_loss"]
                    for k in [1, 5, 10, 20]:
                        row[f"post_switch_mse_k{k}"] = sw_metrics[i].get(f"post_switch_mse_k{k}", float("nan"))
                rows.append(row)

        print(f"  {ds}: done")

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir",    default="generated_data")
    ap.add_argument("--train_steps", type=int, default=10_000)
    ap.add_argument("--n_instances", type=int, default=3)
    ap.add_argument("--seed",        type=int, default=0)
    ap.add_argument("--pool_path",   default=None)
    ap.add_argument("--cusum_threshold", type=float, default=5.0)
    ap.add_argument("--cusum_drift",     type=float, default=0.5)
    ap.add_argument("--latency_csv",     default="results_detection_latency.csv")
    ap.add_argument("--reset_csv",       default="results_context_reset.csv")
    ap.add_argument("--rls_csv",         default="results_rls.csv")
    args = ap.parse_args()

    device = resolve_device("cuda")
    context_len = 64
    batch_size = 128
    val_frac = 0.3
    lr = 3e-4

    print(f"Device: {device}")
    print(f"Training full-family model for {args.train_steps} steps...")

    model = build_model(context_len, 256, 4, 6, 0.1, args.seed, device)
    sampler = build_sampler(
        ar_coeff_scale=1.2, seed=args.seed,
        family_weights=FAMILY_PRESETS["full"],
        pool_path=args.pool_path,
    )
    val_loader = get_val_monitor_loader(args.data_dir, context_len, val_frac, batch_size)
    train_iid(model, sampler, val_loader, args.train_steps, batch_size, lr, device)
    print("Training done.\n")

    # ── 1. Detection latency three-way comparison ─────────────────
    print("Running detection latency analysis...")
    latency_df = run_detection_latency_suite(
        model, args.data_dir, DATASETS_B1, args.n_instances,
        context_len, val_frac, batch_size, device,
        cusum_threshold=args.cusum_threshold,
        cusum_drift=args.cusum_drift,
    )
    latency_df.to_csv(args.latency_csv, index=False)
    print(f"Saved {len(latency_df)} rows to {args.latency_csv}")

    if len(latency_df) > 0:
        print("\nMean detection delay by dataset:")
        print(
            latency_df.groupby("dataset")[
                ["cusum_raw_delay", "cusum_residual_delay", "probe_delay", "recovery_time"]
            ].mean().round(2).to_string()
        )

    # ── 2. CUSUM context reset ────────────────────────────────────
    print("\nRunning CUSUM context reset evaluation...")
    reset_rows = []
    for ds in DATASETS_B1:
        for ri in range(args.n_instances):
            npz_path = str(Path(args.data_dir) / f"{ds}_r{ri}.npz")
            if not Path(npz_path).exists():
                continue
            result = eval_series_context_reset(
                model, npz_path, context_len, val_frac, device,
                cusum_threshold=args.cusum_threshold,
                cusum_drift=args.cusum_drift,
                dataset_name=ds, instance=ri,
            )
            if result:
                per_sw = result.pop("per_switch", [])
                for sw in per_sw:
                    reset_rows.append({**result, **sw})
    if reset_rows:
        reset_df = pd.DataFrame(reset_rows)
        reset_df.to_csv(args.reset_csv, index=False)
        print(f"Saved {len(reset_df)} rows to {args.reset_csv}")

    # ── 3. RLS adaptive head ──────────────────────────────────────
    print("\nRunning RLS adaptive head evaluation...")
    rls_df = eval_suite_rls(
        model, args.data_dir, DATASETS_B1, args.n_instances,
        context_len, val_frac, device,
        forgetting_factors=[0.95, 0.99],
        use_cusum_variants=True,
        out_csv=args.rls_csv,
    )
    print(f"Saved {len(rls_df)} rows to {args.rls_csv}")

    print("\n=== Complete ===")
    print(f"  {args.latency_csv}  — detection latency three-way comparison")
    print(f"  {args.reset_csv}    — CUSUM context reset per-switch metrics")
    print(f"  {args.rls_csv}      — RLS adaptive head per-switch metrics")


if __name__ == "__main__":
    main()

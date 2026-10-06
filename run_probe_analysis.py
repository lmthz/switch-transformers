# run_probe_analysis.py
"""
Train one model (full-family, 10k steps, same config as Exp G baseline)
and run two analyses:

  1. Probe evaluation  — fits a linear probe on hidden states to measure
     whether the transformer internally encodes regime identity, independent
     of whether errors differ between regimes.
     Output: results_probe.csv

  2. Per-switch-in-context — computes spike ratio, recovery time, and
     post-switch MSE for each individual switch event, grouped by how many
     prior switches were visible in the model's 64-step context window.
     Output: results_per_switch.csv

Usage:
    python run_probe_analysis.py
    python run_probe_analysis.py --train_steps 1000   # quick smoke test
    python run_probe_analysis.py --data_dir generated_data
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from train_transformer import train_iid, resolve_device
from run_density_experiment import (
    DATASETS_B1, FAMILY_PRESETS,
    build_model, build_sampler,
    eval_suite_extended, eval_suite_per_switch,
    get_val_monitor_loader,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir",    default="generated_data")
    ap.add_argument("--train_steps", type=int, default=10_000)
    ap.add_argument("--n_instances", type=int, default=3)
    ap.add_argument("--seed",        type=int, default=0)
    ap.add_argument("--probe_csv",   default="results_probe.csv")
    ap.add_argument("--switch_csv",  default="results_per_switch.csv")
    args = ap.parse_args()

    device     = resolve_device()
    context_len = 64
    batch_size  = 128
    val_frac    = 0.3
    lr          = 3e-4

    print(f"Device: {device}")
    print(f"Training full-family model for {args.train_steps} steps...")

    model = build_model(context_len, 256, 4, 6, 0.1, args.seed, device)
    sampler = build_sampler(
        ar_coeff_scale=1.2, seed=args.seed,
        family_weights=FAMILY_PRESETS["full"],
    )
    val_loader = get_val_monitor_loader(
        args.data_dir, context_len, val_frac, batch_size,
    )
    train_iid(model, sampler, val_loader, args.train_steps, batch_size, lr, device)
    print("Training done.")

    # ── 1. Probe evaluation ───────────────────────────────────────
    print("\nRunning probe evaluation (use_probe=True)...")
    probe_results = eval_suite_extended(
        model, args.data_dir, DATASETS_B1, args.n_instances,
        context_len, val_frac, batch_size, device,
        use_probe=True,
    )

    # Build comparison table: error-proxy vs probe accuracy per dataset
    rows = []
    for ds in DATASETS_B1:
        if ds not in probe_results:
            continue
        rows.append({
            "dataset":         ds,
            "rmse":            probe_results.get(ds,                       float("nan")),
            "regime_acc_near": probe_results.get(f"{ds}_regime_acc_near",  float("nan")),
            "regime_acc_ss":   probe_results.get(f"{ds}_regime_acc_steady",float("nan")),
            "probe_acc_near":  probe_results.get(f"{ds}_probe_acc_near",   float("nan")),
            "probe_acc_ss":    probe_results.get(f"{ds}_probe_acc_steady", float("nan")),
        })

    probe_df = pd.DataFrame(rows)
    probe_df.to_csv(args.probe_csv, index=False)
    print(f"\nProbe results saved to {args.probe_csv}")
    print(probe_df.to_string(index=False))

    # ── 2. Per-switch-in-context ──────────────────────────────────
    print("\nRunning per-switch-in-context analysis...")
    switch_df = eval_suite_per_switch(
        model, args.data_dir, DATASETS_B1, args.n_instances,
        context_len, val_frac, batch_size, device,
        out_csv=args.switch_csv,
    )

    if len(switch_df) > 0:
        print("\nMean metrics by switches_in_context:")
        print(
            switch_df.groupby("switches_in_context")[
                ["spike_ratio", "recovery_time", "post_switch_mse"]
            ].agg(["mean", "count"]).to_string()
        )


if __name__ == "__main__":
    main()

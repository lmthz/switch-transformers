# run_msar_all.py

"""
Run MSAR baseline on all evaluation datasets and save results to CSV.

Strategy:
- For each dataset, run full order selection on instance r0
- Reuse the selected order for instances r1..r(n-1) — no re-selection
- Average val_rmse across all instances, report mean and std
- This makes running on 30 instances feasible without 30x the compute

Resume behaviour (default):
- Instance-level checkpoint (msar_instance_checkpoint.csv) is written after
  every single instance, so a wall-time kill mid-dataset loses only the
  current instance.
- Dataset-level results (msar_results.csv) are written once all instances
  for a dataset complete, aggregated from the instance checkpoint.
- Pass --fresh to ignore existing results and rerun everything.

Usage:
  python run_msar_all.py                  # resume from existing CSV if present
  python run_msar_all.py --fresh          # ignore existing results, rerun all
  python run_msar_all.py --n_instances 3
  python run_msar_all.py --out my_msar.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd

from baselines.prediction_msar import run_msar, CONFIGS, ARMA_ARIMA_DATASETS
from baselines.prediction_msar import evaluate_msar_fixed_order
from dataclasses import replace

DATASETS: List[str] = [
    "A1_ar2_coeffs_easy", "A2_ar2_coeffs_hard", "A3_ar2_coeffs_plus_var",
    "B1_ar2_variance", "B2_ar2_variance_big", "C1_arma21_coeffs_var",
    "D1_arima211", "D2_arima221", "D3_arima210",
    "E1_drift_only", "E2_level_shift", "F1_seasonal_sarimax",
    "F2_seasonal_exog", "G1_exogenous_only",
    "H3_ar4_coeffs", "H4_ar6_coeffs", "H1_ar10_coeffs",
    "H2_ar1_near_unit_root", "S1_sparse_switching", "S2_frequent_switching",
    "NS0_A1_no_switch_regime0", "NS1_A1_no_switch_regime1", "SW1_A1_single_switch",
]


def run_msar_fixed_order(
    dataset_name: str,
    file_name: str,
    order: int,
    data_dir: Path,
    val_frac: float,
    maxiter: int,
    em_iter: int,
    n_restarts: int = 5,
) -> Dict[str, Any]:
    """Run MSAR with a pre-selected order — no order search."""
    if dataset_name not in CONFIGS:
        raise KeyError(f"unknown dataset {dataset_name}")
    base_cfg = CONFIGS[dataset_name]
    cfg = replace(base_cfg, order=int(order))
    out = evaluate_msar_fixed_order(
        dataset_name=file_name,
        data_dir=data_dir,
        cfg=cfg,
        val_frac=val_frac,
        maxiter=maxiter,
        em_iter=em_iter,
        n_restarts=n_restarts,
    )
    out["selected_order"] = int(order)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="generated_data")
    ap.add_argument("--n_instances", type=int, default=30,
                    help="Number of instances per dataset (default 30). "
                         "Order selected on r0, reused for r1..r(n-1).")
    ap.add_argument("--val_frac", type=float, default=0.3)
    ap.add_argument("--n_restarts", type=int, default=5,
                    help="Random restarts per fit to handle convergence failures (default 5).")
    ap.add_argument("--out", type=str, default="msar_results.csv")
    ap.add_argument("--instance_checkpoint", type=str,
                    default="msar_instance_checkpoint.csv",
                    help="Per-instance checkpoint file (written after every instance).")
    ap.add_argument("--fresh", action="store_true",
                    help="Ignore any existing results and rerun all datasets from scratch.")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_path = Path(args.out)
    inst_path = Path(args.instance_checkpoint)

    candidate_orders = [2, 3, 4, 5, 6, 8, 10]
    maxiter = 150
    em_iter = 10

    # Extra restarts for datasets with frequent convergence failures
    HARD_DATASETS = {"F1_seasonal_sarimax", "F2_seasonal_exog",
                     "D1_arima211", "D2_arima221", "D3_arima210"}
    # H1 AR(10) and ARMA are slow to fit — use fewer restarts to stay within wall time
    FAST_DATASETS = {"H1_ar10_coeffs", "C1_arma21_coeffs_var"}

    # ── Resume: load instance-level checkpoint ────────────────────────
    # inst_done[(dataset, ri)] = {"val_rmse": ..., "train_rmse": ...}
    inst_done: dict[tuple[str, int], Dict[str, Any]] = {}
    inst_rows: List[Dict[str, Any]] = []
    if inst_path.exists() and not args.fresh:
        inst_df = pd.read_csv(inst_path)
        for _, row in inst_df.iterrows():
            inst_done[(row["dataset"], int(row["instance"]))] = row.to_dict()
        inst_rows = inst_df.to_dict("records")

    # ── Resume: load already-completed datasets ───────────────────────
    completed: set[str] = set()
    existing_rows: List[Dict[str, Any]] = []
    if out_path.exists() and not args.fresh:
        existing_df = pd.read_csv(out_path)
        completed = set(existing_df["dataset"].tolist())
        existing_rows = existing_df.to_dict("records")
        if completed:
            print(f"Resuming — {len(completed)}/{len(DATASETS)} datasets already done: "
                  f"{', '.join(sorted(completed))}\n")
    elif args.fresh and out_path.exists():
        print(f"--fresh: ignoring existing {out_path}\n")

    # ── Sanity check r0 files exist ───────────────────────────────────
    missing = [ds for ds in DATASETS
               if not (data_dir / f"{ds}_r0.npz").exists()]
    if missing:
        print(f"[error] {len(missing)} _r0 files missing. "
              f"Run: python data_generation.py")
        for m in missing[:5]:
            print(f"  generated_data/{m}_r0.npz")
        return

    remaining = [ds for ds in DATASETS if ds not in completed]
    print(f"Running MSAR on {len(remaining)} datasets "
          f"({'all' if not completed else f'{len(completed)} skipped'}) "
          f"x {args.n_instances} instances")
    print(f"Order selection on r0 only, reused for r1..r{args.n_instances-1}")
    print(f"Results will be saved to: {out_path}\n")

    new_rows: List[Dict[str, Any]] = []

    for i, ds in enumerate(DATASETS):
        # ── Skip already-completed datasets ──────────────────────────
        if ds in completed:
            print(f"[{i+1:2d}/{len(DATASETS)}] {ds}  [skip]")
            continue

        print(f"[{i+1:2d}/{len(DATASETS)}] {ds}", flush=True)

        if ds in HARD_DATASETS:
            n_restarts = args.n_restarts * 2   # 10 restarts
        elif ds in FAST_DATASETS:
            n_restarts = 1                      # 1 restart — slow datasets
        else:
            n_restarts = args.n_restarts        # 5 restarts (default)

        # ── Step 1: order selection on r0 (run if not already done) ──
        r0_result = None
        selected_order = None

        if (ds, 0) in inst_done:
            cached = inst_done[(ds, 0)]
            selected_order = int(cached["selected_order"])
            r0_result = cached
            print(f"  r0: [cached] val={cached['val_rmse']:.4f}  order={selected_order}")
        else:
            ds_r0 = f"{ds}_r0"
            try:
                r0_result = run_msar(
                    ds,
                    data_dir=data_dir,
                    val_frac=args.val_frac,
                    candidate_orders=candidate_orders,
                    maxiter=maxiter,
                    em_iter=em_iter,
                    file_name=ds_r0,
                    n_restarts=n_restarts,
                )
                selected_order = r0_result.get("selected_order", r0_result["order"])
                print(f"  r0: val={r0_result['val_rmse']:.4f}  "
                      f"order={selected_order}  "
                      f"regime_acc={r0_result['regime_accuracy']:.3f}")
                inst_row = {
                    "dataset": ds, "instance": 0,
                    "val_rmse": r0_result["val_rmse"],
                    "train_rmse": r0_result["train_rmse"],
                    "selected_order": selected_order,
                    "regime_accuracy": r0_result["regime_accuracy"],
                    "noise_rmse": r0_result["noise_rmse"],
                    "oracle_model_rmse": r0_result["oracle_model_rmse"],
                }
                inst_rows.append(inst_row)
                inst_done[(ds, 0)] = inst_row
                pd.DataFrame(inst_rows).to_csv(inst_path, index=False)
            except Exception as e:
                print(f"  r0: [FAILED] {e}")

        if selected_order is None:
            print(f"  skipping r1+ (no order selected for {ds})")
            continue

        # ── Step 2: fixed order on r1..r(n-1) ────────────────────────
        for ri in range(1, args.n_instances):
            if (ds, ri) in inst_done:
                print(f"  r{ri}: [cached]")
                continue
            ds_ri = f"{ds}_r{ri}"
            npz = data_dir / f"{ds_ri}.npz"
            if not npz.exists():
                continue
            try:
                ri_result = run_msar_fixed_order(
                    dataset_name=ds,
                    file_name=ds_ri,
                    order=selected_order,
                    data_dir=data_dir,
                    val_frac=args.val_frac,
                    maxiter=maxiter,
                    em_iter=em_iter,
                    n_restarts=n_restarts,
                )
                inst_row = {
                    "dataset": ds, "instance": ri,
                    "val_rmse": ri_result["val_rmse"],
                    "train_rmse": ri_result["train_rmse"],
                    "selected_order": selected_order,
                    "regime_accuracy": float("nan"),
                    "noise_rmse": float("nan"),
                    "oracle_model_rmse": float("nan"),
                }
                inst_rows.append(inst_row)
                inst_done[(ds, ri)] = inst_row
                pd.DataFrame(inst_rows).to_csv(inst_path, index=False)
            except Exception as e:
                print(f"  r{ri}: [FAILED] {e}")

        # ── Aggregate from instance checkpoint ────────────────────────
        ds_instances = [inst_done[(ds, ri)] for ri in range(args.n_instances)
                        if (ds, ri) in inst_done]
        all_val   = [r["val_rmse"]   for r in ds_instances if np.isfinite(r["val_rmse"])]
        all_train = [r["train_rmse"] for r in ds_instances if np.isfinite(r["train_rmse"])]
        n_ok  = len(all_val)
        n_ran = len(ds_instances)

        val_mean = float(np.mean(all_val))   if all_val   else float("nan")
        val_std  = float(np.std(all_val))    if all_val   else float("nan")
        tr_mean  = float(np.mean(all_train)) if all_train else float("nan")

        if n_ran > 0 and n_ok < n_ran:
            print(f"  note: {n_ran - n_ok}/{n_ran} instances returned NaN (filtered out)")
        print(f"  mean val_rmse={val_mean:.4f} ± {val_std:.4f}  ({n_ok}/{args.n_instances} valid)")

        new_rows.append({
            "dataset":           ds,
            "msar_order":        selected_order,
            "msar_train_rmse":   tr_mean,
            "msar_val_rmse":     val_mean,
            "msar_val_rmse_std": val_std,
            "msar_n_instances":  n_ok,
            "msar_regime_acc":   r0_result["regime_accuracy"] if r0_result else float("nan"),
            "noise_rmse":        r0_result["noise_rmse"] if r0_result else float("nan"),
            "oracle_model_rmse": r0_result["oracle_model_rmse"] if r0_result else float("nan"),
        })

        # ── Write dataset-level CSV once all instances are done ───────
        all_rows = existing_rows + new_rows
        pd.DataFrame(all_rows).to_csv(out_path, index=False)

    # ── Final summary ─────────────────────────────────────────────────
    final_df = pd.read_csv(out_path)   # read back so summary includes skipped rows too
    print(f"\nDone. Results saved to {out_path}")
    print(final_df[["dataset", "msar_val_rmse", "msar_val_rmse_std",
                     "msar_n_instances", "noise_rmse"]].to_string(index=False))


if __name__ == "__main__":
    main()
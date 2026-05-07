# run_msar_all.py
"""
Run MSAR baseline on all evaluation datasets and save results to CSV.

Strategy:
  - For each dataset, run full order selection on instance r0
  - Reuse the selected order for instances r1..r(n-1) — no re-selection
  - Average val_rmse across all instances, report mean and std
  - This makes running on 30 instances feasible without 30x the compute

Usage:
    python run_msar_all.py                          # default: n_instances=30
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
    "F2_seasonal_exog", "G1_exogenous_only", "H1_ar10_coeffs",
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
    ap.add_argument("--data_dir",    type=str,   default="generated_data")
    ap.add_argument("--n_instances", type=int,   default=30,
                    help="Number of instances per dataset (default 30). "
                         "Order selected on r0, reused for r1..r(n-1).")
    ap.add_argument("--val_frac",    type=float, default=0.3)
    ap.add_argument("--n_restarts",  type=int,   default=5,
                    help="Random restarts per fit to handle convergence failures (default 5).")
    ap.add_argument("--out",         type=str,   default="msar_results.csv")
    args = ap.parse_args()

    data_dir         = Path(args.data_dir)
    candidate_orders = [2, 3, 4, 5, 6, 8, 10]
    maxiter          = 150
    em_iter          = 10

    # Datasets known to have convergence problems get more restarts
    HARD_DATASETS = {"F1_seasonal_sarimax", "F2_seasonal_exog",
                     "D1_arima211", "D2_arima221", "D3_arima210", "H1_ar10_coeffs"}

    # Check r0 files exist
    missing = [ds for ds in DATASETS
               if not (data_dir / f"{ds}_r0.npz").exists()]
    if missing:
        print(f"[error] {len(missing)} _r0 files missing. "
              f"Run: python data_generation.py")
        for m in missing[:5]:
            print(f"  generated_data/{m}_r0.npz")
        return

    print(f"Running MSAR on {len(DATASETS)} datasets x {args.n_instances} instances")
    print(f"Order selection on r0 only, reused for r1..r{args.n_instances-1}")
    print(f"Results will be saved to: {args.out}\n")

    rows: List[Dict[str, Any]] = []

    for i, ds in enumerate(DATASETS):
        print(f"[{i+1:2d}/{len(DATASETS)}] {ds}", flush=True)

        # ── Step 1: full order selection on r0 ───────────────────
        ds_r0 = f"{ds}_r0"
        r0_result = None
        selected_order = None
        try:
            n_restarts = args.n_restarts * 2 if ds in HARD_DATASETS else args.n_restarts
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
        except Exception as e:
            print(f"  r0: [FAILED] {e}")

        # ── Step 2: fixed order on r1..r(n-1) ────────────────────
        all_val_rmse = []
        all_train_rmse = []

        if r0_result is not None:
            all_val_rmse.append(r0_result["val_rmse"])
            all_train_rmse.append(r0_result["train_rmse"])

        for ri in range(1, args.n_instances):
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
                all_val_rmse.append(ri_result["val_rmse"])
                all_train_rmse.append(ri_result["train_rmse"])
            except Exception as e:
                print(f"  r{ri}: [FAILED] {e}")

        # Filter out NaN values — some instances may return NaN val_rmse
        # when the val split has no finite predictions (common for ARIMA/seasonal)
        val_finite   = [v for v in all_val_rmse   if np.isfinite(v)]
        train_finite = [v for v in all_train_rmse if np.isfinite(v)]
        n_ok     = len(val_finite)
        n_ran    = len(all_val_rmse)
        val_mean = float(np.mean(val_finite))   if val_finite   else float("nan")
        val_std  = float(np.std(val_finite))    if val_finite   else float("nan")
        tr_mean  = float(np.mean(train_finite)) if train_finite else float("nan")
        if n_ran > 0 and n_ok < n_ran:
            print(f"  note: {n_ran - n_ok}/{n_ran} instances returned NaN val_rmse (filtered out)")
        print(f"  mean val_rmse={val_mean:.4f} ± {val_std:.4f}  ({n_ok}/{args.n_instances} valid instances)")

        rows.append({
            "dataset":           ds,
            "msar_order":        selected_order if selected_order else float("nan"),
            "msar_train_rmse":   tr_mean,
            "msar_val_rmse":     val_mean,
            "msar_val_rmse_std": val_std,
            "msar_n_instances":  n_ok,
            "msar_regime_acc":   r0_result["regime_accuracy"]   if r0_result else float("nan"),
            "noise_rmse":        r0_result["noise_rmse"]        if r0_result else float("nan"),
            "oracle_model_rmse": r0_result["oracle_model_rmse"] if r0_result else float("nan"),
        })

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\nDone. Results saved to {args.out}")
    print(df[["dataset", "msar_val_rmse", "msar_val_rmse_std",
              "msar_n_instances", "noise_rmse"]].to_string(index=False))


if __name__ == "__main__":
    main()
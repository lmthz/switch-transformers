# run_msar_all.py
"""
Run MSAR baseline on all evaluation datasets and save results to CSV.
Run this once after data_generation.py — it is slow (~2-3 hours) because
statsmodels EM fitting is expensive. Results are cached in msar_results.csv
so run_compare.py can load them without re-running MSAR.

Usage:
    python run_msar_all.py                          # default: n_instances=3
    python run_msar_all.py --n_instances 1          # single instance
    python run_msar_all.py --out my_msar.csv        # custom output path
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd

from baselines.prediction_msar import run_msar


DATASETS: List[str] = [
    "A1_ar2_coeffs_easy",
    "A2_ar2_coeffs_hard",
    "A3_ar2_coeffs_plus_var",
    "B1_ar2_variance",
    "B2_ar2_variance_big",
    "C1_arma21_coeffs_var",
    "D1_arima211",
    "D2_arima221",
    "D3_arima210",
    "E1_drift_only",
    "E2_level_shift",
    "F1_seasonal_sarimax",
    "F2_seasonal_exog",
    "G1_exogenous_only",
    "H1_ar10_coeffs",
    "H2_ar1_near_unit_root",
    "S1_sparse_switching",
    "S2_frequent_switching",
    "NS0_A1_no_switch_regime0",
    "NS1_A1_no_switch_regime1",
    "SW1_A1_single_switch",
]


def main():
    ap = argparse.ArgumentParser(description="Run MSAR on all datasets and save results.")
    ap.add_argument("--data_dir",    type=str, default="generated_data")
    ap.add_argument("--n_instances", type=int, default=3,
                    help="Number of instances per dataset type (default 3). "
                         "MSAR is run on instance r0 only — this flag just "
                         "validates that r0 files exist.")
    ap.add_argument("--val_frac",    type=float, default=0.3)
    ap.add_argument("--out",         type=str, default="msar_results.csv")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    candidate_orders = [2, 3, 4, 5, 6, 8, 10]
    maxiter  = 150
    em_iter  = 10

    # Verify r0 files exist
    missing = [ds for ds in DATASETS if not (data_dir / f"{ds}_r0.npz").exists()]
    if missing:
        print(f"[error] {len(missing)} _r0 files missing. Run: python data_generation.py")
        for m in missing[:5]:
            print(f"  generated_data/{m}_r0.npz")
        return

    print(f"Running MSAR on {len(DATASETS)} datasets (r0 instance only)")
    print(f"Results will be saved to: {args.out}\n")

    rows: List[Dict[str, Any]] = []

    for i, ds in enumerate(DATASETS):
        ds_r0 = f"{ds}_r0"
        print(f"[{i+1:2d}/{len(DATASETS)}] {ds_r0} ...", flush=True)

        msar = None
        try:
            msar = run_msar(
                ds,                   # base name for CONFIGS lookup
                data_dir=data_dir,
                val_frac=args.val_frac,
                candidate_orders=candidate_orders,
                maxiter=maxiter,
                em_iter=em_iter,
                file_name=ds_r0,      # actual file to load (_r0 suffix)
            )
            print(
                f"         train={msar['train_rmse']:.4f}  val={msar['val_rmse']:.4f}  "
                f"regime_acc={msar['regime_accuracy']:.3f}  noise={msar['noise_rmse']:.4f}  "
                f"order={msar.get('selected_order', msar['order'])}"
            )
        except Exception as e:
            print(f"         [FAILED] {e}")

        rows.append({
            "dataset":           ds,
            "msar_order":        msar.get("selected_order", msar["order"]) if msar else float("nan"),
            "msar_train_rmse":   msar["train_rmse"]        if msar else float("nan"),
            "msar_val_rmse":     msar["val_rmse"]          if msar else float("nan"),
            "msar_regime_acc":   msar["regime_accuracy"]   if msar else float("nan"),
            "noise_rmse":        msar["noise_rmse"]        if msar else float("nan"),
            "oracle_model_rmse": msar["oracle_model_rmse"] if msar else float("nan"),
        })

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\nDone. Results saved to {args.out}")
    print(df[["dataset", "msar_val_rmse", "noise_rmse"]].to_string(index=False))


if __name__ == "__main__":
    main()
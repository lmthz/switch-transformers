# run_compare.py
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any

import pandas as pd

from baselines.prediction_msar import run_msar
from train_transformer import train_one_dataset


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
    data_dir = "generated_data"
    val_frac = 0.3

    # msar hyperparams
    candidate_orders = [2, 3, 4, 5, 6, 8, 10]
    maxiter = 150
    em_iter = 10

    # transformer hyperparams
    context_len = 64
    steps = 1500
    batch_size = 128
    lr = 3e-4
    d_model = 64
    n_heads = 2
    n_layers = 4
    dropout = 0.2
    seed = 0
    device = "cuda"

    rows: List[Dict[str, Any]] = []

    for ds in DATASETS:
        npz_path = Path(data_dir) / f"{ds}.npz"
        if not npz_path.exists():
            print(f"[missing] {npz_path} does not exist. run: python data_generation.py")
            return

        print(f"\n=== {ds} ===")

        try:
            msar = run_msar(
                ds,
                data_dir=data_dir,
                val_frac=val_frac,
                candidate_orders=candidate_orders,
                maxiter=maxiter,
                em_iter=em_iter,
            )
        except Exception as e:
            print(f"[msar] all orders failed for {ds}, skipping: {e}")
            continue
        print(
            f"msar: train_rmse={msar['train_rmse']:.4f} val_rmse={msar['val_rmse']:.4f} "
            f"regime_acc(train)={msar['regime_accuracy']:.3f} noise_rmse={msar['noise_rmse']:.4f} "
            f"order={msar.get('selected_order', msar['order'])}"
        )

        tr = train_one_dataset(
            npz_path=str(npz_path),
            context_len=context_len,
            val_frac=val_frac,
            steps=steps,
            batch_size=batch_size,
            lr=lr,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            seed=seed,
            device=device,
        )
        print(f"transformer: train_rmse={tr['train_rmse']:.4f} val_rmse={tr['val_rmse']:.4f}")

        rows.append(
            {
                "dataset": ds,
                "msar_order": msar.get("selected_order", msar["order"]),
                "msar_train_rmse": msar["train_rmse"],
                "msar_val_rmse": msar["val_rmse"],
                "msar_regime_acc_train": msar["regime_accuracy"],
                "noise_rmse": msar["noise_rmse"],
                "tr_train_rmse": tr["train_rmse"],
                "tr_val_rmse": tr["val_rmse"],
            }
        )

    df = pd.DataFrame(rows)
    print("\nsummary table")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
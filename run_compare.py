# run_compare.py
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any

import pandas as pd

from baselines.prediction_msar import run_msar
from train_transformer import train_one_dataset, train_iid, eval_loop, resolve_device
from train_transformer import MSARBatchSampler, MSARSamplerConfig
from models.transformer_forecaster import TransformerConfig, CausalTransformerForecaster
from data.synthetic_npz_dataset import make_train_val_datasets
from torch.utils.data import DataLoader
import numpy as np
import torch


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


def eval_transformer_on_dataset(
    model: CausalTransformerForecaster,
    npz_path: str,
    context_len: int,
    val_frac: float,
    batch_size: int,
    device: torch.device,
) -> Dict[str, Any]:
    """Evaluate a pre-trained model on one dataset via forward passes only."""
    ds_train, ds_val, stdzr, _ = make_train_val_datasets(
        npz_path=npz_path,
        context_len=context_len,
        val_frac=val_frac,
    )
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=False)
    val_loader   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False)

    _, train_rmse = eval_loop(model, train_loader, device)
    _, val_rmse   = eval_loop(model, val_loader,   device)

    return {
        "train_rmse": float(train_rmse),
        "val_rmse":   float(val_rmse),
        "n_train":    len(ds_train),
        "n_val":      len(ds_val),
    }


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Compare MSAR baseline vs decoder transformer.")
    ap.add_argument(
        "--pool_path", type=str, default=None,
        help="Path to pre-generated series pool .npz from generate_pool.py.",
    )
    ap.add_argument(
        "--ar_coeff_scale", type=float, default=0.6,
        help="AR coefficient scale used during training (informational only when using pool).",
    )
    ap.add_argument(
        "--steps", type=int, default=22000,
        help="Number of iid training steps (default: 22000).",
    )
    ap.add_argument(
        "--experiment_name", type=str, default=None,
        help="Label for this run. Results saved to results_<name>.csv.",
    )
    args = ap.parse_args()

    data_dir = "generated_data"
    val_frac  = 0.3

    # msar hyperparams
    candidate_orders = [2, 3, 4, 5, 6, 8, 10]
    maxiter  = 150
    em_iter  = 10

    # transformer hyperparams
    context_len = 64
    steps       = args.steps
    batch_size  = 128
    lr          = 3e-4
    d_model     = 256
    n_heads     = 4
    n_layers    = 6
    dropout     = 0.1
    seed        = 0
    device_str  = "cuda"
    training_mode = "iid"

    print(f"\n{'='*60}")
    print(f"Experiment: {args.experiment_name or 'unnamed'}")
    print(f"  steps={steps}  ar_coeff_scale={args.ar_coeff_scale}")
    print(f"  pool_path={args.pool_path}")
    print(f"  architecture: decoder-only (dense next-step supervision)")
    print(f"{'='*60}\n")

    device = resolve_device(device_str)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # --------------------------------------------------------
    # Train transformer once (iid mode)
    # --------------------------------------------------------
    if training_mode == "iid":
        print("=== Training transformer (iid, decoder-only) ===")
        cfg = TransformerConfig(
            context_len=context_len,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )
        model = CausalTransformerForecaster(cfg).to(device)
        model.train()

        sampler_cfg = MSARSamplerConfig(
            series_len=max(512, context_len * 4),
            k_regimes=2,
            ar_coeff_scale=args.ar_coeff_scale,
            ma_coeff_scale=0.4,
            sar_coeff_scale=0.35,
            sigma_lo=0.15,
            sigma_hi=0.70,
            persistence_lo=0.85,
            persistence_hi=0.98,
            burn_in=100,
            mix_ar=0.22,
            mix_ar_near_unit=0.05,
            mix_ar_no_switch=0.05,
            mix_arma=0.13,
            mix_arima1=0.13,
            mix_arima2=0.07,
            mix_seasonal=0.12,
            mix_exog_const=0.08,
            mix_exog_sine=0.08,
            mix_exog_seasonal=0.07,
        )
        sampler = MSARBatchSampler(sampler_cfg, seed=seed)

        if args.pool_path is not None:
            sampler.load_pool(args.pool_path)

        first_npz = str(Path(data_dir) / f"{DATASETS[0]}.npz")
        _, ds_val_monitor, _, _ = make_train_val_datasets(first_npz, context_len, val_frac)
        val_loader_monitor = DataLoader(ds_val_monitor, batch_size=batch_size, shuffle=False)

        train_iid(model, sampler, val_loader_monitor, steps, batch_size, lr, device)
        model.eval()
        print("=== Transformer training complete ===\n")

    rows: List[Dict[str, Any]] = []

    for ds in DATASETS:
        npz_path = Path(data_dir) / f"{ds}.npz"
        if not npz_path.exists():
            print(f"[missing] {npz_path} — run: python data_generation.py")
            return

        print(f"\n=== {ds} ===")

        msar = None
        try:
            msar = run_msar(
                ds,
                data_dir=data_dir,
                val_frac=val_frac,
                candidate_orders=candidate_orders,
                maxiter=maxiter,
                em_iter=em_iter,
            )
            print(
                f"msar: train_rmse={msar['train_rmse']:.4f} val_rmse={msar['val_rmse']:.4f} "
                f"regime_acc(train)={msar['regime_accuracy']:.3f} noise_rmse={msar['noise_rmse']:.4f} "
                f"oracle_model_rmse={msar['oracle_model_rmse']:.4f} "
                f"order={msar.get('selected_order', msar['order'])}"
            )
        except Exception as e:
            print(f"[msar] all orders failed for {ds}: {e}")

        if training_mode == "iid":
            tr = eval_transformer_on_dataset(
                model=model,
                npz_path=str(npz_path),
                context_len=context_len,
                val_frac=val_frac,
                batch_size=batch_size,
                device=device,
            )
        else:
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
                device=device_str,
                training_mode="fixed",
                ar_coeff_scale=args.ar_coeff_scale,
            )

        print(f"transformer: train_rmse={tr['train_rmse']:.4f} val_rmse={tr['val_rmse']:.4f}")

        rows.append({
            "dataset":               ds,
            "msar_order":            msar.get("selected_order", msar["order"]) if msar else float("nan"),
            "msar_train_rmse":       msar["train_rmse"]        if msar else float("nan"),
            "msar_val_rmse":         msar["val_rmse"]          if msar else float("nan"),
            "msar_regime_acc_train": msar["regime_accuracy"]   if msar else float("nan"),
            "noise_rmse":            msar["noise_rmse"]        if msar else float("nan"),
            "oracle_model_rmse":     msar["oracle_model_rmse"] if msar else float("nan"),
            "tr_train_rmse":         tr["train_rmse"],
            "tr_val_rmse":           tr["val_rmse"],
        })

    df = pd.DataFrame(rows)
    print("\nsummary table")
    print(df.to_string(index=False))

    if args.experiment_name:
        out_path = f"results_{args.experiment_name}.csv"
        df.to_csv(out_path, index=False)
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
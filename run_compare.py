# run_compare.py
"""
Train transformer and evaluate on all datasets.
Loads pre-computed MSAR results from msar_results.csv (produced by run_msar_all.py).
Does not run MSAR itself — run run_msar_all.py separately first.

Usage:
    python run_compare.py --pool_path series_pool.npz --steps 100000
    python run_compare.py --msar_csv msar_results.csv --experiment_name my_run
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from train_transformer import train_iid, eval_loop, resolve_device
from train_transformer import MSARBatchSampler, MSARSamplerConfig
from models.transformer_forecaster import TransformerConfig, CausalTransformerForecaster
from data.synthetic_npz_dataset import make_train_val_datasets


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
    return {"train_rmse": float(train_rmse), "val_rmse": float(val_rmse)}


def main():
    ap = argparse.ArgumentParser(
        description="Train transformer and compare to pre-computed MSAR results."
    )
    ap.add_argument("--pool_path",       type=str,   default=None)
    ap.add_argument("--ar_coeff_scale",  type=float, default=0.6)
    ap.add_argument("--steps",           type=int,   default=100000)
    ap.add_argument("--experiment_name", type=str,   default=None)
    ap.add_argument("--n_instances",     type=int,   default=3,
                    help="Number of dataset instances per type (default 3).")
    ap.add_argument(
        "--msar_csv", type=str, default="msar_results.csv",
        help="Path to pre-computed MSAR results CSV from run_msar_all.py "
             "(default: msar_results.csv)."
    )
    args = ap.parse_args()

    data_dir    = "generated_data"
    val_frac    = 0.3
    context_len = 64
    batch_size  = 128
    lr          = 3e-4
    d_model     = 256
    n_heads     = 4
    n_layers    = 6
    dropout     = 0.1
    seed        = 0
    device_str  = "cuda"

    # ── Load pre-computed MSAR results ────────────────────────────
    msar_csv = Path(args.msar_csv)
    if not msar_csv.exists():
        print(f"[error] MSAR results not found at {msar_csv}")
        print(f"        Run first: python run_msar_all.py")
        return
    msar_df = pd.read_csv(msar_csv).set_index("dataset")
    print(f"Loaded MSAR results from {msar_csv} ({len(msar_df)} datasets)\n")

    # ── Verify evaluation dataset files exist ─────────────────────
    suffixes = [f"_r{i}" for i in range(args.n_instances)]
    missing  = [
        f"{ds}{suf}"
        for ds in DATASETS
        for suf in suffixes
        if not (Path(data_dir) / f"{ds}{suf}.npz").exists()
    ]
    if missing:
        print(f"[error] {len(missing)} dataset files missing. Run: python data_generation.py")
        for m in missing[:5]:
            print(f"  generated_data/{m}.npz")
        return

    print(f"\n{'='*60}")
    print(f"Experiment: {args.experiment_name or 'unnamed'}")
    print(f"  steps={args.steps}  ar_coeff_scale={args.ar_coeff_scale}")
    print(f"  pool_path={args.pool_path}")
    print(f"  n_instances={args.n_instances}")
    print(f"  msar_csv={args.msar_csv}")
    print(f"  architecture: decoder-only (dense next-step supervision)")
    print(f"{'='*60}\n")

    device = resolve_device(device_str)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ── Train transformer ─────────────────────────────────────────
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

    first_npz = str(Path(data_dir) / f"{DATASETS[0]}_r0.npz")
    _, ds_val_monitor, _, _ = make_train_val_datasets(first_npz, context_len, val_frac)
    val_loader_monitor = DataLoader(ds_val_monitor, batch_size=batch_size, shuffle=False)

    train_iid(model, sampler, val_loader_monitor, args.steps, batch_size, lr, device)
    model.eval()
    print("=== Transformer training complete ===\n")

    # ── Evaluate transformer on all instances ─────────────────────
    rows: List[Dict[str, Any]] = []

    for ds in DATASETS:
        print(f"\n=== {ds} ===")

        # Pull MSAR row
        if ds in msar_df.index:
            m = msar_df.loc[ds]
            print(
                f"msar (r0): val={m['msar_val_rmse']:.4f}  "
                f"regime_acc={m['msar_regime_acc']:.3f}  "
                f"noise={m['noise_rmse']:.4f}  "
                f"order={m['msar_order']}"
            )
        else:
            m = None
            print(f"[msar] no results found for {ds} in {args.msar_csv}")

        # Transformer on all n_instances
        tr_vals, tr_trains = [], []
        for i in range(args.n_instances):
            npz_path = str(Path(data_dir) / f"{ds}_r{i}.npz")
            tr_i = eval_transformer_on_dataset(
                model=model,
                npz_path=npz_path,
                context_len=context_len,
                val_frac=val_frac,
                batch_size=batch_size,
                device=device,
            )
            tr_vals.append(tr_i["val_rmse"])
            tr_trains.append(tr_i["train_rmse"])

        tr_val_mean  = float(np.mean(tr_vals))
        tr_val_std   = float(np.std(tr_vals))
        tr_train_mean = float(np.mean(tr_trains))

        print(
            f"transformer ({args.n_instances} instances): "
            f"val_rmse={tr_val_mean:.4f} ± {tr_val_std:.4f}  "
            f"[{', '.join(f'{v:.4f}' for v in tr_vals)}]"
        )

        rows.append({
            "dataset":            ds,
            "msar_order":         m["msar_order"]        if m is not None else float("nan"),
            "msar_val_rmse":      m["msar_val_rmse"]     if m is not None else float("nan"),
            "msar_regime_acc":    m["msar_regime_acc"]   if m is not None else float("nan"),
            "noise_rmse":         m["noise_rmse"]        if m is not None else float("nan"),
            "oracle_model_rmse":  m["oracle_model_rmse"] if m is not None else float("nan"),
            "tr_val_rmse_mean":   tr_val_mean,
            "tr_val_rmse_std":    tr_val_std,
            "tr_train_rmse_mean": tr_train_mean,
            **{f"tr_val_r{i}": tr_vals[i] for i in range(args.n_instances)},
        })

    df = pd.DataFrame(rows)
    print("\nsummary table")
    display_cols = ["dataset", "msar_val_rmse", "tr_val_rmse_mean", "tr_val_rmse_std", "noise_rmse"]
    print(df[display_cols].to_string(index=False))

    if args.experiment_name:
        out_path = f"results_{args.experiment_name}.csv"
        df.to_csv(out_path, index=False)
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
# run_compare.py
"""
Train transformer and evaluate on all datasets.
Loads pre-computed MSAR results from msar_results.csv (produced by run_msar_all.py).
Does not run MSAR itself — run run_msar_all.py separately first.

W&B logging:
  - Training loss and val RMSE curves over steps
  - Gradient norm over steps
  - Per-dataset final results (transformer vs MSAR bar charts)
  - All hyperparameters as run config

Usage:
    python run_compare.py --pool_path series_pool.npz --steps 100000
    python run_compare.py --no_wandb   # disable W&B logging
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
from torch.utils.data import ConcatDataset


# One representative per process family for monitoring validation during training.
VAL_MONITOR_DATASETS = [
    "A1_ar2_coeffs_easy",      # AR switching
    "C1_arma21_coeffs_var",    # ARMA
    "D1_arima211",             # ARIMA
    "F1_seasonal_sarimax",     # Seasonal
    "G1_exogenous_only",       # Exogenous
    "H2_ar1_near_unit_root",   # Near-unit-root
]


def get_val_monitor_loader(data_dir, context_len, val_frac, batch_size):
    """
    Build a combined validation DataLoader from one dataset per process family.
    Gives a more balanced training signal than monitoring on A1 alone.
    """
    val_sets = []
    for ds in VAL_MONITOR_DATASETS:
        npz = Path(data_dir) / f"{ds}_r0.npz"
        if npz.exists():
            _, ds_val, _, _ = make_train_val_datasets(str(npz), context_len, val_frac)
            val_sets.append(ds_val)
    if not val_sets:
        raise RuntimeError(f"No validation datasets found in {data_dir}")
    combined = ConcatDataset(val_sets)
    return DataLoader(combined, batch_size=batch_size, shuffle=False)


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
    ap.add_argument("--n_instances",     type=int,   default=3)
    ap.add_argument("--msar_csv",        type=str,   default="msar_results.csv")
    ap.add_argument(
        "--no_wandb", action="store_true",
        help="Disable Weights & Biases logging."
    )
    ap.add_argument(
        "--wandb_project", type=str, default="switch-transformers",
        help="W&B project name (default: switch-transformers)."
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

    # ── Load MSAR results ─────────────────────────────────────────
    msar_csv = Path(args.msar_csv)
    if not msar_csv.exists():
        print(f"[error] MSAR results not found at {msar_csv}")
        print(f"        Run first: python run_msar_all.py")
        return
    msar_df = pd.read_csv(msar_csv).set_index("dataset")
    print(f"Loaded MSAR results from {msar_csv} ({len(msar_df)} datasets)\n")

    # ── Verify evaluation files ───────────────────────────────────
    suffixes = [f"_r{i}" for i in range(args.n_instances)]
    missing  = [
        f"{ds}{suf}"
        for ds in DATASETS for suf in suffixes
        if not (Path(data_dir) / f"{ds}{suf}.npz").exists()
    ]
    if missing:
        print(f"[error] {len(missing)} dataset files missing. Run: python data_generation.py")
        for m in missing[:5]:
            print(f"  generated_data/{m}.npz")
        return

    # ── Initialise W&B ────────────────────────────────────────────
    wandb_run = None
    if not args.no_wandb:
        try:
            import wandb
            import os
            os.environ["WANDB_MODE"] = "offline"  # save locally, sync after
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.experiment_name,
                config={
                    # Model architecture
                    "context_len":    context_len,
                    "d_model":        d_model,
                    "n_heads":        n_heads,
                    "n_layers":       n_layers,
                    "dropout":        dropout,
                    # Training
                    "steps":          args.steps,
                    "batch_size":     batch_size,
                    "lr":             lr,
                    "ar_coeff_scale": args.ar_coeff_scale,
                    "seed":           seed,
                    "pool_path":      args.pool_path,
                    "n_instances":    args.n_instances,
                    # Architecture description
                    "architecture":   "decoder-only, dense next-step supervision",
                    "training_mode":  "iid",
                },
            )
            print(f"W&B run initialised: {wandb_run.url}\n")
        except Exception as e:
            print(f"[warning] W&B init failed ({e}). Continuing without logging.")
            wandb_run = None

    print(f"\n{'='*60}")
    print(f"Experiment: {args.experiment_name or 'unnamed'}")
    print(f"  steps={args.steps}  ar_coeff_scale={args.ar_coeff_scale}")
    print(f"  pool_path={args.pool_path}  n_instances={args.n_instances}")
    print(f"  wandb={'enabled' if wandb_run else 'disabled'}")
    print(f"{'='*60}\n")

    device = resolve_device(device_str)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ── Build model ───────────────────────────────────────────────
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

    # Log model parameter count to W&B
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    if wandb_run:
        wandb_run.config.update({"n_params": n_params})

    # ── Build sampler ─────────────────────────────────────────────
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

    val_loader_monitor = get_val_monitor_loader(data_dir, context_len, val_frac, batch_size)

    # ── Train ─────────────────────────────────────────────────────
    train_iid(
        model, sampler, val_loader_monitor,
        args.steps, batch_size, lr, device,
        wandb_run=wandb_run,
    )
    model.eval()
    print("=== Transformer training complete ===\n")

    # ── Evaluate on all datasets ──────────────────────────────────
    rows: List[Dict[str, Any]] = []

    for ds in DATASETS:
        print(f"\n=== {ds} ===")

        m = msar_df.loc[ds] if ds in msar_df.index else None
        if m is not None:
            print(
                f"msar (r0): val={m['msar_val_rmse']:.4f}  "
                f"regime_acc={m['msar_regime_acc']:.3f}  "
                f"noise={m['noise_rmse']:.4f}  "
                f"order={m['msar_order']}"
            )

        tr_vals, tr_trains = [], []
        for i in range(args.n_instances):
            npz_path = str(Path(data_dir) / f"{ds}_r{i}.npz")
            tr_i = eval_transformer_on_dataset(
                model=model, npz_path=npz_path,
                context_len=context_len, val_frac=val_frac,
                batch_size=batch_size, device=device,
            )
            tr_vals.append(tr_i["val_rmse"])
            tr_trains.append(tr_i["train_rmse"])

        tr_val_mean   = float(np.mean(tr_vals))
        tr_val_std    = float(np.std(tr_vals))
        tr_train_mean = float(np.mean(tr_trains))

        print(
            f"transformer ({args.n_instances} instances): "
            f"val_rmse={tr_val_mean:.4f} ± {tr_val_std:.4f}  "
            f"[{', '.join(f'{v:.4f}' for v in tr_vals)}]"
        )

        # Log per-dataset results to W&B
        if wandb_run is not None:
            log = {
                f"eval/{ds}/tr_val_rmse_mean": tr_val_mean,
                f"eval/{ds}/tr_val_rmse_std":  tr_val_std,
            }
            if m is not None:
                log[f"eval/{ds}/msar_val_rmse"]    = float(m["msar_val_rmse"])
                log[f"eval/{ds}/noise_rmse"]        = float(m["noise_rmse"])
                # Gap: positive means transformer is worse than MSAR
                log[f"eval/{ds}/gap_vs_msar"]       = tr_val_mean - float(m["msar_val_rmse"])
                # Ratio: <1 means transformer beats MSAR
                if float(m["msar_val_rmse"]) > 0:
                    log[f"eval/{ds}/ratio_vs_msar"] = tr_val_mean / float(m["msar_val_rmse"])
            wandb_run.log(log)

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

    # Log aggregate summary metrics to W&B
    if wandb_run is not None:
        valid = df.dropna(subset=["msar_val_rmse", "tr_val_rmse_mean"])
        wandb_run.summary["mean_tr_val_rmse"]     = float(df["tr_val_rmse_mean"].mean())
        wandb_run.summary["mean_gap_vs_msar"]     = float(
            (valid["tr_val_rmse_mean"] - valid["msar_val_rmse"]).mean()
        )
        wandb_run.summary["n_datasets_beat_msar"] = int(
            (valid["tr_val_rmse_mean"] < valid["msar_val_rmse"]).sum()
        )
        wandb_run.finish()
        print(f"\nW&B run finished: {wandb_run.url}")

    if args.experiment_name:
        out_path = f"results_{args.experiment_name}.csv"
        df.to_csv(out_path, index=False)
        print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
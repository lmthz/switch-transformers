# run_density_experiment.py
"""
Data density experiments — two axes following the ICL literature.

BACKGROUND
----------
Raventós et al. (2023): transformers need a minimum number of distinct
pretraining tasks M to generalise OOD. Below threshold M* the model behaves
like a Bayesian estimator over the training distribution. Above M* it learns
a general algorithm that works on all tasks.

Wang et al. (2024): even with sufficient task diversity, transformers fail
when the TEST function class differs from the TRAINING function class.

Yadlowsky et al. (2023): ICL capabilities are tied to coverage of pretraining
data, not fundamental inductive biases.

EXPERIMENTS
-----------
A  — Linear regression (Raventós replication, sanity check)
     Vary M distinct beta vectors, measure OOD RMSE vs noise floor.
     Direct comparison to published results.

B1 — Process family coverage sweep (most important)
     Train on subsets of the 10 sampler families:
       ar_only → ar_arma → ar_arma_arima → full
     Tests the Wang et al. question: does training on AR generalise to ARIMA?
     Must generate on-the-fly — pool has fixed family mixture.

B2 — AR order coverage sweep
     Restrict training to low-order AR, test on H1 (AR(10)).
     Must generate on-the-fly — pool has fixed ar_order_hi=10.

B3 — Coefficient magnitude sweep
     Vary ar_coeff_scale within the AR family.
     Must generate on-the-fly — pool has fixed ar_coeff_scale=0.6.

C  — Training steps sweep (number of examples)
     500 to 100k steps, find threshold where learning saturates.
     Uses pre-generated pool for speed.

W&B logs loss curves for every condition so you can verify convergence
without needing to run twice. Based on the main run_compare W&B results,
convergence occurs around 20k steps, so 25k is used as default.

Usage:
    python run_density_experiment.py
    python run_density_experiment.py --experiments B1 C
    python run_density_experiment.py --pool_path series_pool.npz
    python run_density_experiment.py --no_wandb
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from train_transformer import train_iid, eval_loop, resolve_device
from train_transformer import MSARBatchSampler, MSARSamplerConfig
from models.transformer_forecaster import TransformerConfig, CausalTransformerForecaster
from data.synthetic_npz_dataset import make_train_val_datasets


# ── Dataset list ─────────────────────────────────────────────────
DATASETS: List[str] = [
    "A1_ar2_coeffs_easy", "A2_ar2_coeffs_hard", "A3_ar2_coeffs_plus_var",
    "B1_ar2_variance", "B2_ar2_variance_big", "C1_arma21_coeffs_var",
    "D1_arima211", "D2_arima221", "D3_arima210",
    "E1_drift_only", "E2_level_shift", "F1_seasonal_sarimax",
    "F2_seasonal_exog", "G1_exogenous_only", "H1_ar10_coeffs",
    "H2_ar1_near_unit_root", "S1_sparse_switching", "S2_frequent_switching",
    "NS0_A1_no_switch_regime0", "NS1_A1_no_switch_regime1", "SW1_A1_single_switch",
]

AR_DATASETS       = ["A1_ar2_coeffs_easy", "A2_ar2_coeffs_hard", "A3_ar2_coeffs_plus_var",
                     "B1_ar2_variance", "B2_ar2_variance_big", "S1_sparse_switching",
                     "S2_frequent_switching", "H2_ar1_near_unit_root"]
ARMA_DATASETS     = ["C1_arma21_coeffs_var"]
ARIMA_DATASETS    = ["D1_arima211", "D2_arima221", "D3_arima210"]
SEASONAL_DATASETS = ["F1_seasonal_sarimax", "F2_seasonal_exog"]
EXOG_DATASETS     = ["E1_drift_only", "E2_level_shift", "G1_exogenous_only"]


# ── Family mix weight presets ─────────────────────────────────────
# Weights for [ar, ar_near_unit, ar_no_switch, arma, arima1, arima2,
#              seasonal, exog_const, exog_sine, exog_seasonal]
FAMILY_PRESETS = {
    "ar_only": dict(
        mix_ar=0.60, mix_ar_near_unit=0.15, mix_ar_no_switch=0.15,
        mix_arma=0.00, mix_arima1=0.00, mix_arima2=0.00,
        mix_seasonal=0.00, mix_exog_const=0.05, mix_exog_sine=0.05,
        mix_exog_seasonal=0.00,
    ),
    "ar_arma": dict(
        mix_ar=0.35, mix_ar_near_unit=0.10, mix_ar_no_switch=0.10,
        mix_arma=0.35, mix_arima1=0.00, mix_arima2=0.00,
        mix_seasonal=0.00, mix_exog_const=0.05, mix_exog_sine=0.05,
        mix_exog_seasonal=0.00,
    ),
    "ar_arma_arima": dict(
        mix_ar=0.22, mix_ar_near_unit=0.07, mix_ar_no_switch=0.07,
        mix_arma=0.18, mix_arima1=0.18, mix_arima2=0.10,
        mix_seasonal=0.00, mix_exog_const=0.09, mix_exog_sine=0.09,
        mix_exog_seasonal=0.00,
    ),
    "full": dict(
        mix_ar=0.22, mix_ar_near_unit=0.05, mix_ar_no_switch=0.05,
        mix_arma=0.13, mix_arima1=0.13, mix_arima2=0.07,
        mix_seasonal=0.12, mix_exog_const=0.08, mix_exog_sine=0.08,
        mix_exog_seasonal=0.07,
    ),
}


# ── Shared helpers ───────────────────────────────────────────────

def build_model(context_len, d_model, n_heads, n_layers, dropout, seed, device):
    cfg = TransformerConfig(
        context_len=context_len, d_model=d_model,
        n_heads=n_heads, n_layers=n_layers, dropout=dropout,
    )
    torch.manual_seed(seed)
    model = CausalTransformerForecaster(cfg).to(device)
    model.train()
    return model


def build_sampler(
    ar_coeff_scale: float = 0.6,
    ar_order_lo: int = 1,
    ar_order_hi: int = 10,
    seed: int = 0,
    pool_path: Optional[str] = None,
    family_weights: Optional[dict] = None,
) -> MSARBatchSampler:
    """
    Build a sampler with the given settings.
    pool_path: if provided, load the pre-generated pool.
               Raises an error (not silent) if pool loading fails so
               you know immediately if the pool is missing or corrupt.
    """
    weights = family_weights or FAMILY_PRESETS["full"]
    cfg = MSARSamplerConfig(
        series_len=512, k_regimes=2,
        ar_coeff_scale=ar_coeff_scale,
        ma_coeff_scale=0.4, sar_coeff_scale=0.35,
        sigma_lo=0.15, sigma_hi=0.70,
        persistence_lo=0.85, persistence_hi=0.98,
        burn_in=100,
        ar_order_lo=ar_order_lo, ar_order_hi=ar_order_hi,
        **weights,
    )
    sampler = MSARBatchSampler(cfg, seed=seed)
    if pool_path is not None:
        # No try/except — fail loudly if pool is missing or wrong format
        sampler.load_pool(pool_path)
        print(f"  Using pool: {pool_path}")
    else:
        print(f"  Using on-the-fly generation (no pool)")
    return sampler


def eval_suite(
    model, data_dir, datasets, n_instances,
    context_len, val_frac, batch_size, device,
) -> Dict[str, float]:
    """Evaluate model on dataset list, averaging across instances."""
    model.eval()
    results = {}
    for ds in datasets:
        vals = []
        for i in range(n_instances):
            npz = Path(data_dir) / f"{ds}_r{i}.npz"
            if not npz.exists():
                continue
            _, ds_val, _, _ = make_train_val_datasets(str(npz), context_len, val_frac)
            loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False)
            _, rmse = eval_loop(model, loader, device)
            vals.append(rmse)
        if vals:
            results[ds] = float(np.mean(vals))
    model.train()

    def group_mean(group):
        vals = [results[d] for d in group if d in results]
        return float(np.mean(vals)) if vals else float("nan")

    results["mean_all"]      = group_mean(DATASETS)
    results["mean_ar"]       = group_mean(AR_DATASETS)
    results["mean_arima"]    = group_mean(ARIMA_DATASETS)
    results["mean_seasonal"] = group_mean(SEASONAL_DATASETS)
    results["mean_exog"]     = group_mean(EXOG_DATASETS)
    return results


def train_and_eval(
    model, sampler, val_loader_monitor,
    steps, batch_size, lr, device,
    data_dir, n_instances, context_len, val_frac,
    msar_df, datasets=None,
    wandb_run=None, wandb_prefix="",
) -> Dict[str, Any]:
    if datasets is None:
        datasets = DATASETS

    train_iid(model, sampler, val_loader_monitor, steps, batch_size, lr, device,
              wandb_run=wandb_run if wandb_prefix else None)

    results = eval_suite(
        model, data_dir, datasets, n_instances,
        context_len, val_frac, batch_size, device,
    )

    if msar_df is not None:
        gaps = [
            results[ds] - float(msar_df.loc[ds, "msar_val_rmse"])
            for ds in datasets
            if ds in results and ds in msar_df.index
            and not np.isnan(float(msar_df.loc[ds, "msar_val_rmse"]))
        ]
        results["mean_gap_vs_msar"] = float(np.mean(gaps)) if gaps else float("nan")

    if wandb_run is not None and wandb_prefix:
        wandb_run.log({
            f"{wandb_prefix}/{k}": v
            for k, v in results.items() if isinstance(v, float)
        })

    return results


# One representative per process family for monitoring validation during training.
# Using multiple families gives a more balanced signal than A1 alone.
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
    Missing files are skipped gracefully.
    """
    from torch.utils.data import ConcatDataset
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


# ================================================================
# EXPERIMENT A — Linear regression (Raventós replication)
# ================================================================

def run_experiment_a(
    device, seed=0, d=1, context_len=64,
    steps=25000, batch_size=128, wandb_run=None,
) -> pd.DataFrame:
    """
    Vary M (distinct beta vectors in training pool) and measure OOD RMSE.
    Below threshold M*: transformer only learned the M training tasks.
    Above M*: transformer learned a general regression algorithm.
    """
    print("\n" + "="*60)
    print("EXPERIMENT A: Linear regression (Raventós replication)")
    print(f"  d={d}  context_len={context_len}  steps={steps}")
    print("="*60)

    M_values = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096, 16384]
    noise_sigma = 0.1
    rows = []

    for M in M_values:
        print(f"\n--- M={M} distinct beta vectors ---")
        torch.manual_seed(seed)
        np.random.seed(seed)
        rng = np.random.default_rng(seed)

        beta_pool = rng.standard_normal((M,)).astype(np.float32)  # M scalars

        cfg = TransformerConfig(
            context_len=context_len, d_model=128,
            n_heads=4, n_layers=4, dropout=0.0,
        )
        model = CausalTransformerForecaster(cfg).to(device)
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
        loss_fn = torch.nn.MSELoss()

        t0 = time.time()
        for step in range(steps):
            B = batch_size
            betas = beta_pool[np.random.randint(0, M, size=B)]  # (B,)
            x     = np.random.standard_normal((B, context_len)).astype(np.float32)
            noise = np.random.normal(0, noise_sigma, (B, context_len)).astype(np.float32)
            y     = betas[:, None] * x + noise  # (B, L)

            x_t = torch.from_numpy(x[:, :, None]).to(device)
            y_t = torch.from_numpy(y[:, :, None]).to(device)
            y_target = torch.cat([y_t[:, 1:, :], y_t[:, -1:, :]], dim=1)

            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x_t), y_target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if step % 100 == 0 and wandb_run is not None:
                wandb_run.log({f"exp_a/M{M}/loss": float(loss.item()),
                               f"exp_a/M{M}/step": step})

        elapsed = time.time() - t0

        # Evaluate on OOD betas drawn fresh — never seen during training
        model.eval()
        n_test    = 1024
        test_beta = np.random.standard_normal(n_test).astype(np.float32)
        test_x    = np.random.standard_normal((n_test, context_len)).astype(np.float32)
        test_y    = test_beta[:, None] * test_x + np.random.normal(
            0, noise_sigma, (n_test, context_len)).astype(np.float32)

        x_t    = torch.from_numpy(test_x[:, :, None]).to(device)
        y_true = torch.from_numpy(test_y[:, -1:]).to(device)
        with torch.no_grad():
            yhat = model.predict_next(x_t)
        ood_rmse = float(torch.sqrt(
            torch.nn.functional.mse_loss(yhat, y_true)).item())

        ratio = ood_rmse / noise_sigma
        print(f"  OOD RMSE: {ood_rmse:.4f}  noise floor: {noise_sigma:.3f}  "
              f"ratio: {ratio:.2f}x  [{elapsed:.0f}s]")

        rows.append({
            "M": M, "ood_rmse": ood_rmse,
            "noise_floor": noise_sigma,
            "ratio_to_noise": ratio,
            "steps": steps,
        })

        if wandb_run is not None:
            wandb_run.log({
                "exp_a/M":               M,
                "exp_a/ood_rmse":        ood_rmse,
                "exp_a/ratio_to_noise":  ratio,
            })

    df = pd.DataFrame(rows)
    print("\nExperiment A summary:")
    print(df[["M", "ood_rmse", "ratio_to_noise"]].to_string(index=False))
    return df


# ================================================================
# EXPERIMENT B1 — Process family coverage sweep
# ================================================================

def run_experiment_b1(
    data_dir, device, msar_df,
    steps=25000, n_instances=3, seed=0, wandb_run=None,
    b1_pools: dict = None,
) -> pd.DataFrame:
    """
    Train on subsets of the 10 process families; test on all 21 datasets.
    Wang et al. question: does AR training generalise to ARIMA / seasonal?

    b1_pools: optional dict mapping preset name to pool path, e.g.
      {"ar_only": "pool_b1_ar_only.npz", "full": "series_pool.npz"}
      If a preset is not in b1_pools, on-the-fly generation is used.
      Pre-generated pools make reruns much faster.
    """
    print("\n" + "="*60)
    print("EXPERIMENT B1: Process family coverage sweep")
    print(f"  steps={steps}  n_instances={n_instances}  (on-the-fly generation)")
    print("="*60)

    context_len = 64
    batch_size  = 128
    lr          = 3e-4
    val_frac    = 0.3
    val_loader  = get_val_monitor_loader(data_dir, context_len, val_frac, batch_size)
    rows        = []

    for preset_name, weights in FAMILY_PRESETS.items():
        print(f"\n--- {preset_name} ---")
        nonzero = {k.replace("mix_", ""): v for k, v in weights.items() if v > 0}
        print(f"  active families: {nonzero}")
        torch.manual_seed(seed)
        np.random.seed(seed)

        model   = build_model(context_len, 256, 4, 6, 0.1, seed, device)
        pool_for_preset = (b1_pools or {}).get(preset_name, None)
        sampler = build_sampler(
            ar_coeff_scale=0.6, seed=seed,
            pool_path=pool_for_preset,
            family_weights=weights,
        )

        results = train_and_eval(
            model, sampler, val_loader, steps, batch_size, lr, device,
            data_dir, n_instances, context_len, val_frac, msar_df,
            wandb_run=wandb_run, wandb_prefix=f"exp_b1/{preset_name}",
        )

        print(f"  mean_all={results['mean_all']:.4f}  "
              f"mean_ar={results['mean_ar']:.4f}  "
              f"mean_arima={results['mean_arima']:.4f}  "
              f"mean_seasonal={results['mean_seasonal']:.4f}")
        if "mean_gap_vs_msar" in results:
            print(f"  mean_gap_vs_msar={results['mean_gap_vs_msar']:.4f}")

        row = {"family_preset": preset_name, "steps": steps}
        row.update({k: v for k, v in results.items() if isinstance(v, float)})
        rows.append(row)

    df = pd.DataFrame(rows)
    print("\nExperiment B1 summary:")
    cols = ["family_preset", "mean_all", "mean_ar", "mean_arima", "mean_seasonal"]
    print(df[cols].to_string(index=False))
    return df


# ================================================================
# EXPERIMENT B2 — AR order coverage sweep
# ================================================================

def run_experiment_b2(
    data_dir, device, msar_df,
    steps=25000, n_instances=3, seed=0, wandb_run=None,
) -> pd.DataFrame:
    """
    Restrict training AR order; test on H1 (AR(10)) vs A1-A3 (AR(2)).
    Cannot use pool — pool has fixed ar_order_hi=10.
    """
    print("\n" + "="*60)
    print("EXPERIMENT B2: AR order coverage sweep")
    print(f"  steps={steps}  n_instances={n_instances}  (on-the-fly generation)")
    print("="*60)

    context_len = 64
    batch_size  = 128
    lr          = 3e-4
    val_frac    = 0.3
    val_loader  = get_val_monitor_loader(data_dir, context_len, val_frac, batch_size)

    order_configs = [
        ("lo_order",  1, 2),
        ("mid_order", 1, 4),
        ("hi_order",  1, 6),
        ("full",      1, 10),
    ]
    rows = []

    for name, lo, hi in order_configs:
        print(f"\n--- {name}: ar_order in [{lo}, {hi}] ---")
        torch.manual_seed(seed)
        np.random.seed(seed)

        model   = build_model(context_len, 256, 4, 6, 0.1, seed, device)
        sampler = build_sampler(
            ar_coeff_scale=0.6,
            ar_order_lo=lo, ar_order_hi=hi,
            seed=seed, pool_path=None,
            family_weights=FAMILY_PRESETS["full"],
        )

        results = train_and_eval(
            model, sampler, val_loader, steps, batch_size, lr, device,
            data_dir, n_instances, context_len, val_frac, msar_df,
            wandb_run=wandb_run, wandb_prefix=f"exp_b2/{name}",
        )

        h1 = results.get("H1_ar10_coeffs", float("nan"))
        a1 = results.get("A1_ar2_coeffs_easy", float("nan"))
        print(f"  H1 (AR10): {h1:.4f}  A1 (AR2): {a1:.4f}  "
              f"mean_all: {results['mean_all']:.4f}")

        row = {"order_name": name, "ar_order_lo": lo, "ar_order_hi": hi, "steps": steps}
        row.update({k: v for k, v in results.items() if isinstance(v, float)})
        rows.append(row)

    df = pd.DataFrame(rows)
    print("\nExperiment B2 summary:")
    cols = ["order_name", "ar_order_hi", "H1_ar10_coeffs", "A1_ar2_coeffs_easy", "mean_all"]
    available = [c for c in cols if c in df.columns]
    print(df[available].to_string(index=False))
    return df


# ================================================================
# EXPERIMENT B3 — Coefficient magnitude sweep
# ================================================================

def run_experiment_b3(
    data_dir, device, msar_df,
    steps=25000, n_instances=3, seed=0, wandb_run=None,
) -> pd.DataFrame:
    """
    Vary ar_coeff_scale within the AR family.
    Cannot use pool — pool has fixed ar_coeff_scale=0.6.
    """
    print("\n" + "="*60)
    print("EXPERIMENT B3: AR coefficient magnitude sweep")
    print(f"  steps={steps}  n_instances={n_instances}  (on-the-fly generation)")
    print("="*60)

    context_len = 64
    batch_size  = 128
    lr          = 3e-4
    val_frac    = 0.3
    val_loader  = get_val_monitor_loader(data_dir, context_len, val_frac, batch_size)

    scales = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2]
    rows   = []

    for scale in scales:
        print(f"\n--- ar_coeff_scale={scale} ---")
        torch.manual_seed(seed)
        np.random.seed(seed)

        model   = build_model(context_len, 256, 4, 6, 0.1, seed, device)
        sampler = build_sampler(
            ar_coeff_scale=scale, seed=seed,
            pool_path=None,
            family_weights=FAMILY_PRESETS["full"],
        )

        results = train_and_eval(
            model, sampler, val_loader, steps, batch_size, lr, device,
            data_dir, n_instances, context_len, val_frac, msar_df,
            wandb_run=wandb_run, wandb_prefix=f"exp_b3/scale_{scale}",
        )

        a1 = results.get("A1_ar2_coeffs_easy", float("nan"))
        print(f"  A1 (has coeff 1.2): {a1:.4f}  mean_all: {results['mean_all']:.4f}  "
              f"mean_ar: {results['mean_ar']:.4f}")

        row = {"ar_coeff_scale": scale, "steps": steps}
        row.update({k: v for k, v in results.items() if isinstance(v, float)})
        rows.append(row)

    df = pd.DataFrame(rows)
    print("\nExperiment B3 summary:")
    cols = ["ar_coeff_scale", "mean_all", "mean_ar", "A1_ar2_coeffs_easy"]
    available = [c for c in cols if c in df.columns]
    print(df[available].to_string(index=False))
    return df


# ================================================================
# EXPERIMENT C — Training steps sweep (uses pre-generated pool)
# ================================================================

def run_experiment_c(
    data_dir, device, msar_df, pool_path,
    n_instances=3, seed=0, wandb_run=None,
) -> pd.DataFrame:
    """
    Vary training steps from 500 to 100k.
    Uses the pre-generated pool (ar_coeff_scale=0.6, full family mixture).
    Training distribution is held fixed — only quantity of examples varies.

    W&B loss curves from run_compare showed convergence around 20k steps.
    This experiment measures performance at each step count explicitly to
    confirm that threshold and check whether it differs across dataset families.
    """
    print("\n" + "="*60)
    print("EXPERIMENT C: Training steps sweep")
    print(f"  n_instances={n_instances}  ar_coeff_scale=0.6")
    print(f"  pool_path={pool_path}")
    print("  Total series seen = steps x 128")
    print("="*60)

    if pool_path is None:
        raise ValueError("Experiment C requires --pool_path. "
                         "Run: python generate_pool.py --out series_pool.npz")

    context_len = 64
    batch_size  = 128
    lr          = 3e-4
    val_frac    = 0.3
    val_loader  = get_val_monitor_loader(data_dir, context_len, val_frac, batch_size)

    step_counts = [500, 1000, 2000, 5000, 10000, 25000, 50000, 100000]
    rows = []

    for steps in step_counts:
        total = steps * batch_size
        print(f"\n--- steps={steps:,}  (total series: {total:,}) ---")
        torch.manual_seed(seed)
        np.random.seed(seed)

        model   = build_model(context_len, 256, 4, 6, 0.1, seed, device)
        sampler = build_sampler(
            ar_coeff_scale=0.6, seed=seed,
            pool_path=pool_path,       # uses pool — correct for Exp C
            family_weights=FAMILY_PRESETS["full"],
        )

        results = train_and_eval(
            model, sampler, val_loader, steps, batch_size, lr, device,
            data_dir, n_instances, context_len, val_frac, msar_df,
            wandb_run=wandb_run, wandb_prefix=f"exp_c/steps_{steps}",
        )

        print(f"  mean_all={results['mean_all']:.4f}  "
              f"mean_ar={results['mean_ar']:.4f}  "
              f"F1={results.get('F1_seasonal_sarimax', float('nan')):.4f}")
        if "mean_gap_vs_msar" in results:
            print(f"  mean_gap_vs_msar={results['mean_gap_vs_msar']:.4f}")

        if wandb_run is not None:
            wandb_run.log({
                "exp_c/steps":         steps,
                "exp_c/total_series":  total,
                "exp_c/mean_all":      results["mean_all"],
            })

        row = {"steps": steps, "total_series": total}
        row.update({k: v for k, v in results.items() if isinstance(v, float)})
        rows.append(row)

    df = pd.DataFrame(rows)
    print("\nExperiment C summary:")
    cols = ["steps", "total_series", "mean_all", "mean_ar", "F1_seasonal_sarimax"]
    available = [c for c in cols if c in df.columns]
    print(df[available].to_string(index=False))
    return df


# ================================================================
# Main
# ================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Data density experiments."
    )
    ap.add_argument(
        "--experiments", nargs="+", default=["A", "B1", "B2", "B3", "C"],
        choices=["A", "B1", "B2", "B3", "C"],
    )
    ap.add_argument("--data_dir",      type=str,   default="generated_data")
    ap.add_argument("--pool_path",     type=str,   default=None,
                    help="Required for Experiment C. Not used for B2/B3.")
    ap.add_argument("--pool_b1_ar_only",      type=str, default=None,
                    help="Pre-generated pool for B1 ar_only preset.")
    ap.add_argument("--pool_b1_ar_arma",      type=str, default=None,
                    help="Pre-generated pool for B1 ar_arma preset.")
    ap.add_argument("--pool_b1_ar_arma_arima",type=str, default=None,
                    help="Pre-generated pool for B1 ar_arma_arima preset.")
    ap.add_argument("--pool_b1_full",         type=str, default=None,
                    help="Pre-generated pool for B1 full preset.")
    ap.add_argument("--msar_csv",      type=str,   default="msar_results.csv")
    ap.add_argument("--n_instances",   type=int,   default=3)
    ap.add_argument("--seed",          type=int,   default=0)
    ap.add_argument("--no_wandb",      action="store_true")
    ap.add_argument("--wandb_project", type=str,   default="switch-transformers")
    ap.add_argument("--exp_b_steps",   type=int,   default=25000,
                    help="Steps for B1/B2/B3 (default 25000, based on convergence from W&B).")
    ap.add_argument("--exp_a_steps",   type=int,   default=25000,
                    help="Steps for Experiment A (default 25000).")
    args = ap.parse_args()

    device = resolve_device("cuda")

    # Load MSAR results
    msar_df = None
    if Path(args.msar_csv).exists():
        msar_df = pd.read_csv(args.msar_csv).set_index("dataset")
        print(f"Loaded MSAR results from {args.msar_csv} ({len(msar_df)} datasets)")
    else:
        print(f"[warning] {args.msar_csv} not found — gap vs MSAR will not be computed")

    # Check evaluation datasets for time series experiments
    ts_exps = {"B1", "B2", "B3", "C"}
    if any(e in args.experiments for e in ts_exps):
        missing = [
            ds for ds in DATASETS for i in range(args.n_instances)
            if not (Path(args.data_dir) / f"{ds}_r{i}.npz").exists()
        ]
        if missing:
            print(f"[error] {len(missing)} dataset files missing. "
                  f"Run: python data_generation.py")
            return

    # Check pool for Experiment C
    if "C" in args.experiments and args.pool_path is None:
        print("[error] Experiment C requires --pool_path series_pool.npz")
        return

    # Initialise W&B
    wandb_run = None
    if not args.no_wandb:
        try:
            import wandb, os
            os.environ["WANDB_MODE"] = "offline"
            wandb_run = wandb.init(
                project=args.wandb_project,
                name="density_experiments",
                config={
                    "experiments":  args.experiments,
                    "pool_path":    args.pool_path,
                    "n_instances":  args.n_instances,
                    "exp_b_steps":  args.exp_b_steps,
                    "exp_a_steps":  args.exp_a_steps,
                    "note": "25k steps based on W&B convergence analysis of run_compare",
                },
            )
            print("W&B run initialised (offline mode)\n")
        except Exception as e:
            print(f"[warning] W&B init failed ({e}). Continuing without logging.")

    saved = []

    if "A" in args.experiments:
        df = run_experiment_a(
            device=device, seed=args.seed,
            steps=args.exp_a_steps, wandb_run=wandb_run,
        )
        df.to_csv("results_density_exp_a.csv", index=False)
        saved.append("results_density_exp_a.csv")

    if "B1" in args.experiments:
        b1_pools = {}
        if args.pool_b1_ar_only:       b1_pools["ar_only"]       = args.pool_b1_ar_only
        if args.pool_b1_ar_arma:       b1_pools["ar_arma"]       = args.pool_b1_ar_arma
        if args.pool_b1_ar_arma_arima: b1_pools["ar_arma_arima"] = args.pool_b1_ar_arma_arima
        if args.pool_b1_full:          b1_pools["full"]          = args.pool_b1_full

        df = run_experiment_b1(
            data_dir=args.data_dir, device=device, msar_df=msar_df,
            steps=args.exp_b_steps, n_instances=args.n_instances,
            seed=args.seed, wandb_run=wandb_run,
            b1_pools=b1_pools if b1_pools else None,
        )
        df.to_csv("results_density_exp_b1.csv", index=False)
        saved.append("results_density_exp_b1.csv")

    if "B2" in args.experiments:
        df = run_experiment_b2(
            data_dir=args.data_dir, device=device, msar_df=msar_df,
            steps=args.exp_b_steps, n_instances=args.n_instances,
            seed=args.seed, wandb_run=wandb_run,
        )
        df.to_csv("results_density_exp_b2.csv", index=False)
        saved.append("results_density_exp_b2.csv")

    if "B3" in args.experiments:
        df = run_experiment_b3(
            data_dir=args.data_dir, device=device, msar_df=msar_df,
            steps=args.exp_b_steps, n_instances=args.n_instances,
            seed=args.seed, wandb_run=wandb_run,
        )
        df.to_csv("results_density_exp_b3.csv", index=False)
        saved.append("results_density_exp_b3.csv")

    if "C" in args.experiments:
        df = run_experiment_c(
            data_dir=args.data_dir, device=device, msar_df=msar_df,
            pool_path=args.pool_path,
            n_instances=args.n_instances,
            seed=args.seed, wandb_run=wandb_run,
        )
        df.to_csv("results_density_exp_c.csv", index=False)
        saved.append("results_density_exp_c.csv")

    if wandb_run is not None:
        wandb_run.finish()
        print("\nSync W&B with: wandb sync wandb/offline-run-<id>")

    print("\n=== All density experiments complete ===")
    for f in saved:
        print(f"  {f}")


if __name__ == "__main__":
    main()
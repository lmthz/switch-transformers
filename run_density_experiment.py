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
B1 — Process family coverage sweep (most important)
     Train on subsets of the 10 sampler families:
       ar_only → ar_arma → ar_arma_arima → full
     Tests the Wang et al. question: does training on AR generalise to ARIMA?
     Uses pre-generated pools (one per family preset); falls back to on-the-fly if missing.

B2 — AR order coverage sweep
     Restrict training to low-order AR, test on H1 (AR(10)).
     Uses pre-generated pools (one per order condition); falls back to on-the-fly if missing.

B3 — Coefficient magnitude sweep
     Vary ar_coeff_scale within the AR family.
     Uses pre-generated pools (one per scale condition); falls back to on-the-fly if missing.

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
# ── Full evaluation suite (used for D, E, general evaluation) ────
DATASETS: List[str] = [
    "A1_ar2_coeffs_easy", "A2_ar2_coeffs_hard", "A3_ar2_coeffs_plus_var",
    "B1_ar2_variance", "B2_ar2_variance_big", "C1_arma21_coeffs_var",
    "D1_arima211", "D2_arima221", "D3_arima210",
    "E1_drift_only", "E2_level_shift", "F1_seasonal_sarimax",
    "F2_seasonal_exog", "G1_exogenous_only", "H1_ar10_coeffs",
    "H2_ar1_near_unit_root", "S1_sparse_switching", "S2_frequent_switching",
    "NS0_A1_no_switch_regime0", "NS1_A1_no_switch_regime1", "SW1_A1_single_switch",
]

# ── Family groupings for full suite ───────────────────────────────
AR_DATASETS       = ["A1_ar2_coeffs_easy", "A2_ar2_coeffs_hard", "A3_ar2_coeffs_plus_var",
                     "B1_ar2_variance", "B2_ar2_variance_big", "S1_sparse_switching",
                     "S2_frequent_switching", "H2_ar1_near_unit_root"]
ARMA_DATASETS     = ["C1_arma21_coeffs_var"]
ARIMA_DATASETS    = ["D1_arima211", "D2_arima221", "D3_arima210"]
SEASONAL_DATASETS = ["F1_seasonal_sarimax", "F2_seasonal_exog"]
EXOG_DATASETS     = ["E1_drift_only", "E2_level_shift", "G1_exogenous_only"]

# ── Clean evaluation suites per experiment axis ───────────────────
# Each experiment varies exactly one axis. Datasets that vary a different
# axis are excluded to avoid contamination.

# B1 (family coverage): exclude H1/H2 (vary AR order, not family)
# and A3 (varies both coefficients and sigma simultaneously)
DATASETS_B1 = [
    "A1_ar2_coeffs_easy", "A2_ar2_coeffs_hard",
    "B1_ar2_variance", "B2_ar2_variance_big",         # AR family
    "C1_arma21_coeffs_var",                            # ARMA family
    "D1_arima211", "D2_arima221", "D3_arima210",       # ARIMA family
    "F1_seasonal_sarimax", "F2_seasonal_exog",         # Seasonal family
    "E1_drift_only", "E2_level_shift", "G1_exogenous_only",  # Exog family
    "S1_sparse_switching", "S2_frequent_switching",    # Switching variants
]

# B2 (AR order): evaluation order coverage matches training order coverage.
# A1/A2 are AR(2) in-distribution controls; H3/H4/H1 are OOD targets at
# increasing order (4, 6, 10) using the same coefficient decay pattern.
DATASETS_B2 = [
    "A1_ar2_coeffs_easy", "A2_ar2_coeffs_hard",       # AR(2) control
    "H3_ar4_coeffs",                                   # AR(4) OOD target
    "H4_ar6_coeffs",                                   # AR(6) OOD target
    "H1_ar10_coeffs",                                  # AR(10) OOD target
]

# F (no-switch): same families as B1 but without switching datasets.
# Generated by generate_noswitch_data.py into generated_data_noswitch/.
DATASETS_NOSWITCH = [
    "A1_ar2_coeffs_easy", "A2_ar2_coeffs_hard", "A3_ar2_coeffs_plus_var",
    "B1_ar2_variance", "B2_ar2_variance_big",
    "C1_arma21_coeffs_var",
    "D1_arima211", "D3_arima210",
    "E1_drift_only", "E2_level_shift",
    "F1_seasonal_sarimax", "F2_seasonal_exog",
    "G1_exogenous_only",
    "H1_ar10_coeffs", "H2_ar1_near_unit_root",
    "H3_ar4_coeffs", "H4_ar6_coeffs",
]

# B3 (coefficient magnitude): AR(2) datasets only, exclude A3 (confound),
# H1 (AR(10)), H2 (AR(1))
DATASETS_B3 = [
    "A1_ar2_coeffs_easy", "A2_ar2_coeffs_hard",
    "B1_ar2_variance", "B2_ar2_variance_big",
    "S1_sparse_switching", "S2_frequent_switching",
    "NS0_A1_no_switch_regime0", "NS1_A1_no_switch_regime1", "SW1_A1_single_switch",
]

# Filtered no-switch dataset lists (intersection of each experiment's set with DATASETS_NOSWITCH)
_NS_SET = set(DATASETS_NOSWITCH)
DATASETS_B1_NS = [d for d in DATASETS_B1 if d in _NS_SET]   # removes D2_arima221, S1, S2
DATASETS_B2_NS = [d for d in DATASETS_B2 if d in _NS_SET]   # all 5 stay
DATASETS_B3_NS = [d for d in DATASETS_B3 if d in _NS_SET]   # keeps A1, A2, B1_ar2, B2_ar2 only


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
    ar_coeff_scale: float = 1.2,
    ar_order_lo: int = 2,
    ar_order_hi: int = 2,
    seed: int = 0,
    pool_path: Optional[str] = None,
    family_weights: Optional[dict] = None,
    force_no_switch: bool = False,
) -> MSARBatchSampler:
    """
    Build a sampler with the given settings.
    pool_path: if provided, load the pre-generated pool.
               Raises an error (not silent) if pool loading fails so
               you know immediately if the pool is missing or corrupt.
    force_no_switch: if True, all generated series stay in one regime.
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
        force_no_switch=force_no_switch,
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
# Per-experiment val monitor datasets, aligned with each experiment's evaluation set.

# B1, C, D, E all evaluate on DATASETS_B1 (excludes H1, H2, A3, NS0, NS1, SW1)
VAL_MONITOR_B1CDE = [
    "A1_ar2_coeffs_easy",   # AR switching
    "C1_arma21_coeffs_var", # ARMA
    "D1_arima211",          # ARIMA
    "F1_seasonal_sarimax",  # Seasonal
    "G1_exogenous_only",    # Exogenous
    "E1_drift_only",        # Drift (pure exogenous)
]

# B2 evaluates on DATASETS_B2 (A1, A2, H1 only)
VAL_MONITOR_B2 = [
    "A1_ar2_coeffs_easy",   # AR(2) in-distribution control
    "H1_ar10_coeffs",       # AR(10) OOD target — key B2 signal
]

# B3 evaluates on DATASETS_B3 (AR(2) datasets only)
VAL_MONITOR_B3 = [
    "A1_ar2_coeffs_easy",   # Main AR(2) target
    "A2_ar2_coeffs_hard",   # Hard AR(2)
    "S1_sparse_switching",  # Sparse-switching AR(2)
]

# No-switch mode val monitor: one per process family, all in DATASETS_NOSWITCH
VAL_MONITOR_NS = [
    "A1_ar2_coeffs_easy",   # AR
    "C1_arma21_coeffs_var", # ARMA
    "D1_arima211",          # ARIMA
    "F1_seasonal_sarimax",  # Seasonal
    "G1_exogenous_only",    # Exogenous
]


def get_val_monitor_loader(data_dir, context_len, val_frac, batch_size, datasets=None):
    """
    Build a combined validation DataLoader for training monitoring.
    Pass a per-experiment dataset list; defaults to VAL_MONITOR_B1CDE.
    Missing files are skipped gracefully.
    """
    from torch.utils.data import ConcatDataset
    if datasets is None:
        datasets = VAL_MONITOR_B1CDE
    val_sets = []
    for ds in datasets:
        npz = Path(data_dir) / f"{ds}_r0.npz"
        if npz.exists():
            _, ds_val, _, _ = make_train_val_datasets(str(npz), context_len, val_frac)
            val_sets.append(ds_val)
    if not val_sets:
        raise RuntimeError(f"No validation datasets found in {data_dir}")
    combined = ConcatDataset(val_sets)
    return DataLoader(combined, batch_size=batch_size, shuffle=False)


# ================================================================
# EXPERIMENT B1 — Process family coverage sweep
# ================================================================

def run_experiment_b1(
    data_dir, device, msar_df,
    steps=25000, n_instances=3, seed=0, wandb_run=None,
    b1_pools: dict = None,
    noswitch: bool = False,
) -> pd.DataFrame:
    """
    Train on subsets of the 10 process families; test on all datasets.
    Wang et al. question: does AR training generalise to ARIMA / seasonal?

    b1_pools: optional dict mapping preset name to pool path.
      Ignored when noswitch=True (always on-the-fly with force_no_switch).
    noswitch: if True, train with single-regime series and evaluate on
      DATASETS_B1_NS (no switching datasets only).
    """
    tag     = "B1_NS" if noswitch else "B1"
    datasets = DATASETS_B1_NS if noswitch else DATASETS_B1
    monitor  = VAL_MONITOR_NS if noswitch else VAL_MONITOR_B1CDE
    prefix   = "exp_b1_ns" if noswitch else "exp_b1"

    print("\n" + "="*60)
    print(f"EXPERIMENT {tag}: Process family coverage sweep")
    print(f"  steps={steps}  n_instances={n_instances}  noswitch={noswitch}")
    print("="*60)

    context_len = 64
    batch_size  = 128
    lr          = 3e-4
    val_frac    = 0.3
    val_loader  = get_val_monitor_loader(data_dir, context_len, val_frac, batch_size, monitor)
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
            ar_coeff_scale=1.2, seed=seed,
            ar_order_lo=2, ar_order_hi=2,
            pool_path=pool_for_preset,
            family_weights=weights,
            force_no_switch=noswitch,
        )

        results = train_and_eval(
            model, sampler, val_loader, steps, batch_size, lr, device,
            data_dir, n_instances, context_len, val_frac, msar_df,
            datasets=datasets,
            wandb_run=wandb_run, wandb_prefix=f"{prefix}/{preset_name}",
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
    b2_pools: dict = None,
    noswitch: bool = False,
) -> pd.DataFrame:
    """
    Restrict training AR order; test on H1 (AR(10)) vs A1-A3 (AR(2)).
    b2_pools: optional dict mapping order_name to pool path. Ignored when noswitch=True.
    noswitch: if True, train with single-regime series and evaluate on DATASETS_B2_NS.
    """
    tag      = "B2_NS" if noswitch else "B2"
    datasets = DATASETS_B2_NS if noswitch else DATASETS_B2
    prefix   = "exp_b2_ns" if noswitch else "exp_b2"

    print("\n" + "="*60)
    print(f"EXPERIMENT {tag}: AR order coverage sweep")
    print(f"  steps={steps}  n_instances={n_instances}  noswitch={noswitch}")
    print("="*60)

    context_len = 64
    batch_size  = 128
    lr          = 3e-4
    val_frac    = 0.3
    val_loader  = get_val_monitor_loader(data_dir, context_len, val_frac, batch_size, VAL_MONITOR_B2)

    order_configs = [
        ("lo_order",  2, 2),
        ("mid_order", 2, 4),
        ("hi_order",  2, 6),
        ("full",      2, 10),
    ]
    rows = []

    for name, lo, hi in order_configs:
        print(f"\n--- {name}: ar_order in [{lo}, {hi}] ---")
        torch.manual_seed(seed)
        np.random.seed(seed)

        model   = build_model(context_len, 256, 4, 6, 0.1, seed, device)
        pool_for_order = (b2_pools or {}).get(name, None)
        sampler = build_sampler(
            ar_coeff_scale=1.2,
            ar_order_lo=lo, ar_order_hi=hi,
            seed=seed, pool_path=pool_for_order,
            family_weights=FAMILY_PRESETS["full"],
            force_no_switch=noswitch,
        )

        results = train_and_eval(
            model, sampler, val_loader, steps, batch_size, lr, device,
            data_dir, n_instances, context_len, val_frac, msar_df,
            datasets=datasets,
            wandb_run=wandb_run, wandb_prefix=f"{prefix}/{name}",
        )

        h1 = results.get("H1_ar10_coeffs", float("nan"))
        h4 = results.get("H4_ar6_coeffs",  float("nan"))
        h3 = results.get("H3_ar4_coeffs",  float("nan"))
        a1 = results.get("A1_ar2_coeffs_easy", float("nan"))
        print(f"  H1 (AR10): {h1:.4f}  H4 (AR6): {h4:.4f}  "
              f"H3 (AR4): {h3:.4f}  A1 (AR2): {a1:.4f}  "
              f"mean_all: {results['mean_all']:.4f}")

        row = {"order_name": name, "ar_order_lo": lo, "ar_order_hi": hi, "steps": steps}
        row.update({k: v for k, v in results.items() if isinstance(v, float)})
        rows.append(row)

    df = pd.DataFrame(rows)
    print("\nExperiment B2 summary:")
    cols = ["order_name", "ar_order_hi", "H1_ar10_coeffs", "H4_ar6_coeffs",
            "H3_ar4_coeffs", "A1_ar2_coeffs_easy", "mean_all"]
    available = [c for c in cols if c in df.columns]
    print(df[available].to_string(index=False))
    return df


# ================================================================
# EXPERIMENT B3 — Coefficient magnitude sweep
# ================================================================

def run_experiment_b3(
    data_dir, device, msar_df,
    steps=25000, n_instances=3, seed=0, wandb_run=None,
    b3_pools: dict = None,
    noswitch: bool = False,
) -> pd.DataFrame:
    """
    Vary ar_coeff_scale within the AR family.
    b3_pools: optional dict mapping scale string to pool path. Ignored when noswitch=True.
    noswitch: if True, train with single-regime series and evaluate on DATASETS_B3_NS.
    """
    tag      = "B3_NS" if noswitch else "B3"
    datasets = DATASETS_B3_NS if noswitch else DATASETS_B3
    monitor  = VAL_MONITOR_NS[:2] if noswitch else VAL_MONITOR_B3  # AR-only monitor for NS B3
    prefix   = "exp_b3_ns" if noswitch else "exp_b3"

    print("\n" + "="*60)
    print(f"EXPERIMENT {tag}: AR coefficient magnitude sweep")
    print(f"  steps={steps}  n_instances={n_instances}  noswitch={noswitch}")
    print("="*60)

    context_len = 64
    batch_size  = 128
    lr          = 3e-4
    val_frac    = 0.3
    val_loader  = get_val_monitor_loader(data_dir, context_len, val_frac, batch_size, monitor)

    scales = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2]
    rows   = []

    for scale in scales:
        print(f"\n--- ar_coeff_scale={scale} ---")
        torch.manual_seed(seed)
        np.random.seed(seed)

        model   = build_model(context_len, 256, 4, 6, 0.1, seed, device)
        pool_for_scale = (b3_pools or {}).get(str(scale), None)
        # XOR seed with scale so each condition gets a genuinely different RNG stream.
        # Without this, on-the-fly generation produces series that are rescaled
        # versions of each other and look identical after standardisation.
        scale_seed = seed ^ int(scale * 1000)
        sampler = build_sampler(
            ar_coeff_scale=scale, seed=scale_seed,
            ar_order_lo=2, ar_order_hi=2,
            pool_path=pool_for_scale,
            family_weights=FAMILY_PRESETS["full"],
            force_no_switch=noswitch,
        )

        results = train_and_eval(
            model, sampler, val_loader, steps, batch_size, lr, device,
            data_dir, n_instances, context_len, val_frac, msar_df,
            datasets=datasets,
            wandb_run=wandb_run, wandb_prefix=f"{prefix}/scale_{scale}",
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
    noswitch: bool = False,
) -> pd.DataFrame:
    """
    Vary training steps from 500 to 100k.
    Uses the pre-generated pool (ar_coeff_scale=1.2, full family mixture).
    Training distribution is held fixed — only quantity of examples varies.

    noswitch: if True, expect pool_path to be pool_noswitch.npz and evaluate
      on DATASETS_NOSWITCH. Both training and eval are single-regime.
    """
    tag      = "C_NS" if noswitch else "C"
    datasets = DATASETS_NOSWITCH if noswitch else DATASETS_B1
    monitor  = VAL_MONITOR_NS if noswitch else VAL_MONITOR_B1CDE
    prefix   = "exp_c_ns" if noswitch else "exp_c"

    print("\n" + "="*60)
    print(f"EXPERIMENT {tag}: Training steps sweep")
    print(f"  n_instances={n_instances}  ar_coeff_scale=1.2  noswitch={noswitch}")
    print(f"  pool_path={pool_path}")
    print("  Total series seen = steps x 128")
    print("="*60)

    if pool_path is None:
        pool_hint = "pool_noswitch.npz" if noswitch else "series_pool.npz"
        raise ValueError(f"Experiment {tag} requires --pool_{'noswitch' if noswitch else 'path'}. "
                         f"Run: python generate_pool.py {'--no_switch ' if noswitch else ''}--out {pool_hint}")

    context_len = 64
    batch_size  = 128
    lr          = 3e-4
    val_frac    = 0.3
    val_loader  = get_val_monitor_loader(data_dir, context_len, val_frac, batch_size, monitor)

    step_counts = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 25000, 50000, 100000]
    rows = []

    for steps in step_counts:
        total = steps * batch_size
        print(f"\n--- steps={steps:,}  (total series: {total:,}) ---")
        torch.manual_seed(seed)
        np.random.seed(seed)

        model   = build_model(context_len, 256, 4, 6, 0.1, seed, device)
        sampler = build_sampler(
            ar_coeff_scale=1.2, seed=seed,
            ar_order_lo=2, ar_order_hi=2,
            pool_path=pool_path,
            family_weights=FAMILY_PRESETS["full"],
            force_no_switch=noswitch,
        )

        results = train_and_eval(
            model, sampler, val_loader, steps, batch_size, lr, device,
            data_dir, n_instances, context_len, val_frac, msar_df,
            datasets=datasets,
            wandb_run=wandb_run, wandb_prefix=f"{prefix}/steps_{steps}",
        )

        print(f"  mean_all={results['mean_all']:.4f}  mean_ar={results['mean_ar']:.4f}")
        if "mean_gap_vs_msar" in results:
            print(f"  mean_gap_vs_msar={results['mean_gap_vs_msar']:.4f}")

        if wandb_run is not None:
            wandb_run.log({
                f"{prefix}/steps":        steps,
                f"{prefix}/total_series": total,
                f"{prefix}/mean_all":     results["mean_all"],
            })

        row = {"steps": steps, "total_series": total}
        row.update({k: v for k, v in results.items() if isinstance(v, float)})
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"\nExperiment {tag} summary:")
    cols = ["steps", "total_series", "mean_all", "mean_ar"]
    available = [c for c in cols if c in df.columns]
    print(df[available].to_string(index=False))
    return df


# ================================================================
# EXPERIMENT D — Task diversity sweep (Raventós replication for time series)
# ================================================================

def run_experiment_d(
    data_dir: str,
    device: torch.device,
    msar_df,
    n_instances: int = 3,
    seed: int = 0,
    wandb_run=None,
    pool_dir: str = None,
    noswitch: bool = False,
) -> pd.DataFrame:
    """
    Vary number of distinct training series M while holding steps-per-series
    constant (each series seen exactly once).

    noswitch: if True, train with single-regime series and evaluate on
      DATASETS_NOSWITCH. pool_dir is ignored (always on-the-fly with
      force_no_switch=True since no NS pools are pre-generated).
    """
    tag      = "D_NS" if noswitch else "D"
    datasets = DATASETS_NOSWITCH if noswitch else DATASETS_B1
    monitor  = VAL_MONITOR_NS if noswitch else VAL_MONITOR_B1CDE
    prefix   = "exp_d_ns" if noswitch else "exp_d"

    print("\n" + "="*60)
    print(f"EXPERIMENT {tag}: Task diversity sweep (Raventós replication)")
    print(f"  n_instances={n_instances}  noswitch={noswitch}")
    print(f"  Steps = M // batch_size (each series seen exactly once)")
    print("="*60)

    context_len = 64
    batch_size  = 128
    lr          = 3e-4
    val_frac    = 0.3

    M_values = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768,
                65536, 131072, 262144, 524288]

    val_loader = get_val_monitor_loader(data_dir, context_len, val_frac, batch_size, monitor)
    rows = []

    for M in M_values:
        steps = max(1, M // batch_size)
        total_series = steps * batch_size
        print(f"\n--- M={M:,} distinct series  steps={steps:,} ---")
        torch.manual_seed(seed)
        np.random.seed(seed)

        model   = build_model(context_len, 256, 4, 6, 0.1, seed, device)
        pool_path_d = None
        if pool_dir is not None:
            candidate = Path(pool_dir) / f"pool_d_full_{M}.npz"
            if candidate.exists():
                pool_path_d = str(candidate)
        sampler = build_sampler(
            ar_coeff_scale=1.2, seed=seed,
            ar_order_lo=2, ar_order_hi=2,
            pool_path=pool_path_d,
            family_weights=FAMILY_PRESETS["full"],
            force_no_switch=noswitch,
        )

        train_iid(model, sampler, val_loader, steps, batch_size, lr, device)

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

        print(f"  mean_all={results['mean_all']:.4f}  mean_ar={results['mean_ar']:.4f}")
        if "mean_gap_vs_msar" in results:
            print(f"  mean_gap_vs_msar={results['mean_gap_vs_msar']:.4f}")

        if wandb_run is not None:
            wandb_run.log({
                f"{prefix}/M":        M,
                f"{prefix}/steps":    steps,
                f"{prefix}/mean_all": results["mean_all"],
                f"{prefix}/mean_ar":  results["mean_ar"],
                **({f"{prefix}/mean_gap_vs_msar": results["mean_gap_vs_msar"]}
                   if "mean_gap_vs_msar" in results else {}),
            })

        row = {"M": M, "steps": steps, "total_series": total_series}
        row.update({k: v for k, v in results.items() if isinstance(v, float)})
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"\nExperiment {tag} summary:")
    cols = ["M", "steps", "mean_all", "mean_ar"]
    available = [c for c in cols if c in df.columns]
    print(df[available].to_string(index=False))
    return df


# ================================================================
# EXPERIMENT E — Task diversity × model class (2D sweep)
# ================================================================

def run_experiment_e(
    data_dir: str,
    device: torch.device,
    msar_df,
    n_instances: int = 3,
    seed: int = 0,
    wandb_run=None,
    pool_dir_full: str = None,
    pool_dir_ar_only: str = None,
    noswitch: bool = False,
) -> pd.DataFrame:
    """
    Run Experiment D (pool size sweep) separately for ar_only and full
    family presets.

    noswitch: if True, train with single-regime series and evaluate on
      DATASETS_NOSWITCH. Pool dirs are ignored (always on-the-fly with
      force_no_switch=True).
    """
    tag      = "E_NS" if noswitch else "E"
    datasets = DATASETS_NOSWITCH if noswitch else DATASETS_B1
    monitor  = VAL_MONITOR_NS if noswitch else VAL_MONITOR_B1CDE
    prefix   = "exp_e_ns" if noswitch else "exp_e"

    print("\n" + "="*60)
    print(f"EXPERIMENT {tag}: Task diversity x model class (2D sweep)")
    print(f"  n_instances={n_instances}  noswitch={noswitch}")
    print(f"  Conditions: ar_only vs full family preset")
    print(f"  Steps = M // batch_size (each series seen exactly once)")
    print("="*60)

    context_len = 64
    batch_size  = 128
    lr          = 3e-4
    val_frac    = 0.3

    M_values = [128, 256, 512, 1024, 2048, 4096, 8192, 16384,
                32768, 65536, 131072, 262144, 524288]

    val_loader = get_val_monitor_loader(data_dir, context_len, val_frac, batch_size, monitor)
    rows = []

    for preset_name in ["ar_only", "full"]:
        weights = FAMILY_PRESETS[preset_name]
        print(f"\n{'='*40}")
        print(f"Family preset: {preset_name}")
        print(f"{'='*40}")

        for M in M_values:
            steps = max(1, M // batch_size)
            print(f"\n--- {preset_name}  M={M:,}  steps={steps:,} ---")
            torch.manual_seed(seed)
            np.random.seed(seed)

            model   = build_model(context_len, 256, 4, 6, 0.1, seed, device)
            pool_path_e = None
            if preset_name == "full" and pool_dir_full is not None:
                candidate = Path(pool_dir_full) / f"pool_d_full_{M}.npz"
                if candidate.exists():
                    pool_path_e = str(candidate)
            elif preset_name == "ar_only" and pool_dir_ar_only is not None:
                candidate = Path(pool_dir_ar_only) / f"pool_e_ar_only_{M}.npz"
                if candidate.exists():
                    pool_path_e = str(candidate)
            order_hi = 2  # hold order fixed at AR(2) for all E conditions
            sampler = build_sampler(
                ar_coeff_scale=1.2, seed=seed,
                ar_order_lo=2, ar_order_hi=order_hi,
                pool_path=pool_path_e,
                family_weights=weights,
                force_no_switch=noswitch,
            )

            train_iid(model, sampler, val_loader, steps, batch_size, lr, device)

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

            print(f"  mean_all={results['mean_all']:.4f}  mean_ar={results['mean_ar']:.4f}")

            if wandb_run is not None:
                wandb_run.log({
                    f"{prefix}/{preset_name}/M":        M,
                    f"{prefix}/{preset_name}/steps":    steps,
                    f"{prefix}/{preset_name}/mean_all": results["mean_all"],
                    f"{prefix}/{preset_name}/mean_ar":  results["mean_ar"],
                    **({f"{prefix}/{preset_name}/mean_gap_vs_msar": results["mean_gap_vs_msar"]}
                       if "mean_gap_vs_msar" in results else {}),
                })

            row = {
                "family_preset": preset_name,
                "M": M,
                "steps": steps,
                "total_series": steps * batch_size,
            }
            row.update({k: v for k, v in results.items() if isinstance(v, float)})
            rows.append(row)

    df = pd.DataFrame(rows)
    print(f"\nExperiment {tag} summary:")
    cols = ["family_preset", "M", "steps", "mean_all", "mean_ar"]
    available = [c for c in cols if c in df.columns]
    print(df[available].to_string(index=False))
    return df

def main():
    ap = argparse.ArgumentParser(
        description="Data density experiments."
    )
    ap.add_argument(
        "--experiments", nargs="+", default=["B1", "B2", "B3", "C", "D", "E"],
        choices=["B1", "B2", "B3", "C", "D", "E"],
    )
    ap.add_argument("--data_dir",           type=str, default="generated_data")
    ap.add_argument("--noswitch",           action="store_true",
                    help="Run all experiments with single-regime (no-switch) training and eval. "
                         "Uses data_dir_noswitch and pool_noswitch.")
    ap.add_argument("--data_dir_noswitch",  type=str, default="generated_data_noswitch",
                    help="Directory with no-switch eval datasets (used when --noswitch).")
    ap.add_argument("--pool_noswitch",      type=str, default=None,
                    help="No-switch training pool for Experiment C when --noswitch.")
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
    ap.add_argument("--pool_b2_lo_order",  type=str, default=None)
    ap.add_argument("--pool_b2_mid_order", type=str, default=None)
    ap.add_argument("--pool_b2_hi_order",  type=str, default=None)
    ap.add_argument("--pool_b2_full",      type=str, default=None)
    ap.add_argument("--pool_b3_0_1",  type=str, default=None)
    ap.add_argument("--pool_b3_0_2",  type=str, default=None)
    ap.add_argument("--pool_b3_0_3",  type=str, default=None)
    ap.add_argument("--pool_b3_0_4",  type=str, default=None)
    ap.add_argument("--pool_b3_0_5",  type=str, default=None)
    ap.add_argument("--pool_b3_0_6",  type=str, default=None)
    ap.add_argument("--pool_b3_0_8",  type=str, default=None)
    ap.add_argument("--pool_b3_1_0",  type=str, default=None)
    ap.add_argument("--pool_b3_1_2",  type=str, default=None)
    ap.add_argument("--pool_d_dir",         type=str, default=None,
                    help="Directory with pool_d_full_{M}.npz files for Exp D.")
    ap.add_argument("--pool_e_dir_full",    type=str, default=None,
                    help="Directory with pool_d_full_{M}.npz files for Exp E full.")
    ap.add_argument("--pool_e_dir_ar_only", type=str, default=None,
                    help="Directory with pool_e_ar_only_{M}.npz files for Exp E.")
    ap.add_argument("--msar_csv",      type=str,   default="msar_results.csv")
    ap.add_argument("--n_instances",   type=int,   default=3)
    ap.add_argument("--seed",          type=int,   default=0)
    ap.add_argument("--no_wandb",      action="store_true")
    ap.add_argument("--wandb_project", type=str,   default="switch-transformers")
    ap.add_argument("--exp_b_steps",   type=int,   default=10000,
                    help="Steps for B1/B2/B3 (default 25000, based on convergence from W&B).")
    args = ap.parse_args()

    device = resolve_device("cuda")

    # Resolve data dir and msar based on --noswitch flag
    data_dir = args.data_dir_noswitch if args.noswitch else args.data_dir

    # Load MSAR results (not available for no-switch data)
    msar_df = None
    if not args.noswitch:
        if Path(args.msar_csv).exists():
            msar_df = pd.read_csv(args.msar_csv).set_index("dataset")
            print(f"Loaded MSAR results from {args.msar_csv} ({len(msar_df)} datasets)")
        else:
            print(f"[warning] {args.msar_csv} not found — gap vs MSAR will not be computed")

    # Check evaluation datasets exist
    check_datasets = DATASETS_NOSWITCH if args.noswitch else DATASETS
    if any(e in args.experiments for e in {"B1", "B2", "B3", "C", "D", "E"}):
        missing = [
            ds for ds in check_datasets for i in range(args.n_instances)
            if not (Path(data_dir) / f"{ds}_r{i}.npz").exists()
        ]
        if missing:
            gen_cmd = "python generate_noswitch_data.py" if args.noswitch else "python data_generation.py"
            print(f"[error] {len(missing)} dataset files missing in {data_dir}. Run: {gen_cmd}")
            return

    # Check pool for Experiment C
    if "C" in args.experiments:
        if args.noswitch and args.pool_noswitch is None:
            print("[error] Experiment C with --noswitch requires --pool_noswitch pool_noswitch.npz")
            return
        if not args.noswitch and args.pool_path is None:
            print("[error] Experiment C requires --pool_path series_pool.npz")
            return

    # Initialise W&B
    wandb_run = None
    if not args.no_wandb:
        try:
            import wandb, os
            os.environ["WANDB_MODE"] = "offline"
            run_name = "density_experiments_ns" if args.noswitch else "density_experiments"
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=run_name,
                config={
                    "experiments":  args.experiments,
                    "noswitch":     args.noswitch,
                    "pool_path":    args.pool_noswitch if args.noswitch else args.pool_path,
                    "n_instances":  args.n_instances,
                    "exp_b_steps":  args.exp_b_steps,
                },
            )
            print("W&B run initialised (offline mode)\n")
        except Exception as e:
            print(f"[warning] W&B init failed ({e}). Continuing without logging.")

    ns = args.noswitch
    suffix = "_ns" if ns else ""
    saved = []

    if "B1" in args.experiments:
        b1_pools = {}
        if not ns:
            if args.pool_b1_ar_only:       b1_pools["ar_only"]       = args.pool_b1_ar_only
            if args.pool_b1_ar_arma:       b1_pools["ar_arma"]       = args.pool_b1_ar_arma
            if args.pool_b1_ar_arma_arima: b1_pools["ar_arma_arima"] = args.pool_b1_ar_arma_arima
            if args.pool_b1_full:          b1_pools["full"]          = args.pool_b1_full

        df = run_experiment_b1(
            data_dir=data_dir, device=device, msar_df=msar_df,
            steps=args.exp_b_steps, n_instances=args.n_instances,
            seed=args.seed, wandb_run=wandb_run,
            b1_pools=b1_pools if b1_pools else None,
            noswitch=ns,
        )
        fname = f"results_density_exp_b1{suffix}.csv"
        df.to_csv(fname, index=False)
        saved.append(fname)

    if "B2" in args.experiments:
        b2_pools = {}
        if not ns:
            if args.pool_b2_lo_order:  b2_pools["lo_order"]  = args.pool_b2_lo_order
            if args.pool_b2_mid_order: b2_pools["mid_order"] = args.pool_b2_mid_order
            if args.pool_b2_hi_order:  b2_pools["hi_order"]  = args.pool_b2_hi_order
            if args.pool_b2_full:      b2_pools["full"]      = args.pool_b2_full
        df = run_experiment_b2(
            data_dir=data_dir, device=device, msar_df=msar_df,
            steps=args.exp_b_steps, n_instances=args.n_instances,
            seed=args.seed, wandb_run=wandb_run,
            b2_pools=b2_pools if b2_pools else None,
            noswitch=ns,
        )
        fname = f"results_density_exp_b2{suffix}.csv"
        df.to_csv(fname, index=False)
        saved.append(fname)

    if "B3" in args.experiments:
        b3_pools = {}
        if not ns:
            if args.pool_b3_0_1: b3_pools["0.1"] = args.pool_b3_0_1
            if args.pool_b3_0_2: b3_pools["0.2"] = args.pool_b3_0_2
            if args.pool_b3_0_3: b3_pools["0.3"] = args.pool_b3_0_3
            if args.pool_b3_0_4: b3_pools["0.4"] = args.pool_b3_0_4
            if args.pool_b3_0_5: b3_pools["0.5"] = args.pool_b3_0_5
            if args.pool_b3_0_6: b3_pools["0.6"] = args.pool_b3_0_6
            if args.pool_b3_0_8: b3_pools["0.8"] = args.pool_b3_0_8
            if args.pool_b3_1_0: b3_pools["1.0"] = args.pool_b3_1_0
            if args.pool_b3_1_2: b3_pools["1.2"] = args.pool_b3_1_2
        df = run_experiment_b3(
            data_dir=data_dir, device=device, msar_df=msar_df,
            steps=args.exp_b_steps, n_instances=args.n_instances,
            seed=args.seed, wandb_run=wandb_run,
            b3_pools=b3_pools if b3_pools else None,
            noswitch=ns,
        )
        fname = f"results_density_exp_b3{suffix}.csv"
        df.to_csv(fname, index=False)
        saved.append(fname)

    if "C" in args.experiments:
        pool_c = args.pool_noswitch if ns else args.pool_path
        df = run_experiment_c(
            data_dir=data_dir, device=device, msar_df=msar_df,
            pool_path=pool_c,
            n_instances=args.n_instances,
            seed=args.seed, wandb_run=wandb_run,
            noswitch=ns,
        )
        fname = f"results_density_exp_c{suffix}.csv"
        df.to_csv(fname, index=False)
        saved.append(fname)

    if "D" in args.experiments:
        df = run_experiment_d(
            data_dir=data_dir, device=device, msar_df=msar_df,
            n_instances=args.n_instances,
            seed=args.seed, wandb_run=wandb_run,
            pool_dir=args.pool_d_dir,
            noswitch=ns,
        )
        fname = f"results_density_exp_d{suffix}.csv"
        df.to_csv(fname, index=False)
        saved.append(fname)

    if "E" in args.experiments:
        df = run_experiment_e(
            data_dir=data_dir, device=device, msar_df=msar_df,
            n_instances=args.n_instances,
            seed=args.seed, wandb_run=wandb_run,
            pool_dir_full=args.pool_e_dir_full or args.pool_d_dir,
            pool_dir_ar_only=args.pool_e_dir_ar_only,
            noswitch=ns,
        )
        fname = f"results_density_exp_e{suffix}.csv"
        df.to_csv(fname, index=False)
        saved.append(fname)

    if wandb_run is not None:
        wandb_run.finish()
        print("\nSync W&B with: wandb sync wandb/offline-run-<id>")

    print("\n=== All density experiments complete ===")
    for f in saved:
        print(f"  {f}")


if __name__ == "__main__":
    main()
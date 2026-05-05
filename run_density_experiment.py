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

# ----------------------------------------------------------------
# Small GPT-style transformer for Experiment A.
# Accepts (B, L, token_dim) input — needed for d>1 regression.
# ----------------------------------------------------------------

class ICLTransformer(torch.nn.Module):
    """
    GPT-style decoder-only transformer for ICL regression.
    Follows Raventós et al. (2023): interleaved (x, y) token sequences.
    """
    def __init__(self, token_dim: int, d_model: int = 256,
                 n_heads: int = 8, n_layers: int = 12,
                 max_seq_len: int = 256, dropout: float = 0.0):
        super().__init__()
        self.token_dim = token_dim
        self.in_proj   = torch.nn.Linear(token_dim, d_model)
        self.pos_emb   = torch.nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=4*d_model, dropout=dropout,
            batch_first=True, activation="gelu", norm_first=True,
        )
        self.decoder  = torch.nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out_proj = torch.nn.Linear(d_model, 1)
        torch.nn.init.normal_(self.pos_emb, std=0.02)

    def _causal_mask(self, L, device):
        return torch.triu(
            torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, L, _ = tokens.shape
        h = self.in_proj(tokens) + self.pos_emb[:, :L, :]
        h = self.decoder(h, mask=self._causal_mask(L, tokens.device))
        return self.out_proj(h)  # (B, L, 1)

    def predict_last(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.forward(tokens)[:, -1, :]  # (B, 1)


def build_icl_sequence(betas, n_examples, noise_sigma, d, rng):
    """
    Build interleaved ICL sequence following Raventós et al. (2023).
    Format: [x_1, y_1_pad, x_2, y_2_pad, ..., x_n, y_n_pad, x_query]
    x tokens: d-dimensional. y tokens: d-dim, value in last position.
    """
    B = betas.shape[0]
    x_ic   = rng.standard_normal((B, n_examples, d)).astype(np.float32)
    noise  = rng.normal(0, noise_sigma, (B, n_examples)).astype(np.float32)
    y_ic   = np.einsum("bd,bnd->bn", betas, x_ic) + noise  # (B, n)
    x_q    = rng.standard_normal((B, 1, d)).astype(np.float32)
    noise_q = rng.normal(0, noise_sigma, (B,)).astype(np.float32)
    y_q    = np.einsum("bd,bd->b", betas, x_q[:, 0, :]) + noise_q

    tokens = np.zeros((B, 2 * n_examples + 1, d), dtype=np.float32)
    for i in range(n_examples):
        tokens[:, 2*i,   :] = x_ic[:, i, :]
        tokens[:, 2*i+1, d-1] = y_ic[:, i]
    tokens[:, -1, :] = x_q[:, 0, :]
    return tokens, y_ic, y_q


def run_experiment_a(
    device, seed=0, d=10, context_len=40,
    steps=50000, batch_size=64, wandb_run=None,
) -> pd.DataFrame:
    """
    Replication of Raventós et al. (2023).

    Setup matching the paper:
      - d=10 dimensional regression: y = beta @ x + N(0, 0.1)
      - beta ~ N(0, I_d), same prior as Raventós
      - Interleaved sequence: [x1, y1, x2, y2, ..., xn, yn, x_query]
      - context_len=40 in-context examples (matches Raventós n=40)
      - GPT transformer: d_model=256, 8 heads, 12 layers
      - Vary M (distinct beta vectors) from 4 to 16384
      - Evaluate OOD RMSE on fresh betas never seen in training

    Expected result (Raventós Fig 2):
      Below M*: OOD RMSE >> noise floor (transformer specialised to training tasks)
      Above M*: OOD RMSE approaches noise floor (learned general algorithm)
    """
    print("\n" + "="*60)
    print("EXPERIMENT A: Linear regression (Raventós et al. 2023 replication)")
    print(f"  d={d}  context_len={context_len}  steps={steps}")
    print(f"  sequence_len={2*context_len+1}  model: d_model=256, 8 heads, 12 layers")
    print("="*60)

    M_values   = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096, 16384]
    noise_sigma = 0.1
    seq_len    = 2 * context_len + 1
    rows       = []
    np_rng     = np.random.default_rng(seed)

    for M in M_values:
        print(f"\n--- M={M} distinct beta vectors ---")
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)

        beta_pool = rng.standard_normal((M, d)).astype(np.float32)

        model = ICLTransformer(
            token_dim=d, d_model=256, n_heads=8, n_layers=12,
            max_seq_len=seq_len + 10, dropout=0.0,
        ).to(device)
        model.train()
        opt   = torch.optim.AdamW(model.parameters(), lr=1e-4)
        loss_fn = torch.nn.MSELoss()
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

        t0 = time.time()
        for step in range(steps):
            B    = batch_size
            idx  = rng.integers(0, M, size=B)
            betas = beta_pool[idx]
            tokens, y_ic, _ = build_icl_sequence(
                betas, context_len, noise_sigma, d, rng)

            tokens_t = torch.from_numpy(tokens).to(device)
            y_ic_t   = torch.from_numpy(y_ic).to(device)
            y_pos    = list(range(1, 2*context_len, 2))

            opt.zero_grad(set_to_none=True)
            preds    = model(tokens_t)
            loss     = loss_fn(preds[:, y_pos, 0], y_ic_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

            if step % 500 == 0 and wandb_run is not None:
                wandb_run.log({
                    f"exp_a/M{M}/loss": float(loss.item()),
                    f"exp_a/M{M}/step": step,
                })

        elapsed = time.time() - t0

        # Evaluate on OOD betas never seen during training
        model.eval()
        n_test = 512
        rng_test   = np.random.default_rng(seed + 9999)
        test_betas = rng_test.standard_normal((n_test, d)).astype(np.float32)
        tokens_test, _, y_q_test = build_icl_sequence(
            test_betas, context_len, noise_sigma, d, rng_test)

        tokens_t = torch.from_numpy(tokens_test).to(device)
        y_true   = torch.from_numpy(y_q_test).to(device).unsqueeze(1)
        with torch.no_grad():
            yhat = model.predict_last(tokens_t)
        ood_rmse  = float(torch.sqrt(
            torch.nn.functional.mse_loss(yhat, y_true)).item())
        ridge_rmse = noise_sigma * float(np.sqrt(1 + d / (context_len + d)))
        ratio = ood_rmse / noise_sigma

        print(f"  OOD RMSE: {ood_rmse:.4f}  noise floor: {noise_sigma:.3f}  "
              f"ridge approx: {ridge_rmse:.3f}  ratio: {ratio:.2f}x  [{elapsed:.0f}s]")

        rows.append({
            "M": M, "ood_rmse": ood_rmse,
            "noise_floor": noise_sigma,
            "ridge_rmse":  ridge_rmse,
            "ratio_to_noise": ratio,
            "steps": steps, "d": d,
        })

        if wandb_run is not None:
            wandb_run.log({
                "exp_a/M":              M,
                "exp_a/ood_rmse":       ood_rmse,
                "exp_a/ridge_rmse":     ridge_rmse,
                "exp_a/ratio_to_noise": ratio,
            })

    df = pd.DataFrame(rows)
    print("\nExperiment A summary:")
    print(df[["M", "ood_rmse", "ridge_rmse", "ratio_to_noise"]].to_string(index=False))
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
    b2_pools: dict = None,
) -> pd.DataFrame:
    """
    Restrict training AR order; test on H1 (AR(10)) vs A1-A3 (AR(2)).
    b2_pools: optional dict mapping order_name to pool path.
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
        pool_for_order = (b2_pools or {}).get(name, None)
        sampler = build_sampler(
            ar_coeff_scale=0.6,
            ar_order_lo=lo, ar_order_hi=hi,
            seed=seed, pool_path=pool_for_order,
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
    b3_pools: dict = None,
) -> pd.DataFrame:
    """
    Vary ar_coeff_scale within the AR family.
    b3_pools: optional dict mapping scale string to pool path.
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
        pool_for_scale = (b3_pools or {}).get(str(scale), None)
        sampler = build_sampler(
            ar_coeff_scale=scale, seed=seed,
            pool_path=pool_for_scale,
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

    step_counts = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 25000, 50000, 100000]
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
) -> pd.DataFrame:
    """
    Vary number of distinct training series M while holding steps-per-series
    constant. This directly replicates the Raventós M-axis in a time series
    setting.

    Raventós (2023) varied M (number of distinct beta vectors) and found a
    critical threshold M* below which the transformer acts like the Bayesian
    estimator over M training tasks, and above which it learns a general
    algorithm matching ridge regression.

    Here M is the pool size — the number of distinct parameter combinations
    (AR coefficients, noise levels, regime sequences) available during training.
    Steps are set to M // batch_size so each series is seen exactly once
    per run, keeping steps-per-series = 1 across all M values.

    If a Raventós-style phase transition exists in the time series setting,
    we expect to see a critical M* below which performance is poor (transformer
    specialised to training parameter combinations) and above which it
    generalises to the evaluation datasets.

    Note: each M value requires generating a fresh pool of that size on-the-fly
    since the pre-generated pools have fixed size. This is slower but ensures
    each series is genuinely distinct.
    """
    print("\n" + "="*60)
    print("EXPERIMENT D: Task diversity sweep (Raventós replication)")
    print(f"  n_instances={n_instances}")
    print(f"  Steps = M // batch_size (each series seen exactly once)")
    print("="*60)

    context_len = 64
    batch_size  = 128
    lr          = 3e-4
    val_frac    = 0.3

    # M values — log-spaced from very small to large
    # At M=128: steps = 1  (essentially no training)
    # At M=500k: steps = 3906 (full pool, matches main run_compare)
    M_values = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768,
                65536, 131072, 262144, 500000]

    val_loader = get_val_monitor_loader(data_dir, context_len, val_frac, batch_size)
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
            ar_coeff_scale=0.6, seed=seed,
            pool_path=pool_path_d,
            family_weights=FAMILY_PRESETS["full"],
        )

        train_iid(model, sampler, val_loader, steps, batch_size, lr, device)

        results = eval_suite(
            model, data_dir, DATASETS, n_instances,
            context_len, val_frac, batch_size, device,
        )

        if msar_df is not None:
            gaps = [
                results[ds] - float(msar_df.loc[ds, "msar_val_rmse"])
                for ds in DATASETS
                if ds in results and ds in msar_df.index
                and not np.isnan(float(msar_df.loc[ds, "msar_val_rmse"]))
            ]
            results["mean_gap_vs_msar"] = float(np.mean(gaps)) if gaps else float("nan")

        print(f"  mean_all={results['mean_all']:.4f}  "
              f"mean_ar={results['mean_ar']:.4f}  "
              f"mean_arima={results['mean_arima']:.4f}  "
              f"mean_seasonal={results['mean_seasonal']:.4f}")
        if "mean_gap_vs_msar" in results:
            print(f"  mean_gap_vs_msar={results['mean_gap_vs_msar']:.4f}")

        if wandb_run is not None:
            wandb_run.log({
                "exp_d/M":             M,
                "exp_d/steps":         steps,
                "exp_d/mean_all":      results["mean_all"],
                "exp_d/mean_ar":       results["mean_ar"],
                "exp_d/mean_arima":    results["mean_arima"],
                "exp_d/mean_seasonal": results["mean_seasonal"],
                **({
                    "exp_d/mean_gap_vs_msar": results["mean_gap_vs_msar"]
                } if "mean_gap_vs_msar" in results else {}),
            })

        row = {"M": M, "steps": steps, "total_series": total_series}
        row.update({k: v for k, v in results.items() if isinstance(v, float)})
        rows.append(row)

    df = pd.DataFrame(rows)
    print("\nExperiment D summary:")
    cols = ["M", "steps", "mean_all", "mean_ar", "mean_arima", "mean_seasonal"]
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
) -> pd.DataFrame:
    """
    Run Experiment D (pool size sweep) separately for ar_only and full
    family presets. This asks whether the Raventós M* threshold depends
    on which process families are in the training pool.

    Two conditions on the M axis:
      ar_only: transformer only trained on AR switching dynamics
      full:    transformer trained on all 10 process families

    If the curves converge at high M: with enough distinct examples the
    transformer can generalise even from a restricted training distribution —
    quantity of data overcomes quality of coverage.

    If the curves stay separated at high M: model class coverage is a
    fundamental constraint that more data cannot overcome — the transformer
    cannot learn ARIMA or seasonal dynamics if it has never seen them,
    regardless of how many AR examples it has seen.

    Steps = M // batch_size so each series is seen exactly once per run,
    matching Experiment D and the Raventós setup.
    """
    print("\n" + "="*60)
    print("EXPERIMENT E: Task diversity x model class (2D sweep)")
    print(f"  n_instances={n_instances}")
    print(f"  Conditions: ar_only vs full family preset")
    print(f"  Steps = M // batch_size (each series seen exactly once)")
    print("="*60)

    context_len = 64
    batch_size  = 128
    lr          = 3e-4
    val_frac    = 0.3

    M_values = [128, 256, 512, 1024, 2048, 4096, 8192, 16384,
                32768, 65536, 131072, 262144, 500000]

    val_loader = get_val_monitor_loader(data_dir, context_len, val_frac, batch_size)
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
            sampler = build_sampler(
                ar_coeff_scale=0.6, seed=seed,
                pool_path=pool_path_e,
                family_weights=weights,
            )

            train_iid(model, sampler, val_loader, steps, batch_size, lr, device)

            results = eval_suite(
                model, data_dir, DATASETS, n_instances,
                context_len, val_frac, batch_size, device,
            )

            if msar_df is not None:
                gaps = [
                    results[ds] - float(msar_df.loc[ds, "msar_val_rmse"])
                    for ds in DATASETS
                    if ds in results and ds in msar_df.index
                    and not np.isnan(float(msar_df.loc[ds, "msar_val_rmse"]))
                ]
                results["mean_gap_vs_msar"] = float(np.mean(gaps)) if gaps else float("nan")

            print(f"  mean_all={results['mean_all']:.4f}  "
                  f"mean_ar={results['mean_ar']:.4f}  "
                  f"mean_arima={results['mean_arima']:.4f}  "
                  f"mean_seasonal={results['mean_seasonal']:.4f}")

            if wandb_run is not None:
                wandb_run.log({
                    f"exp_e/{preset_name}/M":             M,
                    f"exp_e/{preset_name}/steps":         steps,
                    f"exp_e/{preset_name}/mean_all":      results["mean_all"],
                    f"exp_e/{preset_name}/mean_ar":       results["mean_ar"],
                    f"exp_e/{preset_name}/mean_arima":    results["mean_arima"],
                    f"exp_e/{preset_name}/mean_seasonal": results["mean_seasonal"],
                    **({
                        f"exp_e/{preset_name}/mean_gap_vs_msar": results["mean_gap_vs_msar"]
                    } if "mean_gap_vs_msar" in results else {}),
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
    print("\nExperiment E summary:")
    cols = ["family_preset", "M", "steps", "mean_all", "mean_ar",
            "mean_arima", "mean_seasonal"]
    available = [c for c in cols if c in df.columns]
    print(df[available].to_string(index=False))
    return df

def main():
    ap = argparse.ArgumentParser(
        description="Data density experiments."
    )
    ap.add_argument(
        "--experiments", nargs="+", default=["A", "B1", "B2", "B3", "C", "D", "E"],
        choices=["A", "B1", "B2", "B3", "C", "D", "E"],
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
    ap.add_argument("--exp_a_steps",   type=int,   default=10000,
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
    ts_exps = {"B1", "B2", "B3", "C", "D", "E"}
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
        b2_pools = {}
        if args.pool_b2_lo_order:  b2_pools["lo_order"]  = args.pool_b2_lo_order
        if args.pool_b2_mid_order: b2_pools["mid_order"] = args.pool_b2_mid_order
        if args.pool_b2_hi_order:  b2_pools["hi_order"]  = args.pool_b2_hi_order
        if args.pool_b2_full:      b2_pools["full"]      = args.pool_b2_full
        df = run_experiment_b2(
            data_dir=args.data_dir, device=device, msar_df=msar_df,
            steps=args.exp_b_steps, n_instances=args.n_instances,
            seed=args.seed, wandb_run=wandb_run,
            b2_pools=b2_pools if b2_pools else None,
        )
        df.to_csv("results_density_exp_b2.csv", index=False)
        saved.append("results_density_exp_b2.csv")

    if "B3" in args.experiments:
        b3_pools = {}
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
            data_dir=args.data_dir, device=device, msar_df=msar_df,
            steps=args.exp_b_steps, n_instances=args.n_instances,
            seed=args.seed, wandb_run=wandb_run,
            b3_pools=b3_pools if b3_pools else None,
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

    if "D" in args.experiments:
        df = run_experiment_d(
            data_dir=args.data_dir, device=device, msar_df=msar_df,
            n_instances=args.n_instances,
            seed=args.seed, wandb_run=wandb_run,
            pool_dir=args.pool_d_dir,
        )
        df.to_csv("results_density_exp_d.csv", index=False)
        saved.append("results_density_exp_d.csv")

    if "E" in args.experiments:
        df = run_experiment_e(
            data_dir=args.data_dir, device=device, msar_df=msar_df,
            n_instances=args.n_instances,
            seed=args.seed, wandb_run=wandb_run,
            pool_dir_full=args.pool_e_dir_full or args.pool_d_dir,
            pool_dir_ar_only=args.pool_e_dir_ar_only,
        )
        df.to_csv("results_density_exp_e.csv", index=False)
        saved.append("results_density_exp_e.csv")

    if wandb_run is not None:
        wandb_run.finish()
        print("\nSync W&B with: wandb sync wandb/offline-run-<id>")

    print("\n=== All density experiments complete ===")
    for f in saved:
        print(f"  {f}")


if __name__ == "__main__":
    main()
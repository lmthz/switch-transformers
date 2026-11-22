"""
run_multisystem_train.py

Trains per-regime CausalTransformerForecasters for the thesis switch-filter
architecture. For each dataset and each regime k in {0, ..., K-1}, trains a
transformer on contiguous segments of the training split where states == k.
After training, estimates per-regime sigma_hat from val-split residuals (the
noise level the transformer achieves within that regime).

The resulting checkpoints are used by run_multisystem_eval.py:
  - transformer_log_lik(y_t, pred_t, sigma_hat) gives an approximate Gaussian
    log-likelihood that feeds into the Bayesian switch filter.

Usage (smoke test):
    python run_multisystem_train.py --dataset A1_ar2_coeffs_easy --device cpu --steps 200

Usage (full sweep):
    python run_multisystem_train.py --all_datasets
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from models.transformer_forecaster import CausalTransformerForecaster, TransformerConfig

# ── Dataset list ───────────────────────────────────────────────────────────────

DATASETS = [
    "A1_ar2_coeffs_easy", "A2_ar2_coeffs_hard",
    "A3_ar2_coeffs_plus_var",
    "B1_ar2_variance",    "B2_ar2_variance_big",
    "C1_arma21_coeffs_var",
    "D1_arima211",        "D2_arima221",         "D3_arima210",
    "E1_drift_only",      "E2_level_shift",
    "F1_seasonal_sarimax","F2_seasonal_exog",
    "G1_exogenous_only",
    "H1_ar10_coeffs",     "H2_ar1_near_unit_root",
    "S1_sparse_switching","S2_frequent_switching",
]

# ── Data helpers ───────────────────────────────────────────────────────────────

class RegimeDataset(Dataset):
    """Sliding-window dataset containing only windows from one regime."""

    def __init__(self, x: np.ndarray, y: np.ndarray):
        # x: (N, context_len), y: (N,)
        self.x = torch.from_numpy(x).unsqueeze(-1)   # (N, L, 1)
        self.y = torch.from_numpy(y)                  # (N,)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]



def extract_regime_windows(
    npz_path: str,
    regime: int,
    context_len: int,
    val_frac: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Returns:
      train_x    : (N_train, context_len) float32
      train_y    : (N_train,) float32  — next-step targets for each train window
      val_x      : (N_val, context_len) float32
      val_y      : (N_val,) float32
      stdzr_mean : float
      stdzr_std  : float
    """
    data = np.load(npz_path)
    y      = np.asarray(data["y"],      dtype=np.float64).flatten()
    states = np.asarray(data["states"], dtype=int).flatten()
    T      = len(y)
    n_train = int(T * (1.0 - val_frac))

    stdzr_mean = float(y[:n_train].mean())
    stdzr_std  = float(y[:n_train].std()) + 1e-8
    y_std = ((y - stdzr_mean) / stdzr_std).astype(np.float32)

    def _windows(y_region: np.ndarray, states_region: np.ndarray):
        # Include any window where the PREDICTION TARGET is in the desired regime.
        # The context may span a regime boundary; the transformer learns to predict
        # regime-k dynamics conditioned on whatever context precedes regime k.
        xs, ys = [], []
        for t in range(context_len, len(y_region)):
            if states_region[t] == regime:
                xs.append(y_region[t - context_len : t])
                ys.append(y_region[t])
        if not xs:
            return (np.empty((0, context_len), np.float32),
                    np.empty(0, np.float32))
        return np.stack(xs), np.array(ys, np.float32)

    train_x, train_y = _windows(y_std[:n_train], states[:n_train])
    val_x,   val_y   = _windows(y_std[n_train:], states[n_train:])
    return train_x, train_y, val_x, val_y, stdzr_mean, stdzr_std


# ── Sigma estimation ───────────────────────────────────────────────────────────

@torch.no_grad()
def estimate_sigma_hat(
    model: CausalTransformerForecaster,
    val_x: np.ndarray,
    val_y: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> float:
    """
    Compute std of last-position residuals on regime-specific val windows.
    sigma_hat is the transformer's within-regime noise level (standardized space).
    """
    if len(val_x) == 0:
        return 1.0

    dataset = RegimeDataset(val_x, val_y)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    residuals = []

    model.eval()
    for x, y in loader:
        x    = x.to(device)
        pred = model(x)[:, -1, 0].cpu().numpy()   # last-position prediction
        residuals.append(y.numpy() - pred)

    residuals_all = np.concatenate(residuals)
    return float(residuals_all.std()) + 1e-8


@torch.no_grad()
def estimate_val_mse(
    model: CausalTransformerForecaster,
    val_x: np.ndarray,
    val_y: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> float:
    if len(val_x) == 0:
        return float("nan")
    dataset = RegimeDataset(val_x, val_y)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    mses = []
    model.eval()
    for x, y in loader:
        x    = x.to(device)
        pred = model(x)[:, -1, 0].cpu().numpy()
        mses.append(((y.numpy() - pred) ** 2).mean())
    return float(np.mean(mses))


# ── Training ───────────────────────────────────────────────────────────────────

def _ckpt_path(ckpt_dir: str, dataset: str, regime: int) -> str:
    return str(Path(ckpt_dir) / f"{dataset}_regime{regime}_ckpt.pt")


def _final_path(ckpt_dir: str, dataset: str, regime: int) -> str:
    return str(Path(ckpt_dir) / f"{dataset}_regime{regime}_final.pt")


def train_one(
    npz_path: str,
    dataset: str,
    regime: int,
    ckpt_dir: str,
    context_len: int = 64,
    val_frac: float = 0.3,
    steps: int = 15000,
    batch_size: int = 128,
    lr: float = 3e-4,
    save_every: int = 500,
    device: torch.device = torch.device("cpu"),
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 4,
    dropout: float = 0.1,
) -> dict:
    """Train one per-regime transformer. Skip if final checkpoint exists."""
    final = _final_path(ckpt_dir, dataset, regime)
    if os.path.exists(final):
        print(f"  [skip] {dataset} regime={regime} — final model exists")
        return {"dataset": dataset, "regime": regime, "skipped": True}

    if not os.path.exists(npz_path):
        print(f"  [skip] {dataset} — {npz_path} not found")
        return {}

    train_x, train_y, val_x, val_y, stdzr_mean, stdzr_std = \
        extract_regime_windows(npz_path, regime, context_len, val_frac)

    if len(train_x) == 0:
        print(f"  [skip] {dataset} regime={regime} — no training windows found")
        return {}

    print(f"  [train] {dataset} regime={regime}  "
          f"train_windows={len(train_x)}  val_windows={len(val_x)}")

    cfg   = TransformerConfig(context_len=context_len, d_model=d_model,
                               n_heads=n_heads, n_layers=n_layers, dropout=dropout)
    model = CausalTransformerForecaster(cfg).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    train_loader = DataLoader(
        RegimeDataset(train_x, train_y),
        batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0,
    )

    start_step = 0
    best_val   = float("inf")
    ckpt       = _ckpt_path(ckpt_dir, dataset, regime)

    if os.path.exists(ckpt):
        saved = torch.load(ckpt, map_location=device)
        model.load_state_dict(saved["model_state"])
        opt.load_state_dict(saved["optimizer_state"])
        start_step = saved["step"]
        best_val   = saved["best_val"]
        print(f"  [resume] step={start_step}  best_val={best_val:.4f}")

    model.train()
    it   = iter(train_loader)
    pbar = tqdm(range(start_step, steps),
                desc=f"{dataset[:18]} r{regime}", leave=False)

    for step in pbar:
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(train_loader)
            x, y = next(it)

        x = x.to(device)
        y = y.to(device)

        # Dense next-step supervision at all L positions
        y_seq = torch.cat([x[:, 1:, :], y.unsqueeze(1).unsqueeze(-1)], dim=1)  # (B,L,1)

        opt.zero_grad(set_to_none=True)
        pred = model(x)                       # (B, L, 1)
        loss = loss_fn(pred, y_seq)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if (step + 1) % 200 == 0:
            val_mse = estimate_val_mse(model, val_x, val_y, device)
            pbar.set_postfix(train_loss=f"{loss.item():.4f}", val_mse=f"{val_mse:.4f}")
            if val_mse < best_val:
                best_val = val_mse
            model.train()

        if (step + 1) % save_every == 0:
            torch.save({
                "model_state":     model.state_dict(),
                "optimizer_state": opt.state_dict(),
                "step":            step + 1,
                "best_val":        best_val,
            }, ckpt)

    # Final metrics
    sigma_hat = estimate_sigma_hat(model, val_x, val_y, device)
    val_mse   = estimate_val_mse(model, val_x, val_y, device)

    torch.save({
        "model_state":  model.state_dict(),
        "cfg":          cfg.__dict__,
        "regime":       regime,
        "dataset":      dataset,
        "sigma_hat":    sigma_hat,
        "stdzr_mean":   stdzr_mean,
        "stdzr_std":    stdzr_std,
        "val_mse":      val_mse,
    }, final)

    if os.path.exists(ckpt):
        os.remove(ckpt)

    print(f"  [done]  {dataset} regime={regime}  "
          f"val_mse={val_mse:.4f}  sigma_hat={sigma_hat:.4f}")
    return {"dataset": dataset, "regime": regime,
            "val_mse": val_mse, "sigma_hat": sigma_hat}


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()

    tgt = ap.add_mutually_exclusive_group()
    tgt.add_argument("--dataset",      type=str)
    tgt.add_argument("--all_datasets", action="store_true")

    ap.add_argument("--data_dir",    type=str, default="generated_data")
    ap.add_argument("--instance",    type=int, default=0)
    ap.add_argument("--ckpt_dir",    type=str, default="runs/per_regime")
    ap.add_argument("--context_len", type=int, default=64)
    ap.add_argument("--val_frac",    type=float, default=0.3)
    ap.add_argument("--steps",       type=int, default=15000)
    ap.add_argument("--batch_size",  type=int, default=128)
    ap.add_argument("--lr",          type=float, default=3e-4)
    ap.add_argument("--save_every",  type=int, default=500)
    ap.add_argument("--d_model",     type=int, default=128)
    ap.add_argument("--n_heads",     type=int, default=4)
    ap.add_argument("--n_layers",    type=int, default=4)
    ap.add_argument("--dropout",     type=float, default=0.1)
    ap.add_argument("--device",      type=str, default="cuda")
    ap.add_argument("--seed",        type=int, default=42)

    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.ckpt_dir, exist_ok=True)

    datasets = DATASETS if args.all_datasets else [args.dataset]
    if not args.all_datasets and args.dataset is None:
        ap.error("Supply --dataset <name> or --all_datasets")

    train_kwargs = dict(
        ckpt_dir=args.ckpt_dir,
        context_len=args.context_len,
        val_frac=args.val_frac,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        save_every=args.save_every,
        device=device,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
    )

    print(f"Training {len(datasets)} datasets × 2 regimes | device={args.device} | steps={args.steps}")

    for ds in datasets:
        npz_path = str(Path(args.data_dir) / f"{ds}_r{args.instance}.npz")
        if not os.path.exists(npz_path):
            npz_path = str(Path(args.data_dir) / f"{ds}.npz")
        for regime in range(2):
            train_one(npz_path=npz_path, dataset=ds, regime=regime, **train_kwargs)

    print("Done.")


if __name__ == "__main__":
    main()

# train_transformer.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.synthetic_npz_dataset import make_train_val_datasets, SlidingWindowDataset
from data.msar_sampler import MSARBatchSampler, MSARSamplerConfig
from models.transformer_forecaster import TransformerConfig, CausalTransformerForecaster
from metrics import mse_rmse


# ============================================================
# Device
# ============================================================

def resolve_device(device_str: str) -> torch.device:
    if device_str == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            print("CUDA requested but not available. Falling back to CPU.")
            return torch.device("cpu")
    return torch.device("cpu")


# ============================================================
# Eval loop — always uses last position prediction
# ============================================================

@torch.no_grad()
def eval_loop(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    errs: List[np.ndarray] = []
    for x, y, _s in loader:
        x = x.to(device)
        y = y.to(device)
        # Use predict_last so eval works correctly in both dense and standard modes
        yhat = model.predict_last(x)
        e = (y - yhat).detach().cpu().numpy().reshape(-1)
        errs.append(e)
    model.train()
    if not errs:
        return float("nan"), float("nan")
    eall = np.concatenate(errs)
    return mse_rmse(eall)


# ============================================================
# Garg-style iid training
# ============================================================

def train_iid(
    model: CausalTransformerForecaster,
    sampler: MSARBatchSampler,
    val_loader: DataLoader,
    steps: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> None:
    """
    Train on fresh synthetic MSAR series every step.

    Standard mode: one prediction per window (last position), one MSE loss.

    Dense supervision mode: predict at every position simultaneously.
      y_seq is constructed from the context window itself:
        - positions 0..L-2: predict the next value already in the context
        - position L-1: predicts the true next value (same as standard target y)
      Loss = mean MSE across all L predictions.
      This gives L times more gradient signal per forward pass at no extra cost
      since the hidden states at all positions are already computed.
    """
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    context_len = model.cfg.context_len
    dense = model.cfg.dense_supervision

    pbar = tqdm(range(steps), desc="train (iid)")
    for step in pbar:
        x, y = sampler.sample_batch(batch_size, context_len, device)
        # x: (B, L, 1),  y: (B, 1)

        opt.zero_grad(set_to_none=True)

        if dense:
            # Build target sequence: position t predicts x[t+1] for t < L-1,
            # and the true next value y for t = L-1.
            # x[:, 1:, :] has shape (B, L-1, 1) — the known future values
            # y has shape (B, 1) — the unknown final target
            y_seq = torch.cat([x[:, 1:, :], y.unsqueeze(1)], dim=1)  # (B, L, 1)
            yhat = model(x)        # (B, L, 1)
            loss = loss_fn(yhat, y_seq)
        else:
            yhat = model(x)        # (B, 1)
            loss = loss_fn(yhat, y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 100 == 0 or step == steps - 1:
            _, rmse_v = eval_loop(model, val_loader, device=device)
            pbar.set_postfix(loss=float(loss.item()), val_rmse=float(rmse_v))


# ============================================================
# Fixed-dataset training (kept for comparison)
# ============================================================

def train_fixed(
    model: CausalTransformerForecaster,
    train_loader: DataLoader,
    val_loader: DataLoader,
    steps: int,
    lr: float,
    device: torch.device,
) -> None:
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    pbar = tqdm(range(steps), desc="train (fixed)")
    it = iter(train_loader)
    for step in pbar:
        try:
            x, y, _s = next(it)
        except StopIteration:
            it = iter(train_loader)
            x, y, _s = next(it)

        x = x.to(device)
        y = y.to(device)

        opt.zero_grad(set_to_none=True)
        yhat = model(x)
        loss = loss_fn(yhat, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 100 == 0 or step == steps - 1:
            _, rmse_v = eval_loop(model, val_loader, device=device)
            pbar.set_postfix(loss=float(loss.item()), val_rmse=float(rmse_v))


# ============================================================
# Main entry point
# ============================================================

def train_one_dataset(
    npz_path: str,
    context_len: int,
    val_frac: float,
    steps: int,
    batch_size: int,
    lr: float,
    d_model: int,
    n_heads: int,
    n_layers: int,
    dropout: float,
    seed: int,
    device: str,
    training_mode: str = "iid",
    ar_order: int = 2,
    dense_supervision: bool = False,
    ar_coeff_scale: float = 0.6,
) -> Dict[str, Any]:
    device = resolve_device(device)
    torch.manual_seed(seed)
    np.random.seed(seed)

    ds_train, ds_val, stdzr, _meta = make_train_val_datasets(
        npz_path=npz_path,
        context_len=context_len,
        val_frac=val_frac,
    )
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False)

    cfg = TransformerConfig(
        context_len=context_len,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout,
        dense_supervision=dense_supervision,
    )
    model = CausalTransformerForecaster(cfg).to(device)
    model.train()

    if training_mode == "iid":
        sampler_cfg = MSARSamplerConfig(
            series_len=max(512, context_len * 4),
            k_regimes=2,
            ar_coeff_scale=ar_coeff_scale,
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
        train_iid(model, sampler, val_loader, steps, batch_size, lr, device)
    else:
        train_loader = DataLoader(
            ds_train, batch_size=batch_size, shuffle=True, drop_last=True
        )
        train_fixed(model, train_loader, val_loader, steps, lr, device)

    train_loader_eval = DataLoader(ds_train, batch_size=batch_size, shuffle=False)
    _, train_rmse = eval_loop(model, train_loader_eval, device)
    _, val_rmse = eval_loop(model, val_loader, device)

    return {
        "train_rmse": float(train_rmse),
        "val_rmse": float(val_rmse),
        "std_used": float(stdzr.std),
        "mean_used": float(stdzr.mean),
        "n_train": int(len(ds_train)),
        "n_val": int(len(ds_val)),
        "training_mode": training_mode,
        "model_config": cfg.__dict__,
    }


# ============================================================
# CLI
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="generated_data")
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--context_len", type=int, default=64)
    ap.add_argument("--val_frac", type=float, default=0.3)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--n_layers", type=int, default=6)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--training_mode", type=str, default="iid", choices=["iid", "fixed"])
    ap.add_argument("--ar_order", type=int, default=2)
    ap.add_argument("--dense_supervision", action="store_true")
    ap.add_argument("--ar_coeff_scale", type=float, default=0.6)
    args = ap.parse_args()

    npz_path = str(Path(args.data_dir) / f"{args.dataset}.npz")
    out = train_one_dataset(
        npz_path=npz_path,
        context_len=args.context_len,
        val_frac=args.val_frac,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        seed=args.seed,
        device=args.device,
        training_mode=args.training_mode,
        ar_order=args.ar_order,
        dense_supervision=args.dense_supervision,
        ar_coeff_scale=args.ar_coeff_scale,
    )
    print(f"\ntransformer {args.dataset} [{out['training_mode']}]")
    print(f"train_rmse={out['train_rmse']:.4f}  val_rmse={out['val_rmse']:.4f}")


if __name__ == "__main__":
    main()
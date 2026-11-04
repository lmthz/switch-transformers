"""
run_detection_train.py

Trains a ComparisonDetectionTransformer to detect regime switches rapidly.

Two training variants (--variant):
  detection_only  :  L = MSE(sigmoid(score), detection_target)
                     Backbone learns purely from soft detection signal.
  multitask       :  L = MSE(yhat, y_seq) + 0.1 * MSE(sigmoid(score), detection_target)
                     Prediction loss teaches backbone regime dynamics;
                     detection head sharpens on top.

Both variants use soft decayed labels:  target[t] = exp(-0.5 * (t - switch))
Directly rewards firing fast — 1 step after switch earns more reward than 5 steps after.

Resumable:
  - Checkpoint saved every --save_every steps to runs/detection_{variant}/
  - Skips if {dataset}_final.pt already exists in that directory
  - Resumes from {dataset}_ckpt.pt if present

Usage (smoke test):
    python run_detection_train.py --dataset A1_ar2_coeffs_easy --variant multitask \\
        --steps 100 --device cpu

Usage (full sweep):
    python run_detection_train.py --all_datasets --variant multitask
    python run_detection_train.py --all_datasets --variant detection_only
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.transformer_forecaster import TransformerConfig
from models.detection_transformer import ComparisonDetectionTransformer
from data.detection_dataset import make_detection_datasets

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

DET_WEIGHT = 0.1   # detection loss weight in multitask variant
ALPHA      = 0.5   # decay rate for soft labels

# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def _ckpt_path(ckpt_dir: str, dataset: str) -> str:
    return str(Path(ckpt_dir) / f"{dataset}_ckpt.pt")

def _final_path(ckpt_dir: str, dataset: str) -> str:
    return str(Path(ckpt_dir) / f"{dataset}_final.pt")

def _save_checkpoint(path: str, model, optimizer, step: int, best_val: float) -> None:
    torch.save({
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "step":            step,
        "best_val":        best_val,
    }, path)

def _load_checkpoint(path: str, model, optimizer, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt["step"], ckpt["best_val"]

# ── Validation: mean detection latency at a fixed threshold ───────────────────

@torch.no_grad()
def eval_val_latency(
    model: ComparisonDetectionTransformer,
    val_loader: DataLoader,
    device,
    threshold: float = 0.5,
    max_delay: int = 50,
) -> dict:
    """
    Compute val detection loss and approximate mean detection latency.
    Latency is measured per-batch by counting how many positions back from
    the current step the score first exceeded threshold.  This is an
    approximation that avoids building a full sequential pass for every val
    batch; a proper sequential latency eval happens in run_detection_eval.py.
    """
    model.eval()
    loss_fn = nn.MSELoss()
    losses = []
    hits   = []   # was score >= threshold when target > 0.3?

    for x, y, det in val_loader:
        x   = x.to(device)
        det = det.to(device)
        y   = y.to(device)

        y_seq = torch.cat([x[:, 1:, :], y.unsqueeze(1)], dim=1)

        pred, score = model.forward_detection(x)
        score_sig   = torch.sigmoid(score).squeeze(-1)   # (B,)

        loss_det = loss_fn(score_sig, det)
        losses.append(loss_det.item())

        # Fraction of near-switch samples (target > 0.3) with score >= threshold
        near = (det > 0.3)
        if near.any():
            hits.append((score_sig[near] >= threshold).float().mean().item())

    model.train()
    mean_loss = float(np.mean(losses)) if losses else float("nan")
    hit_rate  = float(np.mean(hits))   if hits   else float("nan")
    return {"val_det_loss": mean_loss, "val_hit_rate": hit_rate}


# ── Training loop ──────────────────────────────────────────────────────────────

def train_one(
    npz_path: str,
    dataset: str,
    ckpt_dir: str,
    variant: str,
    context_len: int,
    val_frac: float,
    steps: int,
    batch_size: int,
    lr: float,
    save_every: int,
    device: torch.device,
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 4,
    dropout: float = 0.1,
    split_k: int = 8,
    det_weight: float = DET_WEIGHT,
    alpha: float = ALPHA,
) -> Dict:
    """Train ComparisonDetectionTransformer on one dataset, one variant."""
    final_path = _final_path(ckpt_dir, dataset)
    if os.path.exists(final_path):
        print(f"  [skip] {dataset} ({variant}) — final model exists")
        return {"dataset": dataset, "variant": variant, "skipped": True}

    if not os.path.exists(npz_path):
        print(f"  [skip] {dataset} — {npz_path} not found")
        return {}

    ds_train, ds_val, info = make_detection_datasets(
        npz_path, context_len=context_len, val_frac=val_frac, alpha=alpha
    )
    if len(ds_train) == 0 or len(ds_val) == 0:
        print(f"  [skip] {dataset} — insufficient data after split")
        return {}

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                              drop_last=True,  num_workers=0)
    val_loader   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False,
                              num_workers=0)

    cfg   = TransformerConfig(context_len=context_len, d_model=d_model,
                               n_heads=n_heads, n_layers=n_layers, dropout=dropout)
    model = ComparisonDetectionTransformer(cfg, split_k=split_k).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    start_step = 0
    best_val   = float("inf")
    ckpt_path  = _ckpt_path(ckpt_dir, dataset)

    if os.path.exists(ckpt_path):
        start_step, best_val = _load_checkpoint(ckpt_path, model, opt, device)
        print(f"  [resume] {dataset} ({variant}) from step {start_step} "
              f"(best val {best_val:.4f})")
    else:
        print(f"  [train]  {dataset} ({variant}) from scratch")

    model.train()
    it   = iter(train_loader)
    pbar = tqdm(range(start_step, steps), desc=f"{dataset[:20]} ({variant})", leave=False)

    for step in pbar:
        try:
            x, y, det = next(it)
        except StopIteration:
            it = iter(train_loader)
            x, y, det = next(it)

        x   = x.to(device)
        y   = y.to(device)
        det = det.to(device)

        # Dense next-step targets (B, L, 1)
        y_seq = torch.cat([x[:, 1:, :], y.unsqueeze(1)], dim=1)

        opt.zero_grad(set_to_none=True)
        pred, score = model.forward_detection(x)
        score_sig   = torch.sigmoid(score).squeeze(-1)   # (B,)

        loss_det = loss_fn(score_sig, det)

        if variant == "multitask":
            loss_pred = loss_fn(pred, y_seq)
            loss = loss_pred + det_weight * loss_det
        else:
            loss = loss_det

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if (step + 1) % 200 == 0:
            metrics = eval_val_latency(model, val_loader, device)
            vl = metrics["val_det_loss"]
            hr = metrics["val_hit_rate"]
            if variant == "multitask":
                pbar.set_postfix(loss=f"{loss.item():.4f}",
                                 loss_det=f"{loss_det.item():.4f}",
                                 val_det=f"{vl:.4f}", hit=f"{hr:.2f}")
            else:
                pbar.set_postfix(loss=f"{loss.item():.4f}",
                                 val_det=f"{vl:.4f}", hit=f"{hr:.2f}")
            if vl < best_val:
                best_val = vl

        if (step + 1) % save_every == 0:
            _save_checkpoint(ckpt_path, model, opt, step + 1, best_val)

    # Final eval and save
    metrics = eval_val_latency(model, val_loader, device)
    final_val = metrics["val_det_loss"]

    torch.save({
        "model_state":  model.state_dict(),
        "cfg":          cfg.__dict__,
        "split_k":      split_k,
        "variant":      variant,
        "val_det_loss": final_val,
        "val_hit_rate": metrics["val_hit_rate"],
        "dataset":      dataset,
        "stdzr_mean":   info["mean"],
        "stdzr_std":    info["std"],
        "alpha":        alpha,
    }, final_path)

    print(f"  [done]   {dataset} ({variant})  val_det_loss={final_val:.4f}  "
          f"hit_rate={metrics['val_hit_rate']:.3f}")

    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    return {"dataset": dataset, "variant": variant,
            "val_det_loss": final_val, "val_hit_rate": metrics["val_hit_rate"]}


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()

    tgt = ap.add_mutually_exclusive_group()
    tgt.add_argument("--dataset",      type=str,
                     help="Single dataset name (e.g. A1_ar2_coeffs_easy)")
    tgt.add_argument("--all_datasets", action="store_true",
                     help="Train on all datasets in DATASETS list")

    ap.add_argument("--variant",   type=str, default="multitask",
                    choices=["detection_only", "multitask"],
                    help="Loss variant: detection_only or multitask")
    ap.add_argument("--data_dir",  type=str, default="generated_data")
    ap.add_argument("--instance",  type=int, default=0,
                    help="Dataset instance suffix (default 0 → uses <dataset>_r0.npz)")
    ap.add_argument("--ckpt_dir",  type=str, default=None,
                    help="Checkpoint directory (default: runs/detection_<variant>)")

    ap.add_argument("--context_len", type=int,   default=64)
    ap.add_argument("--val_frac",    type=float, default=0.3)
    ap.add_argument("--steps",       type=int,   default=30000)
    ap.add_argument("--batch_size",  type=int,   default=128)
    ap.add_argument("--lr",          type=float, default=3e-4)
    ap.add_argument("--save_every",  type=int,   default=500)
    ap.add_argument("--d_model",     type=int,   default=128)
    ap.add_argument("--n_heads",     type=int,   default=4)
    ap.add_argument("--n_layers",    type=int,   default=4)
    ap.add_argument("--dropout",     type=float, default=0.1)
    ap.add_argument("--split_k",     type=int,   default=8,
                    help="Recent window size for comparison head")
    ap.add_argument("--det_weight",  type=float, default=DET_WEIGHT,
                    help="Detection loss weight in multitask variant")
    ap.add_argument("--alpha",       type=float, default=ALPHA,
                    help="Soft label decay rate")
    ap.add_argument("--seed",        type=int,   default=42)
    ap.add_argument("--device",      type=str,   default="cuda")

    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ckpt_dir = args.ckpt_dir or f"runs/detection_{args.variant}"
    os.makedirs(ckpt_dir, exist_ok=True)

    if args.all_datasets:
        datasets = DATASETS
    elif args.dataset is not None:
        datasets = [args.dataset]
    else:
        ap.error("Supply --dataset <name> or --all_datasets")

    train_kwargs = dict(
        ckpt_dir=ckpt_dir,
        variant=args.variant,
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
        split_k=args.split_k,
        det_weight=args.det_weight,
        alpha=args.alpha,
    )

    print(f"Training {len(datasets)} dataset(s)  |  "
          f"variant={args.variant}  device={args.device}  steps={args.steps}")

    for ds in datasets:
        npz_path = str(Path(args.data_dir) / f"{ds}_r{args.instance}.npz")
        train_one(npz_path=npz_path, dataset=ds, **train_kwargs)

    print("Done.")


if __name__ == "__main__":
    main()

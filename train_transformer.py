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
# Eval loop
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
        yhat = model(x)
        e = (y - yhat).detach().cpu().numpy().reshape(-1)
        errs.append(e)
    model.train()
    if not errs:
        return float("nan"), float("nan")
    eall = np.concatenate(errs)
    return mse_rmse(eall)


# ============================================================
# Prefetch queue — generates batches on CPU in background threads
# while GPU does forward/backward, hiding most of the sampler cost.
# ============================================================

import queue
import threading

def _prefetch_worker(
    sampler: "MSARBatchSampler",
    batch_size: int,
    context_len: int,
    q: "queue.Queue",
    stop_event: "threading.Event",
) -> None:
    """
    Background thread: generates numpy batches and puts them in the queue.
    Stays ahead of the training loop by keeping the queue full.
    Numpy releases the GIL during C-level computation so multiple threads
    can generate series in parallel with the GPU doing forward/backward.
    """
    while not stop_event.is_set():
        # sample_batch returns tensors on device — we want numpy here so
        # we temporarily redirect to CPU and return raw arrays
        xs = []
        ys = []
        while len(xs) < batch_size:
            series = sampler._sample_one_series()
            if not np.isfinite(series).all() or series.std() < 1e-6:
                continue
            max_start = len(series) - context_len - 1
            if max_start <= 0:
                continue
            start = sampler.rng.integers(0, max_start)
            ctx = series[start: start + context_len].astype(np.float32)
            tgt = series[start + context_len].astype(np.float32)
            if not (np.isfinite(ctx).all() and np.isfinite(tgt)):
                continue
            xs.append(ctx)
            ys.append(tgt)
        try:
            q.put((np.stack(xs), np.array(ys, dtype=np.float32)), timeout=5)
        except queue.Full:
            pass  # training loop is behind — drop and regenerate


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
    n_prefetch_workers: int = 3,
    prefetch_queue_size: int = 8,
) -> None:
    """
    Train on fresh synthetic MSAR series every step.
    No fixed dataset, no memorization possible.
    val_loader used only for monitoring — no gradients flow from it.

    n_prefetch_workers background threads generate batches on CPU while
    the GPU runs forward/backward, hiding most of the sampler latency.
    """
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    context_len = model.cfg.context_len

    # spin up background workers
    q: queue.Queue = queue.Queue(maxsize=prefetch_queue_size)
    stop_event = threading.Event()
    workers = []
    for _ in range(n_prefetch_workers):
        t = threading.Thread(
            target=_prefetch_worker,
            args=(sampler, batch_size, context_len, q, stop_event),
            daemon=True,
        )
        t.start()
        workers.append(t)

    try:
        pbar = tqdm(range(steps), desc="train (iid)")
        for step in pbar:
            # get next pre-generated batch and move to GPU
            xs_np, ys_np = q.get(timeout=60)
            x = torch.from_numpy(xs_np).unsqueeze(-1).to(device)
            y = torch.from_numpy(ys_np).unsqueeze(-1).to(device)

            opt.zero_grad(set_to_none=True)
            yhat = model(x)
            loss = loss_fn(yhat, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if step % 100 == 0 or step == steps - 1:
                _, rmse_v = eval_loop(model, val_loader, device=device)
                pbar.set_postfix(loss=float(loss.item()), val_rmse=float(rmse_v))
    finally:
        # always shut down workers cleanly even if training crashes
        stop_event.set()
        for t in workers:
            t.join(timeout=5)


# ============================================================
# Fixed-dataset training (original, kept for comparison)
# ============================================================

def train_fixed(
    model: CausalTransformerForecaster,
    train_loader: DataLoader,
    val_loader: DataLoader,
    steps: int,
    lr: float,
    device: torch.device,
) -> None:
    """
    Original training on sliding windows of the fixed dataset.
    Prone to overfitting on short series.
    """
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
) -> Dict[str, Any]:
    """
    training_mode="iid":
        Garg-style. Train on infinite fresh synthetic MSAR series.
        At eval time, actual dataset windows are passed as context
        in the forward pass — in-context inference, no fine-tuning.
        No overfitting. Fair comparison to MSAR.

    training_mode="fixed":
        Original. Train on sliding windows of the fixed dataset.
        Prone to overfitting.
    """
    device = resolve_device(device)
    print("Using device:", device)

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
    )
    model = CausalTransformerForecaster(cfg).to(device)
    model.train()

    if training_mode == "iid":
        sampler_cfg = MSARSamplerConfig(
            series_len=max(512, context_len * 4),
            k_regimes=2,
            ar_coeff_scale=0.6,
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

    # Eval on actual dataset splits via in-context forward passes (no fine-tuning)
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
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--training_mode", type=str, default="iid",
                    choices=["iid", "fixed"])
    ap.add_argument("--ar_order", type=int, default=2)
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
    )

    print(f"\ntransformer {args.dataset} [{out['training_mode']}]")
    print(f"train_rmse={out['train_rmse']:.4f}  val_rmse={out['val_rmse']:.4f}")


if __name__ == "__main__":
    main()
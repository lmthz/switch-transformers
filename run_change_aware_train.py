"""
run_change_aware_train.py

Trains a ChangeAwareTransformer (one model per dataset / instance) on NPZ files
that include ground-truth regime states.  The dataset returns the last switch
position within each context window so the model can learn to devalue pre-switch
observations through the learned decay parameter gamma.

Resumable across 6-hour GPU sessions:
  - Checkpoint saved every --save_every steps to runs/change_aware/
  - Final model saved as {dataset}_r{instance}_final.pt in the same directory
  - On startup, completed pairs (final.pt present) are skipped automatically
  - Partially-completed pairs resume from their latest checkpoint

Usage (single pair, typical):
    python run_change_aware_train.py --dataset A1_ar2_coeffs_easy --instance 0

Usage (full sweep, run from sbatch):
    python run_change_aware_train.py --all_datasets

Evaluation (after training):
    python run_change_aware_train.py --eval_only --eval_csv results_change_aware.csv
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from data.synthetic_npz_dataset import SeriesStandardizer
from models.transformer_forecaster import TransformerConfig
from models.change_aware_transformer import ChangeAwareTransformer
from metrics import mse_rmse

# ── Dataset list (B1 suite) ────────────────────────────────────────────────────

DATASETS_B1 = [
    "A1_ar2_coeffs_easy", "A2_ar2_coeffs_hard",
    "B1_ar2_variance",    "B2_ar2_variance_big",
    "C1_arma21_coeffs_var",
    "D1_arima211",        "D2_arima221",         "D3_arima210",
    "F1_seasonal_sarimax","F2_seasonal_exog",
    "E1_drift_only",      "E2_level_shift",       "G1_exogenous_only",
    "S1_sparse_switching","S2_frequent_switching",
]

# ── Dataset that returns (x, y, last switch position in window) ───────────────

class ChangeAwareSlidingWindowDataset(Dataset):
    """
    Sliding-window dataset that also returns the last regime-switch position
    within each context window.

    switch_pos = k  means the most recent switch in the window started at
                    window-relative position k (0-indexed).  Positions
                    0..k-1 are the old regime; k..L-1 are the new regime.
    switch_pos = 0  means no switch was detected in the window (or the switch
                    is at the very first position, making all context new-regime).
    """

    def __init__(
        self,
        y_std: np.ndarray,
        states: np.ndarray,
        context_len: int,
        start_t: int,
        end_t: int,
    ):
        y_std = np.asarray(y_std, dtype=np.float32)
        states = np.asarray(states, dtype=np.int64)
        assert y_std.ndim == 1 and states.ndim == 1
        assert len(y_std) == len(states)

        self.y = y_std
        self.states = states
        self.L = int(context_len)

        start_t = max(int(start_t), self.L)
        end_t   = min(int(end_t), len(self.y))
        if end_t <= start_t:
            raise ValueError("empty dataset")

        self.start_t = start_t
        self.end_t   = end_t
        self._len    = end_t - start_t

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, i: int):
        t = self.start_t + int(i)
        ctx = self.y[t - self.L : t]   # (L,)
        tgt = self.y[t]                # scalar

        # Find last switch within the window
        states_window = self.states[t - self.L : t]  # (L,)
        # switch_positions[k] = window index where NEW regime starts
        switch_positions = np.where(np.diff(states_window) != 0)[0] + 1
        switch_pos = int(switch_positions[-1]) if len(switch_positions) > 0 else 0

        x          = torch.from_numpy(ctx).unsqueeze(-1)      # (L, 1)
        y          = torch.tensor([tgt], dtype=torch.float32)  # (1,)
        sw         = torch.tensor(switch_pos, dtype=torch.long)
        return x, y, sw


def make_change_aware_datasets(npz_path: str, context_len: int, val_frac: float):
    """Load NPZ, split, standardise, return (train_ds, val_ds, stdzr)."""
    data = np.load(npz_path)
    y      = np.asarray(data["y"],      dtype=float)
    states = np.asarray(data["states"], dtype=int)

    n       = len(y)
    n_train = max(1, min(int((1.0 - val_frac) * n), n - 1))

    mean = float(y[:n_train].mean())
    std  = float(y[:n_train].std())
    std  = std if std > 0 else 1.0
    stdzr = SeriesStandardizer(mean=mean, std=std)

    y_std = stdzr.transform(y)

    ds_train = ChangeAwareSlidingWindowDataset(y_std, states, context_len, 0,       n_train)
    ds_val   = ChangeAwareSlidingWindowDataset(y_std, states, context_len, n_train, n)
    return ds_train, ds_val, stdzr


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def _ckpt_path(ckpt_dir: str, dataset: str, instance: int) -> str:
    return str(Path(ckpt_dir) / f"{dataset}_r{instance}_ckpt.pt")

def _final_path(ckpt_dir: str, dataset: str, instance: int) -> str:
    return str(Path(ckpt_dir) / f"{dataset}_r{instance}_final.pt")

def save_checkpoint(path: str, model, optimizer, step: int, best_val_rmse: float):
    torch.save({
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "step":            step,
        "best_val_rmse":   best_val_rmse,
    }, path)

def load_checkpoint(path: str, model, optimizer, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt["step"], ckpt["best_val_rmse"]


# ── Training loop ──────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_val(model: ChangeAwareTransformer, loader: DataLoader, device) -> float:
    model.eval()
    errs = []
    for x, y, sw in loader:
        x  = x.to(device)
        y  = y.to(device)
        sw = sw.to(device)
        yhat = model.forward(x, change_point_idx=sw)[:, -1, :]
        errs.append((y - yhat).detach().cpu().numpy().reshape(-1))
    model.train()
    if not errs:
        return float("nan")
    _, rmse = mse_rmse(np.concatenate(errs))
    return float(rmse)


def train_one(
    npz_path: str,
    dataset: str,
    instance: int,
    ckpt_dir: str,
    context_len: int,
    val_frac: float,
    steps: int,
    batch_size: int,
    lr: float,
    save_every: int,
    device: torch.device,
    d_model: int = 256,
    n_heads: int = 4,
    n_layers: int = 6,
    dropout: float = 0.1,
) -> Dict:
    """
    Train one ChangeAwareTransformer on a single NPZ file.
    Resumes from checkpoint if one exists; skips if final model already present.
    """
    final_path = _final_path(ckpt_dir, dataset, instance)
    if os.path.exists(final_path):
        print(f"  [skip] {dataset} r{instance} — final model already exists")
        ckpt = torch.load(final_path, map_location=device)
        return {"dataset": dataset, "instance": instance,
                "val_rmse": ckpt.get("val_rmse", float("nan")), "skipped": True}

    if not os.path.exists(npz_path):
        print(f"  [skip] {dataset} r{instance} — {npz_path} not found")
        return {}

    ds_train, ds_val, stdzr = make_change_aware_datasets(npz_path, context_len, val_frac)
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False)

    cfg   = TransformerConfig(context_len=context_len, d_model=d_model,
                               n_heads=n_heads, n_layers=n_layers, dropout=dropout)
    model = ChangeAwareTransformer(cfg).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    start_step     = 0
    best_val_rmse  = float("inf")
    ckpt_path      = _ckpt_path(ckpt_dir, dataset, instance)

    if os.path.exists(ckpt_path):
        start_step, best_val_rmse = load_checkpoint(ckpt_path, model, opt, device)
        print(f"  [resume] {dataset} r{instance} from step {start_step} "
              f"(best val RMSE {best_val_rmse:.4f})")
    else:
        print(f"  [train]  {dataset} r{instance} from scratch")

    model.train()
    it   = iter(train_loader)
    pbar = tqdm(range(start_step, steps),
                desc=f"{dataset} r{instance}", leave=False)

    for step in pbar:
        try:
            x, y, sw = next(it)
        except StopIteration:
            it = iter(train_loader)
            x, y, sw = next(it)

        x  = x.to(device)
        y  = y.to(device)
        sw = sw.to(device)

        # Dense supervision: position t predicts t+1 for all L positions
        y_seq = torch.cat([x[:, 1:, :], y.unsqueeze(1)], dim=1)  # (B, L, 1)

        opt.zero_grad(set_to_none=True)
        yhat = model(x, change_point_idx=sw)          # (B, L, 1)
        loss = loss_fn(yhat, y_seq)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if (step + 1) % 100 == 0:
            val_rmse = eval_val(model, val_loader, device)
            gamma_val = float(model.gamma.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             val_rmse=f"{val_rmse:.4f}",
                             gamma=f"{gamma_val:.4f}")
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse

        if (step + 1) % save_every == 0:
            save_checkpoint(ckpt_path, model, opt, step + 1, best_val_rmse)

    # Save final model
    val_rmse = eval_val(model, val_loader, device)
    torch.save({
        "model_state":  model.state_dict(),
        "cfg":          cfg.__dict__,
        "val_rmse":     val_rmse,
        "gamma":        float(model.gamma.item()),
        "dataset":      dataset,
        "instance":     instance,
        "stdzr_mean":   stdzr.mean,
        "stdzr_std":    stdzr.std,
    }, final_path)
    print(f"  [done]   {dataset} r{instance}  "
          f"val_rmse={val_rmse:.4f}  gamma={model.gamma.item():.4f}")

    # Remove intermediate checkpoint now that final is saved
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    return {"dataset": dataset, "instance": instance,
            "val_rmse": val_rmse, "gamma": float(model.gamma.item())}


# ── Evaluation ─────────────────────────────────────────────────────────────────

def eval_series_change_aware(
    model: ChangeAwareTransformer,
    npz_path: str,
    context_len: int,
    val_frac: float,
    device,
    cusum_threshold: float = 5.0,
    cusum_drift: float = 0.5,
    dataset_name: str = "",
    instance: int = 0,
) -> dict:
    """
    Evaluate a trained ChangeAwareTransformer with CUSUM-detected switch positions.

    At each step:
      1. Compute transformer output using current best-guess switch position.
      2. Observe true value, update CUSUM on squared prediction error.
      3. If CUSUM fires, record current window position as the new switch point.

    The switch position passed to the model is the window-relative index of the
    most recent CUSUM firing (or 0 if CUSUM has not fired recently).
    """
    from adaptation.cusum import CUSUM
    from metrics import (
        train_val_split_indices, post_switch_spike_ratio,
        recovery_time, per_switch_metrics, cumulative_excess_loss,
    )

    data       = np.load(npz_path)
    y_full     = data["y"].astype(np.float32).flatten()
    states_full = data["states"].astype(int).flatten()

    n_total = len(y_full)
    n_train, n_val = train_val_split_indices(n_total, val_frac)
    y_val      = y_full[n_train:]
    states_val = states_full[n_train:]
    n          = len(y_val)

    if n < 2:
        return {}

    cusum = CUSUM(threshold=cusum_threshold, drift=cusum_drift)

    # Rolling observation buffer (pre-seeded with training data)
    buffer: List[float] = list(y_full[:n_train].astype(float))
    sq_errs: List[float] = []

    # Global timestep of last CUSUM fire (None = no fire yet)
    last_fire_global: Optional[int] = None

    preds = []
    model.eval()

    with torch.no_grad():
        for t in range(n):
            # Build context window
            ctx = np.array(buffer[-context_len:], dtype=np.float32).reshape(1, -1, 1)
            x   = torch.tensor(ctx, device=device)

            # Compute switch position within current window
            win_start_global = (n_train + t) - context_len  # global index of window start
            if last_fire_global is not None and last_fire_global > win_start_global:
                switch_pos = last_fire_global - win_start_global
            else:
                switch_pos = 0
            switch_pos = max(0, min(switch_pos, context_len - 1))

            yhat = float(
                model.predict_next_aware(x, change_point_idx=switch_pos)[0, 0].cpu()
            )
            preds.append(yhat)

            # CUSUM update
            sq_err = (float(y_val[t]) - yhat) ** 2
            sq_errs.append(sq_err)
            baseline = float(np.mean(sq_errs[-50:])) if sq_errs else 0.0
            _, fired = cusum.update(sq_err - baseline)
            if fired:
                last_fire_global = n_train + t

            buffer.append(float(y_val[t]))

    preds_arr = np.array(preds, dtype=float)
    sq_err_arr = (y_val.astype(float) - preds_arr) ** 2

    rmse  = float(np.sqrt(sq_err_arr.mean()))
    spike = post_switch_spike_ratio(sq_err_arr, states_val)
    rec   = recovery_time(sq_err_arr, states_val)
    cel   = cumulative_excess_loss(sq_err_arr, states_val)
    per_sw = per_switch_metrics(sq_err_arr, states_val, context_len=context_len)

    return {
        "dataset":                dataset_name,
        "instance":               instance,
        "variant":                "change_aware",
        "rmse":                   rmse,
        "spike_ratio":            spike,
        "recovery_time":          rec,
        "cumulative_excess_loss": cel,
        "per_switch":             per_sw,
    }


def eval_suite_change_aware(
    ckpt_dir: str,
    data_dir: str,
    datasets: List[str],
    n_instances: int,
    context_len: int,
    val_frac: float,
    device,
    cusum_threshold: float = 5.0,
    cusum_drift: float = 0.5,
    out_csv: str = "results_change_aware.csv",
):
    import pandas as pd

    rows = []
    for ds in datasets:
        for ri in range(n_instances):
            final_path = _final_path(ckpt_dir, ds, ri)
            npz_path   = str(Path(data_dir) / f"{ds}_r{ri}.npz")
            if not os.path.exists(final_path) or not os.path.exists(npz_path):
                continue

            ckpt  = torch.load(final_path, map_location=device)
            cfg   = TransformerConfig(**ckpt["cfg"])
            model = ChangeAwareTransformer(cfg).to(device)
            model.load_state_dict(ckpt["model_state"])

            result = eval_series_change_aware(
                model, npz_path, context_len, val_frac, device,
                cusum_threshold=cusum_threshold,
                cusum_drift=cusum_drift,
                dataset_name=ds,
                instance=ri,
            )
            if result:
                rows.append(result)

    if not rows:
        print("No results to save.")
        return

    flat = []
    for r in rows:
        per_sw = r.pop("per_switch", [])
        for sw in per_sw:
            flat.append({**r, **sw})

    df = pd.DataFrame(flat)
    df.to_csv(out_csv, index=False)
    print(f"Saved {len(df)} rows to {out_csv}")
    return df


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()

    # Training targets (mutually exclusive: single pair or full sweep)
    tgt = ap.add_mutually_exclusive_group()
    tgt.add_argument("--dataset",      type=str, help="Single dataset name")
    tgt.add_argument("--all_datasets", action="store_true",
                     help="Train on all DATASETS_B1")

    ap.add_argument("--instance",    type=int,   default=0)
    ap.add_argument("--n_instances", type=int,   default=3,
                    help="Number of instances per dataset when --all_datasets")

    # Paths
    ap.add_argument("--data_dir",  type=str, default="generated_data")
    ap.add_argument("--ckpt_dir",  type=str, default="runs/change_aware")

    # Model / training
    ap.add_argument("--context_len", type=int,   default=64)
    ap.add_argument("--val_frac",    type=float, default=0.3)
    ap.add_argument("--steps",       type=int,   default=30000)
    ap.add_argument("--batch_size",  type=int,   default=128)
    ap.add_argument("--lr",          type=float, default=3e-4)
    ap.add_argument("--save_every",  type=int,   default=500,
                    help="Save checkpoint every N steps")
    ap.add_argument("--d_model",     type=int,   default=256)
    ap.add_argument("--n_heads",     type=int,   default=4)
    ap.add_argument("--n_layers",    type=int,   default=6)
    ap.add_argument("--dropout",     type=float, default=0.1)
    ap.add_argument("--seed",        type=int,   default=0)
    ap.add_argument("--device",      type=str,   default="cuda")

    # Evaluation mode
    ap.add_argument("--eval_only",   action="store_true",
                    help="Skip training; evaluate all finished checkpoints")
    ap.add_argument("--eval_csv",    type=str,   default="results_change_aware.csv")
    ap.add_argument("--cusum_threshold", type=float, default=5.0)
    ap.add_argument("--cusum_drift",     type=float, default=0.5)

    args = ap.parse_args()

    # Device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.ckpt_dir, exist_ok=True)

    # Evaluation-only mode
    if args.eval_only:
        datasets = DATASETS_B1 if args.all_datasets or args.dataset is None \
                   else [args.dataset]
        n_inst   = args.n_instances if args.all_datasets else (args.instance + 1)
        eval_suite_change_aware(
            ckpt_dir=args.ckpt_dir,
            data_dir=args.data_dir,
            datasets=datasets,
            n_instances=n_inst,
            context_len=args.context_len,
            val_frac=args.val_frac,
            device=device,
            cusum_threshold=args.cusum_threshold,
            cusum_drift=args.cusum_drift,
            out_csv=args.eval_csv,
        )
        return

    # Training mode
    if args.all_datasets:
        pairs = [(ds, ri)
                 for ds in DATASETS_B1
                 for ri in range(args.n_instances)]
    else:
        if args.dataset is None:
            ap.error("Supply --dataset <name> or --all_datasets")
        pairs = [(args.dataset, args.instance)]

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

    print(f"Training {len(pairs)} model(s)  |  device={args.device}  "
          f"|  steps={args.steps}")
    for ds, ri in pairs:
        npz_path = str(Path(args.data_dir) / f"{ds}_r{ri}.npz")
        train_one(npz_path=npz_path, dataset=ds, instance=ri, **train_kwargs)


if __name__ == "__main__":
    main()

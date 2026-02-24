# train_transformer.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.synthetic_npz_dataset import make_train_val_datasets
from models.transformer_forecaster import TransformerConfig, CausalTransformerForecaster
from metrics import mse_rmse


# ============================================================
# Device utility (robust to CPU-only PyTorch)
# ============================================================

def resolve_device(device_str: str) -> torch.device:
    """
    Safely resolve device. Falls back to CPU if CUDA
    not available or torch not compiled with CUDA.
    """
    if device_str == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            print("⚠ CUDA requested but not available. Falling back to CPU.")
            return torch.device("cpu")
    return torch.device("cpu")


@torch.no_grad()
def eval_loop(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    errs: List[np.ndarray] = []

    for x, y, _s in loader:
        x = x.to(device)
        y = y.to(device)

        yhat = model(x)
        e = (y - yhat).detach().cpu().numpy().reshape(-1)
        errs.append(e)

    if not errs:
        return float("nan"), float("nan")

    eall = np.concatenate(errs)
    mse, rmse = mse_rmse(eall)
    return mse, rmse


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
) -> Dict[str, Any]:

    # --------------------------------------------------------
    # Resolve device safely
    # --------------------------------------------------------
    device = resolve_device(device)
    print("Using device:", device)

    torch.manual_seed(seed)
    np.random.seed(seed)

    ds_train, ds_val, stdzr, _meta = make_train_val_datasets(
        npz_path=npz_path,
        context_len=context_len,
        val_frac=val_frac,
    )

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False, drop_last=False)

    cfg = TransformerConfig(
        context_len=context_len,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout,
    )

    model = CausalTransformerForecaster(cfg).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    model.train()
    it = iter(train_loader)
    pbar = tqdm(range(int(steps)), desc="train")

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
            _mse_v, rmse_v = eval_loop(model, val_loader, device=device)
            pbar.set_postfix(loss=float(loss.item()), val_rmse=float(rmse_v))

    mse_tr, rmse_tr = eval_loop(
        model,
        DataLoader(ds_train, batch_size=batch_size, shuffle=False),
        device=device,
    )

    mse_v, rmse_v = eval_loop(model, val_loader, device=device)

    return {
        "train_rmse": float(rmse_tr),
        "val_rmse": float(rmse_v),
        "train_mse": float(mse_tr),
        "val_mse": float(mse_v),
        "std_used": float(stdzr.std),
        "mean_used": float(stdzr.mean),
        "n_train": int(len(ds_train)),
        "n_val": int(len(ds_val)),
        "model_state_dict": model.state_dict(),
        "model_config": cfg.__dict__,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="generated_data")
    ap.add_argument("--dataset", type=str, required=True)

    ap.add_argument("--context_len", type=int, default=64)
    ap.add_argument("--val_frac", type=float, default=0.3)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)

    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)

    ap.add_argument("--seed", type=int, default=0)

    # Keep CLI arg but resolve safely later
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--save_metrics", type=str, default=None)

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
    )

    print(f"\ntransformer {args.dataset}")
    print(f"train_rmse_std={out['train_rmse']:.4f}  val_rmse_std={out['val_rmse']:.4f}")

    if args.save_metrics:
        p = Path(args.save_metrics)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "train_rmse": out["train_rmse"],
                "val_rmse": out["val_rmse"],
                "train_mse": out["train_mse"],
                "val_mse": out["val_mse"],
                "model_config": out["model_config"],
            },
            p,
        )
        print(f"saved metrics to {p}")


if __name__ == "__main__":
    main()
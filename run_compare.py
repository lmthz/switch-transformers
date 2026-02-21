from __future__ import annotations

import argparse
from pathlib import Path

import torch

from utils.config import load_config
from utils.paths import make_run_dir
from utils.logging import setup_logger
from utils.checkpoint import load_checkpoint

from baselines.prediction_msar import run_msar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--override", type=str, action="append", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, overrides=args.override)
    run_dir = make_run_dir(cfg)
    logger = setup_logger(run_dir, "compare")

    dataset = cfg["data"]["dataset"]
    data_dir = cfg["data"]["data_dir"]
    val_frac = float(cfg["data"]["val_frac"])

    cand = list(cfg["msar"]["candidate_orders"])
    maxiter = int(cfg["msar"]["maxiter"])
    em_iter = int(cfg["msar"]["em_iter"])

    msar_res = run_msar(dataset, data_dir=data_dir, val_frac=val_frac, candidate_orders=cand, maxiter=maxiter, em_iter=em_iter)
    logger.info(f"msar: train_rmse={msar_res['train_rmse']:.6f} val_rmse={msar_res['val_rmse']:.6f} regime_acc={msar_res['regime_accuracy']:.3f}")

    final_metrics_path = Path(run_dir) / "final_metrics.pt"
    if not final_metrics_path.exists():
        logger.info("transformer final_metrics.pt not found. run train_transformer.py first.")
        return

    tr_res = load_checkpoint(str(final_metrics_path), map_location="cpu")
    logger.info(f"transformer: train_rmse={tr_res['train_rmse']:.6f} val_rmse={tr_res['val_rmse']:.6f}")

    print("\nsummary")
    print(
        f"{dataset} | "
        f"msar_val_rmse={msar_res['val_rmse']:.4f} "
        f"tr_val_rmse={tr_res['val_rmse']:.4f} "
        f"noise_rmse={msar_res['noise_rmse']:.4f} "
        f"regime_acc={msar_res['regime_accuracy']:.3f}"
    )


if __name__ == "__main__":
    main()

#python data_generation.py
#python train_transformer.py --config configs/default.yaml
#python run_compare.py --config configs/default.yaml
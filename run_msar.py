from __future__ import annotations

import argparse
from utils.config import load_config, save_config_snapshot
from utils.paths import make_run_dir
from utils.logging import setup_logger

from baselines.prediction_msar import run_msar

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--override", type=str, action="append", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, overrides=args.override)
    run_dir = make_run_dir(cfg)
    logger = setup_logger(run_dir, "run_msar")
    save_config_snapshot(cfg, run_dir)

    dataset = cfg["data"]["dataset"]
    data_dir = cfg["data"]["data_dir"]
    val_frac = float(cfg["data"]["val_frac"])

    cand = list(cfg["msar"]["candidate_orders"])
    maxiter = int(cfg["msar"]["maxiter"])
    em_iter = int(cfg["msar"]["em_iter"])

    res = run_msar(dataset, data_dir=data_dir, val_frac=val_frac, candidate_orders=cand, maxiter=maxiter, em_iter=em_iter)
    logger.info(
        f"dataset={dataset} "
        f"train_rmse={res['train_rmse']:.6f} val_rmse={res['val_rmse']:.6f} "
        f"regime_acc={res['regime_accuracy']:.3f} noise_rmse={res['noise_rmse']:.6f}"
    )
    if "selected_order" in res:
        logger.info(f"selected_order={res['selected_order']} selection_metric={res.get('selection_metric')}")
    print(res)

if __name__ == "__main__":
    main()
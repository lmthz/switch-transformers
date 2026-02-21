from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
from numpy.linalg import LinAlgError
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression

from metrics import mse_rmse, per_regime_rmse, label_corrected_accuracy, train_val_split_indices


@dataclass
class MSARConfig:
    name: str
    order: int
    k_regimes: int = 2
    switching_ar: bool = True
    switching_variance: bool = True
    switching_exog: bool = False
    trend: str = "c"
    use_exog: bool = False


CONFIGS: Dict[str, MSARConfig] = {
    "A1_ar2_coeffs_easy": MSARConfig("A1_ar2_coeffs_easy", order=2, switching_ar=True, switching_variance=True),
    "A2_ar2_coeffs_hard": MSARConfig("A2_ar2_coeffs_hard", order=2, switching_ar=True, switching_variance=True),
    "A3_ar2_coeffs_plus_var": MSARConfig("A3_ar2_coeffs_plus_var", order=2, switching_ar=True, switching_variance=True),

    "B1_ar2_variance": MSARConfig("B1_ar2_variance", order=2, switching_ar=False, switching_variance=True),
    "B2_ar2_variance_big": MSARConfig("B2_ar2_variance_big", order=2, switching_ar=False, switching_variance=True),

    "C1_arma21_coeffs_var": MSARConfig("C1_arma21_coeffs_var", order=5, switching_ar=True, switching_variance=True),

    "D1_arima211": MSARConfig("D1_arima211", order=5, switching_ar=True, switching_variance=True),
    "D2_arima221": MSARConfig("D2_arima221", order=5, switching_ar=True, switching_variance=True),
    "D3_arima210": MSARConfig("D3_arima210", order=5, switching_ar=True, switching_variance=True),

    "E1_drift_only": MSARConfig("E1_drift_only", order=2, switching_ar=False, switching_variance=False, switching_exog=True, use_exog=True),
    "E2_level_shift": MSARConfig("E2_level_shift", order=2, switching_ar=False, switching_variance=False, switching_exog=True, use_exog=True),

    "F1_seasonal_sarimax": MSARConfig("F1_seasonal_sarimax", order=5, switching_ar=True, switching_variance=False),
    "F2_seasonal_exog": MSARConfig("F2_seasonal_exog", order=5, switching_ar=False, switching_variance=False, switching_exog=True, use_exog=True),

    "G1_exogenous_only": MSARConfig("G1_exogenous_only", order=2, switching_ar=False, switching_variance=False, switching_exog=True, use_exog=True),

    "H1_ar10_coeffs": MSARConfig("H1_ar10_coeffs", order=10, switching_ar=True, switching_variance=True),
    "H2_ar1_near_unit_root": MSARConfig("H2_ar1_near_unit_root", order=1, switching_ar=True, switching_variance=True),

    "S1_sparse_switching": MSARConfig("S1_sparse_switching", order=2, switching_ar=True, switching_variance=True),
    "S2_frequent_switching": MSARConfig("S2_frequent_switching", order=2, switching_ar=True, switching_variance=True),

    "NS0_A1_no_switch_regime0": MSARConfig("NS0_A1_no_switch_regime0", order=2, switching_ar=True, switching_variance=True),
    "NS1_A1_no_switch_regime1": MSARConfig("NS1_A1_no_switch_regime1", order=2, switching_ar=True, switching_variance=True),
    "SW1_A1_single_switch": MSARConfig("SW1_A1_single_switch", order=2, switching_ar=True, switching_variance=True),
}


ARMA_ARIMA_DATASETS = {
    "C1_arma21_coeffs_var",
    "D1_arima211",
    "D2_arima221",
    "D3_arima210",
    "F1_seasonal_sarimax",
    "F2_seasonal_exog",
}


def load_npz_series(data_dir: str, dataset_name: str):
    arr = np.load(Path(data_dir) / f"{dataset_name}.npz")
    y = arr["y"].astype(float)
    states = arr["states"].astype(int)
    T = arr["T"].astype(float)

    sigma = arr["sigma"] if "sigma" in arr.files else None
    ar = arr["ar"] if "ar" in arr.files else None
    ma = arr["ma"] if "ma" in arr.files else None
    eps = arr["eps"] if "eps" in arr.files else None
    z = arr["z"] if "z" in arr.files else None
    d = int(arr["d"]) if "d" in arr.files else None
    return y, states, T, sigma, ar, ma, eps, z, d


def build_exog_for_dataset(dataset_name: str, n_total: int) -> Optional[np.ndarray]:
    t = np.arange(n_total)
    if dataset_name in ("E1_drift_only", "E2_level_shift"):
        return np.ones((n_total, 1), dtype=float)
    if dataset_name == "G1_exogenous_only":
        s = 24
        x = np.sin(2 * np.pi * t / s)
        return x[:, None]
    if dataset_name == "F2_seasonal_exog":
        s = 24
        x1 = np.sin(2 * np.pi * t / s)
        x2 = np.cos(2 * np.pi * t / s)
        return np.column_stack([x1, x2])
    return None


def fit_markov_ar(y_train: np.ndarray, cfg: MSARConfig, exog_train: Optional[np.ndarray], maxiter: int, em_iter: int):
    model = MarkovAutoregression(
        endog=y_train,
        k_regimes=cfg.k_regimes,
        order=cfg.order,
        trend=cfg.trend,
        exog=exog_train if cfg.use_exog else None,
        switching_ar=cfg.switching_ar,
        switching_trend=True,
        switching_exog=cfg.switching_exog,
        switching_variance=cfg.switching_variance,
    )

    fit_kwargs = dict(disp=False, maxiter=maxiter, em_iter=em_iter)
    try:
        res = model.fit(**fit_kwargs)
    except LinAlgError:
        fit_kwargs["em_iter"] = 0
        res = model.fit(**fit_kwargs)
    return model, res


def predict_in_sample(res) -> np.ndarray:
    """
    In-sample one-step-ahead predictions for MarkovAutoregression.

    statsmodels does not consistently implement res.predict() for MarkovSwitching classes,
    but fittedvalues is available and corresponds to in-sample conditional means.
    """
    fv = getattr(res, "fittedvalues", None)
    if fv is None:
        raise RuntimeError("results object has no fittedvalues; cannot compute in-sample predictions")
    return np.asarray(fv, dtype=float)


def predict_out_of_sample(res, steps: int) -> np.ndarray:
    """
    Out-of-sample forecasts for MarkovAutoregression.

    Uses res.forecast(steps=...) which is implemented.
    """
    fc = res.forecast(steps=steps)
    return np.asarray(fc, dtype=float)

def evaluate_msar_fixed_order(
    dataset_name: str,
    data_dir: str,
    cfg: MSARConfig,
    val_frac: float,
    maxiter: int,
    em_iter: int,
) -> Dict[str, Any]:
    y_raw, true_states, T_true, sigma, ar, ma, eps, z, d = load_npz_series(data_dir, dataset_name)
    n = len(y_raw)
    n_train, n_total = train_val_split_indices(n, val_frac)

    # train-only standardization
    mean = float(y_raw[:n_train].mean())
    std = float(y_raw[:n_train].std()) or 1.0
    y = (y_raw - mean) / std

    exog_full = build_exog_for_dataset(dataset_name, n_total) if cfg.use_exog else None
    exog_train = exog_full[:n_train] if exog_full is not None else None

    model, res = fit_markov_ar(y[:n_train], cfg, exog_train, maxiter=maxiter, em_iter=em_iter)

    # predictions for train and val
    # train predictions start at 0, but early indices may be less meaningful due to lag order
    # --- in-sample (train) ---
    pred_train_all = predict_in_sample(res)  # length is typically n_train - order (or similar)
    # align to y_train
    y_train = y[:n_train]
    m_tr = min(len(pred_train_all), len(y_train))
    pred_train = pred_train_all[-m_tr:]
    y_train_aligned = y_train[-m_tr:]

    # --- out-of-sample (val) ---
    n_val = len(y) - n_train
    pred_val = predict_out_of_sample(res, steps=n_val)  # length n_val
    y_val = y[n_train:]
    m_va = min(len(pred_val), len(y_val))
    pred_val = pred_val[:m_va]
    y_val_aligned = y_val[:m_va]

    err_train = y[:len(pred_train)] - pred_train
    err_val = y[n_train : n_train + len(pred_val)] - pred_val

    train_mse, train_rmse = mse_rmse(err_train)
    val_mse, val_rmse = mse_rmse(err_val)

    # regime decoding on train length
    smoothed = np.asarray(res.smoothed_marginal_probabilities)
    decoded = np.argmax(smoothed, axis=1)
    acc = label_corrected_accuracy(decoded, true_states[: len(decoded)], cfg.k_regimes)

    # oracle noise floor in standardized units
    if sigma is not None:
        sigma_vec = np.asarray(sigma, dtype=float).flatten() / std
        noise_rmse = float(np.sqrt(np.mean((sigma_vec[true_states]) ** 2)))
    else:
        noise_rmse = float("nan")

    # per regime rmse on train and val using true states
    # align to the error arrays
    true_train_states = true_states[: len(err_train)]
    true_val_states = true_states[n_train : n_train + len(err_val)]
    per_reg_train = per_regime_rmse(err_train, true_train_states, cfg.k_regimes)
    per_reg_val = per_regime_rmse(err_val, true_val_states, cfg.k_regimes)

    bic = float(res.bic) if hasattr(res, "bic") else float("nan")

    return {
        "dataset": dataset_name,
        "config": cfg,
        "n_train": n_train,
        "train_rmse": train_rmse,
        "val_rmse": val_rmse,
        "train_mse": train_mse,
        "val_mse": val_mse,
        "regime_accuracy": acc["acc"],
        "regime_accuracy_no_swap": acc["acc_no_swap"],
        "regime_accuracy_swap": acc["acc_swap"],
        "noise_rmse": noise_rmse,
        "bic": bic,
        "per_regime_train": per_reg_train,
        "per_regime_val": per_reg_val,
        "mean_used": mean,
        "std_used": std,
    }


def run_msar(dataset_name: str, data_dir: str, val_frac: float, candidate_orders: List[int], maxiter: int, em_iter: int) -> Dict[str, Any]:
    cfg0 = CONFIGS[dataset_name]

    if dataset_name in ARMA_ARIMA_DATASETS:
        best = None
        best_bic = float("inf")
        best_order = None
        for o in candidate_orders:
            cfg = replace(cfg0, order=int(o))
            out = evaluate_msar_fixed_order(dataset_name, data_dir, cfg, val_frac, maxiter, em_iter)
            bic = out.get("bic", float("inf"))
            if np.isfinite(bic) and bic < best_bic:
                best_bic = bic
                best = out
                best_order = o
        if best is None:
            best = evaluate_msar_fixed_order(dataset_name, data_dir, cfg0, val_frac, maxiter, em_iter)
            best_order = cfg0.order
        best["selected_order"] = best_order
        best["selection_metric"] = "BIC"
        return best

    return evaluate_msar_fixed_order(dataset_name, data_dir, cfg0, val_frac, maxiter, em_iter)
# baselines/prediction_msar.py
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
from numpy.linalg import LinAlgError
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression

from metrics import (
    mse_rmse,
    per_regime_rmse,
    label_corrected_accuracy,
    train_val_split_indices,
)


# ================================================================
# CONFIG
# ================================================================

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
    "A1_ar2_coeffs_easy": MSARConfig("A1_ar2_coeffs_easy", order=2),
    "A2_ar2_coeffs_hard": MSARConfig("A2_ar2_coeffs_hard", order=2),
    "A3_ar2_coeffs_plus_var": MSARConfig("A3_ar2_coeffs_plus_var", order=2),

    "B1_ar2_variance": MSARConfig("B1_ar2_variance", order=2, switching_ar=False),
    "B2_ar2_variance_big": MSARConfig("B2_ar2_variance_big", order=2, switching_ar=False),

    "C1_arma21_coeffs_var": MSARConfig("C1_arma21_coeffs_var", order=5),

    "D1_arima211": MSARConfig("D1_arima211", order=5),
    "D2_arima221": MSARConfig("D2_arima221", order=5),
    "D3_arima210": MSARConfig("D3_arima210", order=5),

    "E1_drift_only": MSARConfig("E1_drift_only", order=2, switching_ar=False, switching_variance=False, switching_exog=True, use_exog=True),
    "E2_level_shift": MSARConfig("E2_level_shift", order=2, switching_ar=False, switching_variance=False, switching_exog=True, use_exog=True),

    "F1_seasonal_sarimax": MSARConfig("F1_seasonal_sarimax", order=5),
    "F2_seasonal_exog": MSARConfig("F2_seasonal_exog", order=5, switching_ar=False, switching_variance=False, switching_exog=True, use_exog=True),

    "G1_exogenous_only": MSARConfig("G1_exogenous_only", order=2, switching_ar=False, switching_variance=False, switching_exog=True, use_exog=True),

    "H1_ar10_coeffs": MSARConfig("H1_ar10_coeffs", order=10),
    "H2_ar1_near_unit_root": MSARConfig("H2_ar1_near_unit_root", order=1),

    "S1_sparse_switching": MSARConfig("S1_sparse_switching", order=2),
    "S2_frequent_switching": MSARConfig("S2_frequent_switching", order=2),

    "NS0_A1_no_switch_regime0": MSARConfig("NS0_A1_no_switch_regime0", order=2),
    "NS1_A1_no_switch_regime1": MSARConfig("NS1_A1_no_switch_regime1", order=2),
    "SW1_A1_single_switch": MSARConfig("SW1_A1_single_switch", order=2),
}


ARMA_ARIMA_DATASETS = {
    "C1_arma21_coeffs_var",
    "D1_arima211",
    "D2_arima221",
    "D3_arima210",
    "F1_seasonal_sarimax",
    "F2_seasonal_exog",
}


# ================================================================
# DATA LOADING
# ================================================================

def load_npz_series(data_dir: str, dataset_name: str):
    arr = np.load(Path(data_dir) / f"{dataset_name}.npz")
    y = arr["y"].astype(float)
    states = arr["states"].astype(int)
    T = arr["T"].astype(float)

    sigma = arr["sigma"] if "sigma" in arr.files else None
    return y, states, T, sigma


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


# ================================================================
# MODEL FITTING
# ================================================================

def fit_markov_ar(y_train, cfg, exog_train, maxiter, em_iter):
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

    try:
        res = model.fit(disp=False, maxiter=maxiter, em_iter=em_iter)
    except LinAlgError:
        res = model.fit(disp=False, maxiter=maxiter, em_iter=0)

    return model, res


# ================================================================
# SAFE PROB EXTRACTION
# ================================================================

def _to_2d_array(x) -> Optional[np.ndarray]:
    if x is None:
        return None
    try:
        # pandas DataFrame/Series
        if hasattr(x, "values"):
            x = x.values
    except Exception:
        pass
    a = np.asarray(x)
    if a.ndim == 0:
        return None
    if a.ndim == 1:
        # could be flattened probabilities, not usable
        return None
    return a.astype(float)


def _pad_to_nobs(mat: np.ndarray, nobs: int, order: int) -> np.ndarray:
    """
    statsmodels often returns (nobs-order, k). pad first 'order' rows with nan.
    """
    if mat.ndim != 2:
        raise ValueError("prob matrix must be 2d")
    n_ret, k = mat.shape
    if n_ret == nobs:
        return mat
    if n_ret == nobs - order:
        out = np.full((nobs, k), np.nan, dtype=float)
        out[order:, :] = mat
        return out
    # sometimes off by a small amount due to init handling, handle leniently
    if n_ret < nobs:
        out = np.full((nobs, k), np.nan, dtype=float)
        out[(nobs - n_ret):, :] = mat
        return out
    return mat[-nobs:, :]


def _pad_fittedvalues(fv: np.ndarray, nobs: int, order: int) -> np.ndarray:
    fv = np.asarray(fv, dtype=float).reshape(-1)
    if fv.size == nobs:
        return fv
    if fv.size == nobs - order:
        out = np.full(nobs, np.nan, dtype=float)
        out[order:] = fv
        return out
    if fv.size < nobs:
        out = np.full(nobs, np.nan, dtype=float)
        out[(nobs - fv.size):] = fv
        return out
    return fv[-nobs:]


# ================================================================
# ONE STEP AHEAD VIA FILTER WITH FIXED TRAIN PARAMS
# ================================================================

def predict_full_series_with_fixed_params(
    y_full: np.ndarray,
    cfg: MSARConfig,
    exog_full: Optional[np.ndarray],
    train_params: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    one step ahead predictions for entire series using params estimated on train only.
    """
    model_full = MarkovAutoregression(
        endog=y_full,
        k_regimes=cfg.k_regimes,
        order=cfg.order,
        trend=cfg.trend,
        exog=exog_full if cfg.use_exog else None,
        switching_ar=cfg.switching_ar,
        switching_trend=True,
        switching_exog=cfg.switching_exog,
        switching_variance=cfg.switching_variance,
    )

    res_full = model_full.filter(train_params)

    fv_raw = getattr(res_full, "fittedvalues", None)
    fv = _pad_fittedvalues(np.asarray(fv_raw, dtype=float), nobs=len(y_full), order=cfg.order)

    probs_raw = getattr(res_full, "smoothed_marginal_probabilities", None)
    probs = _to_2d_array(probs_raw)

    if probs is None:
        # fallback: filtered probabilities if smoothing is unavailable
        probs_raw2 = getattr(res_full, "filtered_marginal_probabilities", None)
        probs = _to_2d_array(probs_raw2)

    if probs is None:
        decoded = np.full(len(y_full), -1, dtype=int)
        return fv, decoded

    probs = _pad_to_nobs(probs, nobs=len(y_full), order=cfg.order)

    # if probs has NaNs in padded region, argmax would break, so fill with uniform
    k = probs.shape[1]
    nan_rows = ~np.isfinite(probs).all(axis=1)
    if np.any(nan_rows):
        probs[nan_rows, :] = 1.0 / float(k)

    decoded = np.argmax(probs, axis=1).astype(int)
    return fv, decoded


# ================================================================
# EVALUATION
# ================================================================

def evaluate_msar_fixed_order(
    dataset_name: str,
    data_dir: str,
    cfg: MSARConfig,
    val_frac: float,
    maxiter: int,
    em_iter: int,
) -> Dict[str, Any]:

    y_raw, true_states, _T, sigma = load_npz_series(data_dir, dataset_name)
    n = len(y_raw)
    n_train, _ = train_val_split_indices(n, val_frac)

    # train-only standardization
    mean = float(y_raw[:n_train].mean())
    std = float(y_raw[:n_train].std())
    if std <= 0:
        std = 1.0
    y = (y_raw - mean) / std

    exog_full = build_exog_for_dataset(dataset_name, n) if cfg.use_exog else None
    exog_train = exog_full[:n_train] if exog_full is not None else None

    # fit on train only
    _, res_train = fit_markov_ar(y[:n_train], cfg, exog_train, maxiter, em_iter)

    # fixed-param filtering for full-series one-step predictions
    pred_full, decoded_full = predict_full_series_with_fixed_params(y, cfg, exog_full, res_train.params)

    # evaluate only where prediction is finite and decoded is available
    ok = np.isfinite(pred_full)
    idx = np.where(ok)[0]

    y_ok = y[idx]
    p_ok = pred_full[idx]

    train_mask = idx < n_train
    val_mask = idx >= n_train

    err_train = y_ok[train_mask] - p_ok[train_mask]
    err_val = y_ok[val_mask] - p_ok[val_mask]

    train_mse, train_rmse = mse_rmse(err_train)
    val_mse, val_rmse = mse_rmse(err_val)

    # regime accuracy evaluated on train region only
    dec_train = decoded_full[:n_train]
    acc = label_corrected_accuracy(dec_train, true_states[:n_train], cfg.k_regimes)

    # oracle noise floor in standardized units
    if sigma is not None:
        sigma_vec = np.asarray(sigma, dtype=float).flatten() / std
        noise_rmse = float(np.sqrt(np.mean((sigma_vec[true_states]) ** 2)))
    else:
        noise_rmse = float("nan")

    # per regime rmse using true states
    states_ok = true_states[idx]
    per_reg_train = per_regime_rmse(err_train, states_ok[train_mask], cfg.k_regimes)
    per_reg_val = per_regime_rmse(err_val, states_ok[val_mask], cfg.k_regimes)

    return {
        "dataset": dataset_name,
        "order": int(cfg.order),
        "train_rmse": float(train_rmse),
        "val_rmse": float(val_rmse),
        "regime_accuracy": float(acc["acc"]),
        "regime_accuracy_no_swap": float(acc["acc_no_swap"]),
        "regime_accuracy_swap": float(acc["acc_swap"]),
        "noise_rmse": float(noise_rmse),
        "per_regime_train": per_reg_train,
        "per_regime_val": per_reg_val,
        "std_used": float(std),
    }


def run_msar(
    dataset_name: str,
    data_dir: str = "generated_data",
    val_frac: float = 0.3,
    candidate_orders: Optional[List[int]] = None,
    maxiter: int = 150,
    em_iter: int = 10,
) -> Dict[str, Any]:

    cfg0 = CONFIGS[dataset_name]

    if dataset_name in ARMA_ARIMA_DATASETS and candidate_orders:
        best: Optional[Dict[str, Any]] = None
        best_metric = float("inf")
        for o in candidate_orders:
            cfg = replace(cfg0, order=int(o))
            out = evaluate_msar_fixed_order(dataset_name, data_dir, cfg, val_frac, maxiter, em_iter)
            # for now: choose by val rmse (out of sample)
            metric = out["val_rmse"]
            if np.isfinite(metric) and metric < best_metric:
                best_metric = metric
                best = out
                best["selected_order"] = int(o)
        if best is not None:
            return best

    return evaluate_msar_fixed_order(dataset_name, data_dir, cfg0, val_frac, maxiter, em_iter)
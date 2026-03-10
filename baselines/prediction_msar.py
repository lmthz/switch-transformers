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

    "E1_drift_only": MSARConfig(
        "E1_drift_only", order=2,
        switching_ar=False, switching_variance=False,
        switching_exog=True, use_exog=True
    ),
    "E2_level_shift": MSARConfig(
        "E2_level_shift", order=2,
        switching_ar=False, switching_variance=False,
        switching_exog=True, use_exog=True
    ),

    "F1_seasonal_sarimax": MSARConfig("F1_seasonal_sarimax", order=5),
    "F2_seasonal_exog": MSARConfig(
        "F2_seasonal_exog", order=5,
        switching_ar=False, switching_variance=False,
        switching_exog=True, use_exog=True
    ),

    "G1_exogenous_only": MSARConfig(
        "G1_exogenous_only", order=2,
        switching_ar=False, switching_variance=False,
        switching_exog=True, use_exog=True
    ),

    "H1_high_order_ar10": MSARConfig("H1_high_order_ar10", order=10),

    "I1_near_unit_root": MSARConfig("I1_near_unit_root", order=2),

    "J1_sparse_switching": MSARConfig("J1_sparse_switching", order=2),
    "J2_frequent_switching": MSARConfig("J2_frequent_switching", order=2),

    "K1_no_switch": MSARConfig("K1_no_switch", order=2),
    "K2_single_switch": MSARConfig("K2_single_switch", order=2),
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
# IO
# ================================================================

def load_npz_series(data_dir: Path, dataset_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    p = Path(data_dir) / f"{dataset_name}.npz"
    arr = np.load(p, allow_pickle=True)
    y = np.asarray(arr["y"], dtype=float).flatten()
    states = np.asarray(arr["states"], dtype=int).flatten()
    T = np.asarray(arr["T"], dtype=float)
    sigma = arr["sigma"] if "sigma" in arr.files else None
    return y, states, T, sigma


def build_exog_for_dataset(dataset_name: str, n_total: int) -> Optional[np.ndarray]:
    t = np.arange(n_total)

    if dataset_name == "E1_drift_only":
        # non-constant ramp so it is not collinear with the intercept (trend="c")
        x = t.astype(float) / float(max(n_total - 1, 1))
        return x[:, None]

    if dataset_name == "E2_level_shift":
        # step function (0 then 1) so it is not collinear with the intercept
        x = (t >= (n_total // 2)).astype(float)
        return x[:, None]

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
        return model.fit(maxiter=maxiter, em_iter=em_iter, disp=False)
    except LinAlgError as e:
        print(
            f"[warning] msar fit hit LinAlgError with em_iter={em_iter} for {cfg.name}: {e}. "
            "Retrying with em_iter=0."
        )
        try:
            return model.fit(maxiter=maxiter, em_iter=0, disp=False)
        except LinAlgError as e2:
            raise RuntimeError(
                f"msar fit failed due to LinAlgError even after retry with em_iter=0: {e2}"
            ) from e2


# ================================================================
# SHAPE / PADDING HELPERS
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

    Regime decoding uses filtered_marginal_probabilities (online inference) only.
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

    probs_raw = getattr(res_full, "filtered_marginal_probabilities", None)
    probs = _to_2d_array(probs_raw)

    if probs is None:
        decoded = np.full(len(y_full), -1, dtype=int)
        return fv, decoded

    # statsmodels may return shape (k, nobs-order) or (nobs-order, k)
    if probs.shape[1] != cfg.k_regimes and probs.shape[0] == cfg.k_regimes:
        probs = probs.T

    probs = _pad_to_nobs(probs, nobs=len(y_full), order=cfg.order)

    # padded region may contain NaNs; fill with uniform so argmax is defined
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
    data_dir: Path,
    cfg: MSARConfig,
    val_frac: float,
    maxiter: int,
    em_iter: int,
) -> Dict[str, Any]:
    y_raw, true_states, T, sigma = load_npz_series(data_dir, dataset_name)
    n = len(y_raw)
    n_train, n_val = train_val_split_indices(n, val_frac)

    # train-only standardization
    mu = float(np.mean(y_raw[:n_train]))
    std = float(np.std(y_raw[:n_train]) + 1e-8)
    y = (y_raw - mu) / std

    exog_full = build_exog_for_dataset(dataset_name, n_total=n)
    exog_train = exog_full[:n_train] if (cfg.use_exog and exog_full is not None) else None

    # fit on train only
    res_train = fit_markov_ar(y[:n_train], cfg, exog_train, maxiter, em_iter)

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
    # exclude the first `order` points where statsmodels outputs are padded
    start = int(cfg.order)
    if n_train <= start:
        acc = float("nan")
    else:
        dec_train = decoded_full[start:n_train]
        true_train = true_states[start:n_train]
        valid = dec_train >= 0
        if not np.any(valid):
            acc = float("nan")
        else:
            acc_dict = label_corrected_accuracy(dec_train[valid], true_train[valid], cfg.k_regimes)
            acc = float(acc_dict["acc"])

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
        "train_mse": float(train_mse),
        "val_mse": float(val_mse),
        "regime_accuracy": float(acc),
        "per_regime_rmse_train": per_reg_train,
        "per_regime_rmse_val": per_reg_val,
        "noise_rmse": float(noise_rmse),
        "n": int(n),
        "n_train": int(n_train),
        "n_val": int(n_val),
    }


def run_msar(
    dataset_name: str,
    data_dir: Path,
    val_frac: float = 0.2,
    candidate_orders: Optional[List[int]] = None,
    maxiter: int = 150,
    em_iter: int = 10,
) -> Dict[str, Any]:
    if dataset_name not in CONFIGS:
        raise KeyError(f"unknown dataset {dataset_name}")

    base_cfg = CONFIGS[dataset_name]

    if candidate_orders is None:
        candidate_orders = [base_cfg.order]

    # for ARMA/ARIMA-type data, override order grid if given
    if dataset_name in ARMA_ARIMA_DATASETS and candidate_orders is None:
        candidate_orders = [2, 3, 4, 5, 6, 8, 10]

    best: Optional[Dict[str, Any]] = None
    best_val = float("inf")

    for o in candidate_orders:
        cfg = replace(base_cfg, order=int(o))
        try:
            out = evaluate_msar_fixed_order(
                dataset_name=dataset_name,
                data_dir=data_dir,
                cfg=cfg,
                val_frac=val_frac,
                maxiter=maxiter,
                em_iter=em_iter,
            )
        except (LinAlgError, RuntimeError):
            continue

        v = float(out["val_rmse"])
        if v < best_val:
            best_val = v
            best = out

    if best is None:
        raise RuntimeError(f"no msar run succeeded for {dataset_name}")

    best["selected_order"] = int(best["order"])
    return best
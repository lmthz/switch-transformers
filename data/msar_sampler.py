# data/msar_sampler.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch


# ================================================================
# Config
# ================================================================

@dataclass
class MSARSamplerConfig:
    # Series length per sample — should be >= context_len + 1
    series_len: int = 512

    # Number of regimes
    k_regimes: int = 2

    # AR coefficient magnitude
    ar_coeff_scale: float = 0.6

    # MA coefficient magnitude (smaller to keep process stable)
    ma_coeff_scale: float = 0.4

    # Seasonal AR/MA coefficient magnitude
    sar_coeff_scale: float = 0.35

    # Noise std range
    sigma_lo: float = 0.15
    sigma_hi: float = 0.70

    # Regime persistence range
    persistence_lo: float = 0.85
    persistence_hi: float = 0.98

    # Burn-in steps to discard
    burn_in: int = 100

    # Mix weights — must sum to 1
    # AR, ARMA, ARIMA(d=1), ARIMA(d=2), seasonal AR, exogenous drift
    mix_ar:      float = 0.30
    mix_arma:    float = 0.20
    mix_arima1:  float = 0.20
    mix_arima2:  float = 0.08
    mix_seasonal: float = 0.12
    mix_exog:    float = 0.10

    # AR order range
    ar_order_lo: int = 1
    ar_order_hi: int = 5

    # MA order range
    ma_order_lo: int = 1
    ma_order_hi: int = 3

    # Seasonal period range — sampled from a small set of realistic periods
    # 12 = monthly/annual, 24 = hourly/daily, 7 = weekly
    seasonal_periods: tuple = (7, 12, 24)

    # Exogenous beta magnitude range — how strongly X affects y per regime
    exog_beta_lo: float = 0.0
    exog_beta_hi: float = 1.0


# ================================================================
# Helpers
# ================================================================

def _sample_stable_ar_coeffs(
    rng: np.random.Generator,
    order: int,
    scale: float,
) -> np.ndarray:
    if order == 0:
        return np.array([])
    for _ in range(200):
        coeffs = rng.uniform(-scale, scale, size=order)
        poly = np.concatenate([[1], -coeffs])
        roots = np.roots(poly)
        if np.all(np.abs(roots) > 1.0):
            return coeffs
    return rng.uniform(-0.3, 0.3, size=order)


def _sample_transition_matrix(
    rng: np.random.Generator,
    k: int,
    persistence_lo: float,
    persistence_hi: float,
) -> np.ndarray:
    T = np.zeros((k, k))
    for i in range(k):
        stay = rng.uniform(persistence_lo, persistence_hi)
        T[i, i] = stay
        other = (1.0 - stay) / (k - 1)
        for j in range(k):
            if j != i:
                T[i, j] = other
    return T


def _sample_markov_chain(
    rng: np.random.Generator,
    T: np.ndarray,
    n: int,
) -> np.ndarray:
    k = T.shape[0]
    eigvals, eigvecs = np.linalg.eig(T.T)
    stat = np.real(eigvecs[:, np.isclose(np.real(eigvals), 1.0, atol=1e-6)])
    if stat.shape[1] == 0:
        stat = np.ones(k) / k
    else:
        stat = stat[:, 0]
        stat = np.abs(stat) / np.abs(stat).sum()
    states = np.empty(n, dtype=int)
    states[0] = rng.choice(k, p=stat)
    for t in range(1, n):
        states[t] = rng.choice(k, p=T[states[t - 1]])
    return states


def _standardize(y: np.ndarray) -> np.ndarray:
    mu, std = y.mean(), y.std()
    if std > 1e-6:
        return (y - mu) / std
    return y - mu


# ================================================================
# Process family simulators
# ================================================================

def _simulate_ar(rng: np.random.Generator, cfg: MSARSamplerConfig) -> np.ndarray:
    """Pure AR(p) Markov-switching."""
    k = cfg.k_regimes
    total = cfg.series_len + cfg.burn_in
    p = int(rng.integers(cfg.ar_order_lo, cfg.ar_order_hi + 1))

    ar = np.stack([_sample_stable_ar_coeffs(rng, p, cfg.ar_coeff_scale) for _ in range(k)])
    sigmas = rng.uniform(cfg.sigma_lo, cfg.sigma_hi, size=k)
    T = _sample_transition_matrix(rng, k, cfg.persistence_lo, cfg.persistence_hi)
    states = _sample_markov_chain(rng, T, total)

    y = np.zeros(total)
    for t in range(p, total):
        s = states[t]
        y[t] = np.dot(ar[s], y[t - np.arange(1, p + 1)]) + rng.normal(scale=sigmas[s])

    return _standardize(y[cfg.burn_in:])


def _simulate_arma(rng: np.random.Generator, cfg: MSARSamplerConfig) -> np.ndarray:
    """ARMA(p, q) Markov-switching."""
    k = cfg.k_regimes
    total = cfg.series_len + cfg.burn_in
    p = int(rng.integers(cfg.ar_order_lo, cfg.ar_order_hi + 1))
    q = int(rng.integers(cfg.ma_order_lo, cfg.ma_order_hi + 1))

    ar = np.stack([_sample_stable_ar_coeffs(rng, p, cfg.ar_coeff_scale) for _ in range(k)])
    ma = rng.uniform(-cfg.ma_coeff_scale, cfg.ma_coeff_scale, size=(k, q))
    sigmas = rng.uniform(cfg.sigma_lo, cfg.sigma_hi, size=k)
    T = _sample_transition_matrix(rng, k, cfg.persistence_lo, cfg.persistence_hi)
    states = _sample_markov_chain(rng, T, total)

    y = np.zeros(total)
    eps = np.zeros(total)
    start = max(p, q)
    for t in range(start, total):
        s = states[t]
        eps[t] = rng.normal(scale=sigmas[s])
        ar_part = np.dot(ar[s], y[t - np.arange(1, p + 1)]) if p else 0.0
        ma_part = np.dot(ma[s], eps[t - np.arange(1, q + 1)]) if q else 0.0
        y[t] = ar_part + ma_part + eps[t]

    return _standardize(y[cfg.burn_in:])


def _simulate_arima(rng: np.random.Generator, cfg: MSARSamplerConfig, d: int) -> np.ndarray:
    """ARIMA(p, d, q) Markov-switching via integration of ARMA innovations."""
    k = cfg.k_regimes
    total = cfg.series_len + cfg.burn_in
    p = int(rng.integers(cfg.ar_order_lo, cfg.ar_order_hi + 1))
    q = int(rng.integers(0, cfg.ma_order_hi + 1))

    ar = np.stack([_sample_stable_ar_coeffs(rng, p, cfg.ar_coeff_scale) for _ in range(k)])
    ma = rng.uniform(-cfg.ma_coeff_scale, cfg.ma_coeff_scale, size=(k, q)) if q > 0 else None
    sigmas = rng.uniform(cfg.sigma_lo * 0.3, cfg.sigma_hi * 0.3, size=k)
    T = _sample_transition_matrix(rng, k, cfg.persistence_lo, cfg.persistence_hi)
    states = _sample_markov_chain(rng, T, total)

    z = np.zeros(total)
    eps = np.zeros(total)
    start = max(p, q) if q > 0 else p
    for t in range(start, total):
        s = states[t]
        eps[t] = rng.normal(scale=sigmas[s])
        ar_part = np.dot(ar[s], z[t - np.arange(1, p + 1)]) if p else 0.0
        ma_part = np.dot(ma[s], eps[t - np.arange(1, q + 1)]) if (q > 0 and ma is not None) else 0.0
        z[t] = ar_part + ma_part + eps[t]

    y = z.copy()
    for _ in range(d):
        y = np.cumsum(y)

    y = y[cfg.burn_in:]
    y = np.clip(y, -1e6, 1e6)
    return _standardize(y)


def _simulate_seasonal(rng: np.random.Generator, cfg: MSARSamplerConfig) -> np.ndarray:
    """
    Seasonal AR Markov-switching.

    y_t = AR_nonseasonal(y) + SAR_seasonal(y) + noise

    The seasonal AR term adds lags at multiples of the period s
    (e.g. s=24: lags 24, 48). Regime switching affects both the
    nonseasonal and seasonal AR coefficients, matching F1/F2 datasets.
    """
    k = cfg.k_regimes
    total = cfg.series_len + cfg.burn_in
    p = int(rng.integers(1, 3))                            # nonseasonal AR order 1-2
    P = 1                                                   # seasonal AR order (always 1)
    s = int(rng.choice(list(cfg.seasonal_periods)))         # seasonal period

    ar  = np.stack([_sample_stable_ar_coeffs(rng, p, cfg.ar_coeff_scale) for _ in range(k)])
    sar = rng.uniform(-cfg.sar_coeff_scale, cfg.sar_coeff_scale, size=(k, P))
    sigmas = rng.uniform(cfg.sigma_lo, cfg.sigma_hi, size=k)
    T = _sample_transition_matrix(rng, k, cfg.persistence_lo, cfg.persistence_hi)
    states = _sample_markov_chain(rng, T, total)

    start = max(p, s * P)
    y = np.zeros(total)
    for t in range(start, total):
        s_t = states[t]
        ar_part  = np.dot(ar[s_t],  y[t - np.arange(1, p + 1)])
        sar_part = np.dot(sar[s_t], y[t - s * np.arange(1, P + 1)])
        y[t] = ar_part + sar_part + rng.normal(scale=sigmas[s_t])

    return _standardize(y[cfg.burn_in:])


def _simulate_exog_drift(rng: np.random.Generator, cfg: MSARSamplerConfig) -> np.ndarray:
    """
    AR with exogenous drift Markov-switching.

    y_t = AR(y) + beta_regime * X_t + noise

    X_t is a sine wave with random period and phase — matching E1/E2/G1
    datasets where the regime switches the strength or sign of the
    exogenous effect. The exogenous signal is baked into the observed y,
    so the transformer can learn to exploit it from the series alone.
    """
    k = cfg.k_regimes
    total = cfg.series_len + cfg.burn_in
    p = int(rng.integers(cfg.ar_order_lo, min(3, cfg.ar_order_hi) + 1))

    ar = np.stack([_sample_stable_ar_coeffs(rng, p, cfg.ar_coeff_scale) for _ in range(k)])
    sigmas = rng.uniform(cfg.sigma_lo, cfg.sigma_hi, size=k)

    # sample exogenous betas — regimes differ in their response to X
    # one regime can have beta=0 (no effect), the other has nonzero beta
    betas = np.zeros(k)
    betas[0] = rng.uniform(cfg.exog_beta_lo, cfg.exog_beta_hi)
    betas[1] = rng.uniform(-cfg.exog_beta_hi, cfg.exog_beta_lo)  # opposite sign or zero

    # random sinusoidal exogenous variable
    period = rng.uniform(8, 48)
    phase = rng.uniform(0, 2 * np.pi)
    t_arr = np.arange(total)
    X = np.sin(2 * np.pi * t_arr / period + phase)

    T = _sample_transition_matrix(rng, k, cfg.persistence_lo, cfg.persistence_hi)
    states = _sample_markov_chain(rng, T, total)

    y = np.zeros(total)
    for t in range(p, total):
        s = states[t]
        ar_part = np.dot(ar[s], y[t - np.arange(1, p + 1)])
        y[t] = ar_part + betas[s] * X[t] + rng.normal(scale=sigmas[s])

    return _standardize(y[cfg.burn_in:])


# ================================================================
# Mixed sampler
# ================================================================

class MSARBatchSampler:
    """
    Generates batches of fresh switching time series on the fly,
    drawn from a mixture of AR, ARMA, ARIMA(d=1), ARIMA(d=2),
    seasonal AR, and AR-with-exogenous-drift families.

    Covers all dataset families in the evaluation suite.
    """

    def __init__(self, cfg: MSARSamplerConfig, seed: Optional[int] = None):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self._mix_weights = np.array([
            cfg.mix_ar,
            cfg.mix_arma,
            cfg.mix_arima1,
            cfg.mix_arima2,
            cfg.mix_seasonal,
            cfg.mix_exog,
        ])
        assert abs(self._mix_weights.sum() - 1.0) < 1e-6, \
            f"mix weights must sum to 1, got {self._mix_weights.sum()}"

    def _sample_one_series(self) -> np.ndarray:
        family = self.rng.choice(6, p=self._mix_weights)
        if family == 0:
            return _simulate_ar(self.rng, self.cfg)
        elif family == 1:
            return _simulate_arma(self.rng, self.cfg)
        elif family == 2:
            return _simulate_arima(self.rng, self.cfg, d=1)
        elif family == 3:
            return _simulate_arima(self.rng, self.cfg, d=2)
        elif family == 4:
            return _simulate_seasonal(self.rng, self.cfg)
        else:
            return _simulate_exog_drift(self.rng, self.cfg)

    def sample_batch(
        self,
        batch_size: int,
        context_len: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            x: (batch_size, context_len, 1)  — context windows
            y: (batch_size, 1)               — next-step targets
        """
        xs = []
        ys = []

        while len(xs) < batch_size:
            series = self._sample_one_series()

            if not np.isfinite(series).all() or series.std() < 1e-6:
                continue

            max_start = len(series) - context_len - 1
            if max_start <= 0:
                continue
            start = self.rng.integers(0, max_start)
            ctx = series[start: start + context_len].astype(np.float32)
            tgt = series[start + context_len].astype(np.float32)

            if not (np.isfinite(ctx).all() and np.isfinite(tgt)):
                continue

            xs.append(ctx)
            ys.append(tgt)

        x_t = torch.from_numpy(np.stack(xs)).unsqueeze(-1).to(device)
        y_t = torch.from_numpy(np.array(ys, dtype=np.float32)).unsqueeze(-1).to(device)
        return x_t, y_t

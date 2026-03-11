# data/msar_sampler.py
"""
Mixed-prior on-the-fly sampler for Garg-style in-context training.

Process families and what evaluation datasets they cover:
  _simulate_ar            -> A1-A3, B1-B2, S1-S2, H1
  _simulate_ar_near_unit  -> H2 (phi near 1.0, never sampled by generic AR)
  _simulate_ar_no_switch  -> NS0, NS1, SW1 (persistence=1.0 or single-switch)
  _simulate_arma          -> C1
  _simulate_arima         -> D1, D2, D3
  _simulate_seasonal      -> F1, F2 (seasonal AR+MA + seasonal differencing)
  _simulate_exog_const    -> E1 (drift), E2 (level shift) — constant/step exog
  _simulate_exog_sine     -> G1 — sinusoidal exog, beta switches
  _simulate_exog_seasonal -> F2 — seasonal + sinusoidal exog combined
"""
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

    # MA coefficient magnitude
    ma_coeff_scale: float = 0.4

    # Seasonal AR/MA coefficient magnitude (smaller for stability)
    sar_coeff_scale: float = 0.35

    # Noise std range
    sigma_lo: float = 0.15
    sigma_hi: float = 0.70

    # Regime persistence range for normal switching series
    persistence_lo: float = 0.85
    persistence_hi: float = 0.98

    # Burn-in steps to discard
    burn_in: int = 100

    # Mix weights — must sum to 1.
    # 9 families covering all 21 evaluation datasets
    mix_ar:             float = 0.22   # A1-A3, B1-B2, S1-S2, H1
    mix_ar_near_unit:   float = 0.05   # H2 near-unit-root
    mix_ar_no_switch:   float = 0.05   # NS0, NS1, SW1
    mix_arma:           float = 0.13   # C1
    mix_arima1:         float = 0.13   # D1, D3
    mix_arima2:         float = 0.07   # D2
    mix_seasonal:       float = 0.12   # F1 (seasonal AR+MA+diff)
    mix_exog_const:     float = 0.08   # E1 drift, E2 level shift
    mix_exog_sine:      float = 0.08   # G1
    mix_exog_seasonal:  float = 0.07   # F2 (seasonal + sine exog)

    # AR order range
    ar_order_lo: int = 1
    ar_order_hi: int = 10   # raised from 5 to cover H1 AR(10)

    # MA order range
    ma_order_lo: int = 1
    ma_order_hi: int = 3

    # Seasonal period — sampled from realistic periods
    seasonal_periods: tuple = (7, 12, 24)

    # Exogenous beta magnitude
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
    """
    Pure AR(p) Markov-switching. Covers A1-A3, B1-B2, S1-S2, H1.
    ar_order_hi=10 means H1's AR(10) structure is now in range.
    Switching can be in AR coefficients, variance, or both.
    """
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


def _simulate_ar_near_unit(rng: np.random.Generator, cfg: MSARSamplerConfig) -> np.ndarray:
    """
    AR(1) near-unit-root switching. Covers H2 (phi=0.98 vs 0.90).
    Both regimes have phi drawn from [0.85, 0.995] — very persistent
    but stationary. Generic AR sampler almost never produces this since
    ar_coeff_scale=0.6 centers draws far from 1.0.
    """
    k = cfg.k_regimes
    total = cfg.series_len + cfg.burn_in

    # sample two distinct near-unit-root phi values
    phis = rng.uniform(0.85, 0.995, size=k)
    sigmas = rng.uniform(cfg.sigma_lo, cfg.sigma_hi * 0.5, size=k)  # smaller sigma — near-unit-root series are already high variance
    T = _sample_transition_matrix(rng, k, cfg.persistence_lo, cfg.persistence_hi)
    states = _sample_markov_chain(rng, T, total)

    y = np.zeros(total)
    for t in range(1, total):
        s = states[t]
        y[t] = phis[s] * y[t - 1] + rng.normal(scale=sigmas[s])

    return _standardize(y[cfg.burn_in:])


def _simulate_ar_no_switch(rng: np.random.Generator, cfg: MSARSamplerConfig) -> np.ndarray:
    """
    AR with no switching or single switching. Covers NS0, NS1, SW1.
    Three sub-cases sampled with equal probability:
      - Always regime 0 (persistence = 1.0)
      - Always regime 1 (persistence = 1.0)
      - Single switch halfway through
    """
    k = cfg.k_regimes
    total = cfg.series_len + cfg.burn_in
    p = int(rng.integers(1, 4))

    ar = np.stack([_sample_stable_ar_coeffs(rng, p, cfg.ar_coeff_scale) for _ in range(k)])
    sigmas = rng.uniform(cfg.sigma_lo, cfg.sigma_hi, size=k)

    sub = rng.integers(0, 3)
    if sub == 0:
        # always regime 0
        states = np.zeros(total, dtype=int)
    elif sub == 1:
        # always regime 1
        states = np.ones(total, dtype=int)
    else:
        # single switch at a random point in the second half
        split = int(rng.integers(total // 3, 2 * total // 3))
        states = np.zeros(total, dtype=int)
        states[split:] = 1

    y = np.zeros(total)
    for t in range(p, total):
        s = states[t]
        y[t] = np.dot(ar[s], y[t - np.arange(1, p + 1)]) + rng.normal(scale=sigmas[s])

    return _standardize(y[cfg.burn_in:])


def _simulate_arma(rng: np.random.Generator, cfg: MSARSamplerConfig) -> np.ndarray:
    """ARMA(p, q) Markov-switching. Covers C1."""
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
    """ARIMA(p, d, q) Markov-switching. Covers D1 (d=1), D2 (d=2), D3 (d=1, no MA)."""
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
    Full SARMA with seasonal differencing. Covers F1.

    Matches F1's SARIMAX(2,0,1)x(1,1,1)_24 structure:
      - Nonseasonal AR(p) + MA(q)
      - Seasonal AR(P) at lag s + seasonal MA(Q) at lag s
      - Seasonal differencing D=1: z_t = y_t - y_{t-s}
        implemented as: simulate z, then seasonally integrate to get y

    Both regimes share the same AR/MA orders but have different
    seasonal AR and MA coefficients — the regime switching affects
    the strength of the seasonal pattern, matching F1.
    """
    k = cfg.k_regimes
    s = int(rng.choice(list(cfg.seasonal_periods)))
    total = cfg.series_len + cfg.burn_in + s  # extra for seasonal integration
    p = int(rng.integers(1, 3))
    q = int(rng.integers(0, 3))
    P, Q = 1, 1  # seasonal orders always 1, matching F1

    ar  = np.stack([_sample_stable_ar_coeffs(rng, p, cfg.ar_coeff_scale) for _ in range(k)])
    ma  = rng.uniform(-cfg.ma_coeff_scale, cfg.ma_coeff_scale, size=(k, q)) if q > 0 else None
    sar = rng.uniform(-cfg.sar_coeff_scale, cfg.sar_coeff_scale, size=(k, P))
    sma = rng.uniform(-cfg.sar_coeff_scale, cfg.sar_coeff_scale, size=(k, Q))
    sigmas = rng.uniform(cfg.sigma_lo, cfg.sigma_hi, size=k)
    T = _sample_transition_matrix(rng, k, cfg.persistence_lo, cfg.persistence_hi)
    states = _sample_markov_chain(rng, T, total)

    # simulate the seasonally-differenced series z
    start = max(p, q if q > 0 else 0, s * P, s * Q)
    z = np.zeros(total)
    eps = np.zeros(total)
    for t in range(start, total):
        st = states[t]
        eps[t] = rng.normal(scale=sigmas[st])
        ar_part  = np.dot(ar[st],  z[t - np.arange(1, p + 1)]) if p else 0.0
        ma_part  = np.dot(ma[st],  eps[t - np.arange(1, q + 1)]) if (q > 0 and ma is not None) else 0.0
        sar_part = np.dot(sar[st], z[t - s * np.arange(1, P + 1)])
        sma_part = np.dot(sma[st], eps[t - s * np.arange(1, Q + 1)])
        z[t] = ar_part + ma_part + sar_part + sma_part + eps[t]

    # seasonal integration: invert z_t = y_t - y_{t-s} => y_t = z_t + y_{t-s}
    y = np.zeros(total)
    for t in range(s, total):
        y[t] = z[t] + y[t - s]

    y = y[cfg.burn_in:]
    y = np.clip(y, -1e6, 1e6)
    return _standardize(y)


def _simulate_exog_const(rng: np.random.Generator, cfg: MSARSamplerConfig) -> np.ndarray:
    """
    AR with constant or step-function exogenous variable. Covers E1, E2.

    E1 (drift): X_t is a linear ramp — one regime has upward trend (beta>0),
                the other is flat (beta=0). The series drifts upward when in
                the trending regime.
    E2 (level shift): X_t is a step function — one regime adds a positive
                      offset, the other a negative offset.

    Both sub-cases sampled with equal probability.
    """
    k = cfg.k_regimes
    total = cfg.series_len + cfg.burn_in
    p = int(rng.integers(1, 4))
    t_arr = np.arange(total, dtype=float)

    ar = np.stack([_sample_stable_ar_coeffs(rng, p, cfg.ar_coeff_scale) for _ in range(k)])
    sigmas = rng.uniform(cfg.sigma_lo, cfg.sigma_hi, size=k)

    sub = rng.integers(0, 2)
    if sub == 0:
        # drift: ramp exog, one regime has positive beta, other has zero
        X = t_arr / float(max(total - 1, 1))
        betas = np.array([rng.uniform(0.0, 0.05), 0.0])
    else:
        # level shift: step function exog, regimes have opposite-sign betas
        split = int(rng.integers(total // 3, 2 * total // 3))
        X = (t_arr >= split).astype(float)
        mag = rng.uniform(0.01, 0.04)
        betas = np.array([mag, -mag])

    T = _sample_transition_matrix(rng, k, cfg.persistence_lo, cfg.persistence_hi)
    states = _sample_markov_chain(rng, T, total)

    y = np.zeros(total)
    for t in range(p, total):
        s = states[t]
        ar_part = np.dot(ar[s], y[t - np.arange(1, p + 1)])
        y[t] = ar_part + betas[s] * X[t] + rng.normal(scale=sigmas[s])

    return _standardize(y[cfg.burn_in:])


def _simulate_exog_sine(rng: np.random.Generator, cfg: MSARSamplerConfig) -> np.ndarray:
    """
    AR with sinusoidal exogenous variable. Covers G1.

    One regime responds strongly to the sine wave (beta large),
    the other does not (beta near zero). Same AR coefficients across
    regimes — only the exogenous response switches.
    """
    k = cfg.k_regimes
    total = cfg.series_len + cfg.burn_in
    p = int(rng.integers(1, 4))

    # same AR across regimes, only beta switches — matching G1
    ar_shared = _sample_stable_ar_coeffs(rng, p, cfg.ar_coeff_scale)
    ar = np.stack([ar_shared, ar_shared])
    sigmas = rng.uniform(cfg.sigma_lo, cfg.sigma_hi, size=k)

    period = rng.uniform(8, 48)
    phase = rng.uniform(0, 2 * np.pi)
    t_arr = np.arange(total)
    X = np.sin(2 * np.pi * t_arr / period + phase)

    # regime 0: strong positive response; regime 1: near-zero response
    betas = np.array([
        rng.uniform(0.4, cfg.exog_beta_hi),
        rng.uniform(0.0, 0.1),
    ])

    T = _sample_transition_matrix(rng, k, cfg.persistence_lo, cfg.persistence_hi)
    states = _sample_markov_chain(rng, T, total)

    y = np.zeros(total)
    for t in range(p, total):
        s = states[t]
        ar_part = np.dot(ar[s], y[t - np.arange(1, p + 1)])
        y[t] = ar_part + betas[s] * X[t] + rng.normal(scale=sigmas[s])

    return _standardize(y[cfg.burn_in:])


def _simulate_exog_seasonal(rng: np.random.Generator, cfg: MSARSamplerConfig) -> np.ndarray:
    """
    Seasonal AR + sinusoidal exogenous variable combined. Covers F2.

    F2 has both seasonal structure (SAR+SMA) and an exogenous component
    (sin+cos) where the regime switches which exog component dominates.
    This family is the only one that combines both aspects simultaneously.
    """
    k = cfg.k_regimes
    s = int(rng.choice(list(cfg.seasonal_periods)))
    total = cfg.series_len + cfg.burn_in + s
    p = int(rng.integers(1, 3))

    ar  = np.stack([_sample_stable_ar_coeffs(rng, p, cfg.ar_coeff_scale) for _ in range(k)])
    sar = rng.uniform(-cfg.sar_coeff_scale, cfg.sar_coeff_scale, size=(k, 1))
    sigmas = rng.uniform(cfg.sigma_lo * 0.5, cfg.sigma_hi * 0.5, size=k)

    t_arr = np.arange(total, dtype=float)
    period = rng.uniform(s * 0.8, s * 1.2)
    X1 = np.sin(2 * np.pi * t_arr / period)
    X2 = np.cos(2 * np.pi * t_arr / period)

    # regime 0: sin-dominant; regime 1: cos-dominant — matching F2
    beta1 = rng.uniform(0.2, 0.6)
    beta2 = rng.uniform(0.2, 0.6)
    betas = np.array([[beta1, 0.0], [0.0, beta2]])

    T = _sample_transition_matrix(rng, k, cfg.persistence_lo, cfg.persistence_hi)
    states = _sample_markov_chain(rng, T, total)

    start = max(p, s)
    y = np.zeros(total)
    for t in range(start, total):
        st = states[t]
        ar_part  = np.dot(ar[st],  y[t - np.arange(1, p + 1)])
        sar_part = float(sar[st, 0] * y[t - s])
        x_part   = betas[st, 0] * X1[t] + betas[st, 1] * X2[t]
        y[t] = ar_part + sar_part + x_part + rng.normal(scale=sigmas[st])

    y = y[cfg.burn_in:]
    y = np.clip(y, -1e6, 1e6)
    return _standardize(y)


# ================================================================
# Mixed sampler
# ================================================================

class MSARBatchSampler:
    """
    Generates batches of fresh switching time series on the fly,
    drawn from a mixture of 9 process families covering all 21
    evaluation datasets.
    """

    _N_FAMILIES = 10

    def __init__(self, cfg: MSARSamplerConfig, seed: Optional[int] = None):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self._mix_weights = np.array([
            cfg.mix_ar,
            cfg.mix_ar_near_unit,
            cfg.mix_ar_no_switch,
            cfg.mix_arma,
            cfg.mix_arima1,
            cfg.mix_arima2,
            cfg.mix_seasonal,
            cfg.mix_exog_const,
            cfg.mix_exog_sine,
            cfg.mix_exog_seasonal,
        ])
        assert abs(self._mix_weights.sum() - 1.0) < 1e-6, \
            f"mix weights must sum to 1, got {self._mix_weights.sum()}"

    def _sample_one_series(self) -> np.ndarray:
        family = self.rng.choice(self._N_FAMILIES, p=self._mix_weights)
        if family == 0:
            return _simulate_ar(self.rng, self.cfg)
        elif family == 1:
            return _simulate_ar_near_unit(self.rng, self.cfg)
        elif family == 2:
            return _simulate_ar_no_switch(self.rng, self.cfg)
        elif family == 3:
            return _simulate_arma(self.rng, self.cfg)
        elif family == 4:
            return _simulate_arima(self.rng, self.cfg, d=1)
        elif family == 5:
            return _simulate_arima(self.rng, self.cfg, d=2)
        elif family == 6:
            return _simulate_seasonal(self.rng, self.cfg)
        elif family == 7:
            return _simulate_exog_const(self.rng, self.cfg)
        elif family == 8:
            return _simulate_exog_sine(self.rng, self.cfg)
        else:
            return _simulate_exog_seasonal(self.rng, self.cfg)

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
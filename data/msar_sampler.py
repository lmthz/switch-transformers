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

    # Noise std range
    sigma_lo: float = 0.15
    sigma_hi: float = 0.70

    # Regime persistence range
    persistence_lo: float = 0.85
    persistence_hi: float = 0.98

    # Burn-in steps to discard
    burn_in: int = 100

    # Mix weights for each process family — must sum to 1.
    # AR, ARMA, ARIMA(d=1), ARIMA(d=2)
    mix_ar: float = 0.40
    mix_arma: float = 0.25
    mix_arima1: float = 0.25
    mix_arima2: float = 0.10

    # AR order range — sampled uniformly as int in [ar_order_lo, ar_order_hi]
    ar_order_lo: int = 1
    ar_order_hi: int = 5

    # MA order range
    ma_order_lo: int = 1
    ma_order_hi: int = 3


# ================================================================
# Helpers
# ================================================================

def _sample_stable_ar_coeffs(
    rng: np.random.Generator,
    order: int,
    scale: float,
) -> np.ndarray:
    """
    Sample AR coefficients with rejection sampling for stationarity.
    Roots of characteristic polynomial must lie outside unit circle.
    """
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

def _simulate_ar(
    rng: np.random.Generator,
    cfg: MSARSamplerConfig,
) -> np.ndarray:
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


def _simulate_arma(
    rng: np.random.Generator,
    cfg: MSARSamplerConfig,
) -> np.ndarray:
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


def _simulate_arima(
    rng: np.random.Generator,
    cfg: MSARSamplerConfig,
    d: int,
) -> np.ndarray:
    """
    ARIMA(p, d, q) Markov-switching.
    Simulate stationary ARMA innovations z, then integrate d times to get y.
    """
    k = cfg.k_regimes
    total = cfg.series_len + cfg.burn_in
    p = int(rng.integers(cfg.ar_order_lo, cfg.ar_order_hi + 1))
    q = int(rng.integers(0, cfg.ma_order_hi + 1))

    ar = np.stack([_sample_stable_ar_coeffs(rng, p, cfg.ar_coeff_scale) for _ in range(k)])
    ma = rng.uniform(-cfg.ma_coeff_scale, cfg.ma_coeff_scale, size=(k, q)) if q > 0 else None
    # smaller sigma for integrated processes to avoid explosions
    sigmas = rng.uniform(cfg.sigma_lo * 0.3, cfg.sigma_hi * 0.3, size=k)
    T = _sample_transition_matrix(rng, k, cfg.persistence_lo, cfg.persistence_hi)
    states = _sample_markov_chain(rng, T, total)

    # simulate stationary innovations z
    z = np.zeros(total)
    eps = np.zeros(total)
    start = max(p, q) if q > 0 else p
    for t in range(start, total):
        s = states[t]
        eps[t] = rng.normal(scale=sigmas[s])
        ar_part = np.dot(ar[s], z[t - np.arange(1, p + 1)]) if p else 0.0
        ma_part = np.dot(ma[s], eps[t - np.arange(1, q + 1)]) if (q > 0 and ma is not None) else 0.0
        z[t] = ar_part + ma_part + eps[t]

    # integrate d times
    y = z.copy()
    for _ in range(d):
        y = np.cumsum(y)

    y = y[cfg.burn_in:]

    # clip extreme values before standardizing to avoid inf
    y = np.clip(y, -1e6, 1e6)
    return _standardize(y)


# ================================================================
# Mixed sampler
# ================================================================

class MSARBatchSampler:
    """
    Generates batches of fresh switching time series on the fly,
    drawn from a mixture of AR, ARMA, ARIMA(d=1), ARIMA(d=2) families.

    The transformer learns a broad prior over all switching dynamics
    rather than any single process family.
    """

    def __init__(self, cfg: MSARSamplerConfig, seed: Optional[int] = None):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        # precompute cumulative mix weights for family selection
        self._mix_weights = np.array([
            cfg.mix_ar,
            cfg.mix_arma,
            cfg.mix_arima1,
            cfg.mix_arima2,
        ])
        assert abs(self._mix_weights.sum() - 1.0) < 1e-6, "mix weights must sum to 1"

    def _sample_one_series(self) -> np.ndarray:
        family = self.rng.choice(4, p=self._mix_weights)
        if family == 0:
            return _simulate_ar(self.rng, self.cfg)
        elif family == 1:
            return _simulate_arma(self.rng, self.cfg)
        elif family == 2:
            return _simulate_arima(self.rng, self.cfg, d=1)
        else:
            return _simulate_arima(self.rng, self.cfg, d=2)

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

            # skip degenerate series (all zeros, NaN, Inf)
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

        x_t = torch.from_numpy(np.stack(xs)).unsqueeze(-1).to(device)   # (B, L, 1)
        y_t = torch.from_numpy(np.array(ys, dtype=np.float32)).unsqueeze(-1).to(device)  # (B, 1)
        return x_t, y_t

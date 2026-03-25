# data/msar_sampler.py
"""
Vectorized mixed-prior sampler for Garg-style in-context training.

Key performance change: instead of a Python for loop that generates one
series at a time, every family simulator generates all B series in the
batch simultaneously. The time loop runs `total` (~600) iterations, but
each iteration is a vectorized numpy op over the full batch dimension B.

Before: B * total = 128 * 612 = ~78k Python loop iterations per batch
After:  total = ~612 Python loop iterations, each doing numpy C-ops over B

Process families:
  _simulate_ar_batch            -> A1-A3, B1-B2, S1-S2, H1
  _simulate_ar_near_unit_batch  -> H2
  _simulate_ar_no_switch_batch  -> NS0, NS1, SW1
  _simulate_arma_batch          -> C1
  _simulate_arima_batch         -> D1 (d=1), D2 (d=2), D3
  _simulate_seasonal_batch      -> F1
  _simulate_exog_const_batch    -> E1, E2
  _simulate_exog_sine_batch     -> G1
  _simulate_exog_seasonal_batch -> F2
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
    series_len: int = 512
    k_regimes: int = 2
    ar_coeff_scale: float = 0.6
    ma_coeff_scale: float = 0.4
    sar_coeff_scale: float = 0.35
    sigma_lo: float = 0.15
    sigma_hi: float = 0.70
    persistence_lo: float = 0.85
    persistence_hi: float = 0.98
    burn_in: int = 100
    mix_ar:             float = 0.22
    mix_ar_near_unit:   float = 0.05
    mix_ar_no_switch:   float = 0.05
    mix_arma:           float = 0.13
    mix_arima1:         float = 0.13
    mix_arima2:         float = 0.07
    mix_seasonal:       float = 0.12
    mix_exog_const:     float = 0.08
    mix_exog_sine:      float = 0.08
    mix_exog_seasonal:  float = 0.07
    ar_order_lo: int = 1
    ar_order_hi: int = 10
    ma_order_lo: int = 1
    ma_order_hi: int = 3
    seasonal_periods: tuple = (7, 12, 24)
    exog_beta_lo: float = 0.0
    exog_beta_hi: float = 1.0


# ================================================================
# Scalar helper — used only for AR coefficient rejection sampling
# ================================================================

def _sample_stable_ar_coeffs(rng, order, scale):
    if order == 0:
        return np.array([])
    for _ in range(200):
        c = rng.uniform(-scale, scale, size=order)
        if np.all(np.abs(np.roots(np.concatenate([[1], -c]))) > 1.0):
            return c
    return rng.uniform(-0.3, 0.3, size=order)


# ================================================================
# Vectorized batch helpers
# ================================================================

def _sample_ar_coeffs_batch(rng, B, k, order, scale):
    """Returns (B, k, order) stable AR coefficients."""
    if order == 0:
        return np.zeros((B, k, 0))
    ar = np.zeros((B, k, order))
    for b in range(B):
        for r in range(k):
            ar[b, r] = _sample_stable_ar_coeffs(rng, order, scale)
    return ar


def _sample_markov_chains_batch(rng, k, persistence_lo, persistence_hi, B, n):
    """
    Sample B Markov chains. Transition loop is vectorized over B.
    Returns states (B, n).
    """
    stay = rng.uniform(persistence_lo, persistence_hi, size=(B, k))
    T_all = np.zeros((B, k, k))
    for i in range(k):
        T_all[:, i, i] = stay[:, i]
        off = (1.0 - stay[:, i]) / (k - 1)
        for j in range(k):
            if j != i:
                T_all[:, i, j] = off

    states = np.zeros((B, n), dtype=int)
    for b in range(B):
        eigvals, eigvecs = np.linalg.eig(T_all[b].T)
        stat = np.real(eigvecs[:, np.isclose(np.real(eigvals), 1.0, atol=1e-6)])
        if stat.shape[1] == 0:
            stat_b = np.ones(k) / k
        else:
            stat_b = np.abs(stat[:, 0]); stat_b /= stat_b.sum()
        states[b, 0] = rng.choice(k, p=stat_b)

    b_idx = np.arange(B)
    for t in range(1, n):
        curr = states[:, t - 1]
        probs = T_all[b_idx, curr, :]
        cumprobs = np.cumsum(probs, axis=1)
        u = rng.random(B)
        states[:, t] = np.clip((u[:, None] >= cumprobs).sum(axis=1), 0, k - 1)
    return states


def _scale_noise(rng, B, total, sigmas, states):
    """eps[b,t] ~ N(0, sigma[b, regime_t]^2). Returns (B, total)."""
    raw = rng.standard_normal((B, total))
    sigma_t = sigmas[np.arange(B)[:, None], states]
    return raw * sigma_t


def _std_batch(y):
    """Standardize each row of (B, T) independently."""
    mu = y.mean(axis=1, keepdims=True)
    std = y.std(axis=1, keepdims=True)
    return (y - mu) / np.where(std > 1e-6, std, 1.0)


# ================================================================
# Vectorized family simulators — each returns (B, series_len)
# ================================================================

def _simulate_ar_batch(rng, cfg, B):
    k, total = cfg.k_regimes, cfg.series_len + cfg.burn_in
    p = int(rng.integers(cfg.ar_order_lo, cfg.ar_order_hi + 1))
    ar     = _sample_ar_coeffs_batch(rng, B, k, p, cfg.ar_coeff_scale)
    sigmas = rng.uniform(cfg.sigma_lo, cfg.sigma_hi, size=(B, k))
    states = _sample_markov_chains_batch(rng, k, cfg.persistence_lo, cfg.persistence_hi, B, total)
    eps    = _scale_noise(rng, B, total, sigmas, states)
    bi     = np.arange(B)
    y      = np.zeros((B, total))
    for t in range(p, total):
        ar_t   = ar[bi, states[:, t], :]
        y_lags = y[:, t - p:t][:, ::-1]
        y[:, t] = (ar_t * y_lags).sum(1) + eps[:, t]
    return _std_batch(y[:, cfg.burn_in:])


def _simulate_ar_near_unit_batch(rng, cfg, B):
    k, total = cfg.k_regimes, cfg.series_len + cfg.burn_in
    phis   = rng.uniform(0.85, 0.995, size=(B, k))
    sigmas = rng.uniform(cfg.sigma_lo, cfg.sigma_hi * 0.5, size=(B, k))
    states = _sample_markov_chains_batch(rng, k, cfg.persistence_lo, cfg.persistence_hi, B, total)
    eps    = _scale_noise(rng, B, total, sigmas, states)
    bi     = np.arange(B)
    y      = np.zeros((B, total))
    for t in range(1, total):
        phi_t  = phis[bi, states[:, t]]
        y[:, t] = phi_t * y[:, t - 1] + eps[:, t]
    return _std_batch(y[:, cfg.burn_in:])


def _simulate_ar_no_switch_batch(rng, cfg, B):
    k, total = cfg.k_regimes, cfg.series_len + cfg.burn_in
    p      = int(rng.integers(1, 4))
    ar     = _sample_ar_coeffs_batch(rng, B, k, p, cfg.ar_coeff_scale)
    sigmas = rng.uniform(cfg.sigma_lo, cfg.sigma_hi, size=(B, k))
    sub    = rng.integers(0, 3, size=B)
    states = np.zeros((B, total), dtype=int)
    for b in range(B):
        if sub[b] == 1:
            states[b] = 1
        elif sub[b] == 2:
            split = int(rng.integers(total // 3, 2 * total // 3))
            states[b, split:] = 1
    eps    = _scale_noise(rng, B, total, sigmas, states)
    bi     = np.arange(B)
    y      = np.zeros((B, total))
    for t in range(p, total):
        ar_t   = ar[bi, states[:, t], :]
        y_lags = y[:, t - p:t][:, ::-1]
        y[:, t] = (ar_t * y_lags).sum(1) + eps[:, t]
    return _std_batch(y[:, cfg.burn_in:])


def _simulate_arma_batch(rng, cfg, B):
    k, total = cfg.k_regimes, cfg.series_len + cfg.burn_in
    p = int(rng.integers(cfg.ar_order_lo, cfg.ar_order_hi + 1))
    q = int(rng.integers(cfg.ma_order_lo, cfg.ma_order_hi + 1))
    ar     = _sample_ar_coeffs_batch(rng, B, k, p, cfg.ar_coeff_scale)
    ma     = rng.uniform(-cfg.ma_coeff_scale, cfg.ma_coeff_scale, size=(B, k, q))
    sigmas = rng.uniform(cfg.sigma_lo, cfg.sigma_hi, size=(B, k))
    states = _sample_markov_chains_batch(rng, k, cfg.persistence_lo, cfg.persistence_hi, B, total)
    eps    = _scale_noise(rng, B, total, sigmas, states)
    bi     = np.arange(B)
    y      = np.zeros((B, total))
    for t in range(max(p, q), total):
        ar_t   = ar[bi, states[:, t], :]
        ma_t   = ma[bi, states[:, t], :]
        y_lags = y[:, t - p:t][:, ::-1]
        e_lags = eps[:, t - q:t][:, ::-1]
        y[:, t] = (ar_t * y_lags).sum(1) + (ma_t * e_lags).sum(1) + eps[:, t]
    return _std_batch(y[:, cfg.burn_in:])


def _simulate_arima_batch(rng, cfg, B, d):
    k, total = cfg.k_regimes, cfg.series_len + cfg.burn_in
    p = int(rng.integers(cfg.ar_order_lo, cfg.ar_order_hi + 1))
    q = int(rng.integers(0, cfg.ma_order_hi + 1))
    ar     = _sample_ar_coeffs_batch(rng, B, k, p, cfg.ar_coeff_scale)
    ma     = rng.uniform(-cfg.ma_coeff_scale, cfg.ma_coeff_scale, size=(B, k, q)) if q > 0 else None
    sigmas = rng.uniform(cfg.sigma_lo * 0.3, cfg.sigma_hi * 0.3, size=(B, k))
    states = _sample_markov_chains_batch(rng, k, cfg.persistence_lo, cfg.persistence_hi, B, total)
    eps    = _scale_noise(rng, B, total, sigmas, states)
    bi     = np.arange(B)
    z      = np.zeros((B, total))
    for t in range(max(p, q) if q > 0 else p, total):
        ar_t   = ar[bi, states[:, t], :]
        z_lags = z[:, t - p:t][:, ::-1]
        ma_part = (ma[bi, states[:, t], :] * eps[:, t - q:t][:, ::-1]).sum(1) if (q > 0 and ma is not None) else 0.0
        z[:, t] = (ar_t * z_lags).sum(1) + ma_part + eps[:, t]
    y = z.copy()
    for _ in range(d):
        y = np.cumsum(y, axis=1)
    y = np.clip(y[:, cfg.burn_in:], -1e6, 1e6)
    return _std_batch(y)


def _simulate_seasonal_batch(rng, cfg, B):
    k   = cfg.k_regimes
    s   = int(rng.choice(list(cfg.seasonal_periods)))
    total = cfg.series_len + cfg.burn_in + s
    p   = int(rng.integers(1, 3))
    q   = int(rng.integers(0, 3))
    ar  = _sample_ar_coeffs_batch(rng, B, k, p, cfg.ar_coeff_scale)
    ma  = rng.uniform(-cfg.ma_coeff_scale, cfg.ma_coeff_scale, size=(B, k, q)) if q > 0 else None
    sar = rng.uniform(-cfg.sar_coeff_scale, cfg.sar_coeff_scale, size=(B, k))  # P=1
    sma = rng.uniform(-cfg.sar_coeff_scale, cfg.sar_coeff_scale, size=(B, k))  # Q=1
    sigmas = rng.uniform(cfg.sigma_lo, cfg.sigma_hi, size=(B, k))
    states = _sample_markov_chains_batch(rng, k, cfg.persistence_lo, cfg.persistence_hi, B, total)
    eps    = _scale_noise(rng, B, total, sigmas, states)
    bi     = np.arange(B)
    z      = np.zeros((B, total))
    start  = max(p, q if q > 0 else 0, s, s)
    for t in range(start, total):
        st     = states[:, t]
        ar_t   = ar[bi, st, :]
        z_lags = z[:, t - p:t][:, ::-1]
        sar_t  = sar[bi, st]
        sma_t  = sma[bi, st]
        ma_part = (ma[bi, st, :] * eps[:, t - q:t][:, ::-1]).sum(1) if (q > 0 and ma is not None) else 0.0
        z[:, t] = (ar_t * z_lags).sum(1) + ma_part + sar_t * z[:, t - s] + sma_t * eps[:, t - s] + eps[:, t]
    y = np.zeros((B, total))
    for t in range(s, total):
        y[:, t] = z[:, t] + y[:, t - s]
    y = np.clip(y[:, cfg.burn_in:], -1e6, 1e6)
    return _std_batch(y)


def _simulate_exog_const_batch(rng, cfg, B):
    k, total = cfg.k_regimes, cfg.series_len + cfg.burn_in
    p   = int(rng.integers(1, 4))
    t_a = np.arange(total, dtype=float)
    ar  = _sample_ar_coeffs_batch(rng, B, k, p, cfg.ar_coeff_scale)
    sigmas = rng.uniform(cfg.sigma_lo, cfg.sigma_hi, size=(B, k))
    X     = np.zeros((B, total))
    betas = np.zeros((B, k))
    sub   = rng.integers(0, 2, size=B)
    for b in range(B):
        if sub[b] == 0:
            X[b] = t_a / float(max(total - 1, 1))
            betas[b, 0] = rng.uniform(0.0, 0.05)
        else:
            split = int(rng.integers(total // 3, 2 * total // 3))
            X[b] = (t_a >= split).astype(float)
            mag = rng.uniform(0.01, 0.04)
            betas[b, 0] = mag; betas[b, 1] = -mag
    states = _sample_markov_chains_batch(rng, k, cfg.persistence_lo, cfg.persistence_hi, B, total)
    eps    = _scale_noise(rng, B, total, sigmas, states)
    bi     = np.arange(B)
    y      = np.zeros((B, total))
    for t in range(p, total):
        ar_t   = ar[bi, states[:, t], :]
        beta_t = betas[bi, states[:, t]]
        y_lags = y[:, t - p:t][:, ::-1]
        y[:, t] = (ar_t * y_lags).sum(1) + beta_t * X[:, t] + eps[:, t]
    return _std_batch(y[:, cfg.burn_in:])


def _simulate_exog_sine_batch(rng, cfg, B):
    k, total = cfg.k_regimes, cfg.series_len + cfg.burn_in
    p = int(rng.integers(1, 4))
    ar_sh  = _sample_ar_coeffs_batch(rng, B, 1, p, cfg.ar_coeff_scale)[:, 0, :]  # (B, p)
    ar     = np.stack([ar_sh, ar_sh], axis=1)  # (B, 2, p)
    sigmas = rng.uniform(cfg.sigma_lo, cfg.sigma_hi, size=(B, k))
    periods = rng.uniform(8, 48, size=B)
    phases  = rng.uniform(0, 2 * np.pi, size=B)
    t_a     = np.arange(total, dtype=float)
    X       = np.sin(2 * np.pi * t_a[None, :] / periods[:, None] + phases[:, None])
    betas   = np.stack([rng.uniform(0.4, cfg.exog_beta_hi, size=B),
                        rng.uniform(0.0, 0.1, size=B)], axis=1)  # (B, 2)
    states = _sample_markov_chains_batch(rng, k, cfg.persistence_lo, cfg.persistence_hi, B, total)
    eps    = _scale_noise(rng, B, total, sigmas, states)
    bi     = np.arange(B)
    y      = np.zeros((B, total))
    for t in range(p, total):
        ar_t   = ar[bi, states[:, t], :]
        beta_t = betas[bi, states[:, t]]
        y_lags = y[:, t - p:t][:, ::-1]
        y[:, t] = (ar_t * y_lags).sum(1) + beta_t * X[:, t] + eps[:, t]
    return _std_batch(y[:, cfg.burn_in:])


def _simulate_exog_seasonal_batch(rng, cfg, B):
    k   = cfg.k_regimes
    s   = int(rng.choice(list(cfg.seasonal_periods)))
    total = cfg.series_len + cfg.burn_in + s
    p   = int(rng.integers(1, 3))
    ar  = _sample_ar_coeffs_batch(rng, B, k, p, cfg.ar_coeff_scale)
    sar = rng.uniform(-cfg.sar_coeff_scale, cfg.sar_coeff_scale, size=(B, k))
    sigmas  = rng.uniform(cfg.sigma_lo * 0.5, cfg.sigma_hi * 0.5, size=(B, k))
    t_a     = np.arange(total, dtype=float)
    periods = rng.uniform(s * 0.8, s * 1.2, size=B)
    X1 = np.sin(2 * np.pi * t_a[None, :] / periods[:, None])
    X2 = np.cos(2 * np.pi * t_a[None, :] / periods[:, None])
    betas = np.zeros((B, k, 2))
    betas[:, 0, 0] = rng.uniform(0.2, 0.6, size=B)  # regime 0: sin
    betas[:, 1, 1] = rng.uniform(0.2, 0.6, size=B)  # regime 1: cos
    states = _sample_markov_chains_batch(rng, k, cfg.persistence_lo, cfg.persistence_hi, B, total)
    eps    = _scale_noise(rng, B, total, sigmas, states)
    bi     = np.arange(B)
    y      = np.zeros((B, total))
    for t in range(max(p, s), total):
        st     = states[:, t]
        ar_t   = ar[bi, st, :]
        sar_t  = sar[bi, st]
        beta_t = betas[bi, st, :]
        y_lags = y[:, t - p:t][:, ::-1]
        y[:, t] = (ar_t * y_lags).sum(1) + sar_t * y[:, t - s] + beta_t[:, 0] * X1[:, t] + beta_t[:, 1] * X2[:, t] + eps[:, t]
    y = np.clip(y[:, cfg.burn_in:], -1e6, 1e6)
    return _std_batch(y)


# ================================================================
# Mixed sampler
# ================================================================

class MSARBatchSampler:
    """
    Generates batches of B series simultaneously using vectorized numpy ops.
    No Python loop over batch_size — entire batch generated in one shot.

    Two modes:
      On-the-fly (default): generates fresh synthetic series every call.
      Pool mode: loads a pre-generated pool from disk (via load_pool()) and
                 draws random series from it. Removes all CPU generation
                 overhead from the GPU training loop — series are already
                 in memory, sample_batch just does a random index lookup.
    """

    _N_FAMILIES = 10

    def __init__(self, cfg: MSARSamplerConfig, seed: Optional[int] = None):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self._mix_weights = np.array([
            cfg.mix_ar, cfg.mix_ar_near_unit, cfg.mix_ar_no_switch,
            cfg.mix_arma, cfg.mix_arima1, cfg.mix_arima2,
            cfg.mix_seasonal, cfg.mix_exog_const, cfg.mix_exog_sine, cfg.mix_exog_seasonal,
        ])
        assert abs(self._mix_weights.sum() - 1.0) < 1e-6, \
            f"mix weights must sum to 1, got {self._mix_weights.sum()}"
        self._pool: Optional[np.ndarray] = None   # set by load_pool()

    def load_pool(self, pool_path: str) -> None:
        """
        Load a pre-generated series pool from disk into memory.
        After calling this, sample_batch draws sequentially without
        replacement — each series seen at most once per epoch before
        the pool reshuffles and restarts. Prevents overfitting from
        repeatedly seeing the same series parameters.

        Args:
            pool_path: path to the .npz file produced by generate_pool.py
        """
        print(f"Loading series pool from {pool_path} ...")
        data = np.load(pool_path)
        self._pool = data["series"]          # (N, usable_len) float32
        n, usable_len = self._pool.shape

        # Shuffled index order for sequential no-repeat sampling.
        # _pool_cursor advances by batch_size each call.
        # When exhausted, reshuffle and reset — new epoch, new random windows.
        self._pool_order  = self.rng.permutation(n)
        self._pool_cursor = 0
        self._pool_epochs = 0

        print(
            f"Pool loaded: {n:,} series  x  {usable_len} timesteps  "
            f"dtype={self._pool.dtype}  "
            f"({self._pool.nbytes / 1e6:.0f} MB in memory)"
        )
        if "seed" in data:
            print(f"Pool seed: {int(data['seed'])}")
        print(
            f"Sequential no-repeat: ~{n // 128:,} steps per epoch at batch_size=128"
        )

    def _simulate_batch(self, family: int, B: int) -> np.ndarray:
        fns = [
            lambda B: _simulate_ar_batch(self.rng, self.cfg, B),
            lambda B: _simulate_ar_near_unit_batch(self.rng, self.cfg, B),
            lambda B: _simulate_ar_no_switch_batch(self.rng, self.cfg, B),
            lambda B: _simulate_arma_batch(self.rng, self.cfg, B),
            lambda B: _simulate_arima_batch(self.rng, self.cfg, B, d=1),
            lambda B: _simulate_arima_batch(self.rng, self.cfg, B, d=2),
            lambda B: _simulate_seasonal_batch(self.rng, self.cfg, B),
            lambda B: _simulate_exog_const_batch(self.rng, self.cfg, B),
            lambda B: _simulate_exog_sine_batch(self.rng, self.cfg, B),
            lambda B: _simulate_exog_seasonal_batch(self.rng, self.cfg, B),
        ]
        return fns[family](B)

    def _sample_series_from_pool(self, batch_size: int) -> np.ndarray:
        """
        Draw the next batch_size series from the pool sequentially.
        Each series is seen at most once before the pool reshuffles.
        Returns (batch_size, usable_len) float32 array.
        """
        n_pool = self._pool.shape[0]

        # If not enough series left in current epoch, reshuffle and restart
        if self._pool_cursor + batch_size > n_pool:
            self._pool_order  = self.rng.permutation(n_pool)
            self._pool_cursor = 0
            self._pool_epochs += 1

        idx = self._pool_order[self._pool_cursor : self._pool_cursor + batch_size]
        self._pool_cursor += batch_size
        return self._pool[idx]

    def sample_batch(
        self,
        batch_size: int,
        context_len: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate batch_size (x, y) pairs.

        Pool mode (load_pool() called):
            Draws batch_size series from the in-memory pool via random index.
            No generation overhead — CPU work is ~microseconds.

        On-the-fly mode (default):
            Generates all series simultaneously via vectorized numpy ops.
        """
        if self._pool is not None:
            # ── Pool mode ──────────────────────────────────────────────────
            series = self._sample_series_from_pool(batch_size)
        else:
            # ── On-the-fly mode ────────────────────────────────────────────
            while True:
                family = self.rng.choice(self._N_FAMILIES, p=self._mix_weights)
                candidate = self._simulate_batch(family, batch_size + 8)
                valid = np.isfinite(candidate).all(axis=1) & (candidate.std(axis=1) > 1e-6)
                if valid.sum() >= batch_size:
                    series = candidate[valid][:batch_size]
                    break

        series_len = series.shape[1]
        max_start  = series_len - context_len - 1
        if max_start <= 0:
            raise ValueError(
                f"series_len={series_len} too short for context_len={context_len}. "
                f"{'Pool series are too short — regenerate with larger series_len.' if self._pool is not None else 'Increase cfg.series_len.'}"
            )

        # Vectorized window extraction — same for both modes
        bi     = np.arange(batch_size)
        starts = self.rng.integers(0, max_start, size=batch_size)
        t_idx  = starts[:, None] + np.arange(context_len)[None, :]   # (B, L)
        xs     = series[bi[:, None], t_idx].astype(np.float32)        # (B, L)
        ys     = series[bi, starts + context_len].astype(np.float32)  # (B,)

        # Replace any NaN windows (rare edge case)
        bad = np.where(~(np.isfinite(xs).all(1) & np.isfinite(ys)))[0]
        if len(bad):
            good = int(np.argmax(np.isfinite(xs).all(1) & np.isfinite(ys)))
            xs[bad] = xs[good]; ys[bad] = ys[good]

        x_t = torch.from_numpy(xs).unsqueeze(-1).to(device)
        y_t = torch.from_numpy(ys).unsqueeze(-1).to(device)
        return x_t, y_t
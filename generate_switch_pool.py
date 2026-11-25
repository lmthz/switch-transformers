"""
generate_switch_pool.py

Generates a pool of single-switch AR(2) series for pool-based detection training.
Each series has exactly one switch at a random time τ ∈ [T/3, 2T/3].

Output NPZ keys:
    series       : (N, T) float32 — per-row standardised AR series
    switch_times : (N,)   int32   — switch time index in [0, T) (after burn-in)

Usage (smoke test):
    python generate_switch_pool.py --n_series 1000 --out switch_pool_test.npz

Usage (full pool):
    python generate_switch_pool.py --n_series 200000 --out switch_pool.npz
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from data.msar_sampler import _sample_ar_coeffs_batch, _scale_noise, _std_batch


def _generate_single_switch_ar_batch(
    rng: np.random.Generator,
    B: int,
    series_len: int,
    burn_in: int,
    ar_order: int,
    ar_coeff_scale: float,
    sigma_lo: float,
    sigma_hi: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate B single-switch AR(ar_order) series (K=2 regimes).
    Switch time is uniform in [total//3, 2*total//3).

    Returns
    -------
    y_std        : (B, series_len) float32 — per-row standardised
    switch_times : (B,) int32 — switch index in usable series [0, series_len)
    """
    total = series_len + burn_in
    k = 2

    # AR coefficients: (B, k=2, p)
    ar = _sample_ar_coeffs_batch(rng, B, k, ar_order, ar_coeff_scale)
    # Per-regime noise level: (B, k=2)
    sigmas = rng.uniform(sigma_lo, sigma_hi, size=(B, k))

    # Switch times within the full (burn_in-included) timeline
    tau_full = rng.integers(total // 3, 2 * total // 3, size=B)

    # Build states array: 0 before switch, 1 from switch onward
    states = np.zeros((B, total), dtype=int)
    for b in range(B):
        states[b, tau_full[b]:] = 1

    # Generate AR dynamics with regime-switching coefficients
    eps = _scale_noise(rng, B, total, sigmas, states)
    bi = np.arange(B)
    y = np.zeros((B, total), dtype=np.float32)
    p = ar_order
    for t in range(p, total):
        ar_t = ar[bi, states[:, t], :]      # (B, p) — regime-appropriate coefficients
        y_lags = y[:, t - p:t][:, ::-1]    # (B, p) — most recent lag first
        y[:, t] = (ar_t * y_lags).sum(1) + eps[:, t]

    # Discard burn-in; compute switch times in usable series
    y_usable = y[:, burn_in:]
    switch_times = (tau_full - burn_in).astype(np.int32)
    switch_times = np.clip(switch_times, 0, series_len - 1)

    y_std = _std_batch(y_usable).astype(np.float32)
    return y_std, switch_times


def generate_pool(
    n_series: int,
    series_len: int,
    burn_in: int,
    ar_order: int,
    ar_coeff_scale: float,
    sigma_lo: float,
    sigma_hi: float,
    seed: int,
    out_path: str,
    chunk_size: int = 512,
) -> None:
    rng = np.random.default_rng(seed)
    usable_len = series_len - burn_in  # wait — series_len already IS the usable length

    # Pre-allocate output arrays
    pool_series = np.empty((n_series, series_len), dtype=np.float32)
    pool_switch = np.empty(n_series, dtype=np.int32)

    n_generated = 0
    t0 = time.time()
    print(f"Generating {n_series:,} single-switch AR({ar_order}) series  "
          f"(series_len={series_len}, burn_in={burn_in})")
    print(f"Output: {out_path}\n")

    while n_generated < n_series:
        this_chunk = min(chunk_size, n_series - n_generated)

        while True:
            y, tau = _generate_single_switch_ar_batch(
                rng, this_chunk, series_len, burn_in,
                ar_order, ar_coeff_scale, sigma_lo, sigma_hi,
            )
            valid = (
                np.isfinite(y).all(axis=1) &
                (y.std(axis=1) > 1e-6)
            )
            if valid.sum() >= this_chunk:
                y = y[valid][:this_chunk]
                tau = tau[valid][:this_chunk]
                break

        pool_series[n_generated: n_generated + this_chunk] = y
        pool_switch[n_generated: n_generated + this_chunk] = tau
        n_generated += this_chunk

        elapsed = time.time() - t0
        rate = n_generated / elapsed
        eta = (n_series - n_generated) / rate if rate > 0 else 0
        print(f"\r  {n_generated:>8,}/{n_series:,}  "
              f"({100*n_generated/n_series:.1f}%)  "
              f"{rate:.0f} series/s  ETA {eta:.0f}s",
              end="", flush=True)

    elapsed = time.time() - t0
    print(f"\n\nDone in {elapsed:.1f}s  ({n_series/elapsed:.0f} series/s)")

    print(f"Saving to {out_path} ...")
    np.savez_compressed(
        out_path,
        series=pool_series,
        switch_times=pool_switch,
        n_series=np.array(n_series),
        series_len=np.array(series_len),
        burn_in=np.array(burn_in),
        ar_order=np.array(ar_order),
        ar_coeff_scale=np.array(ar_coeff_scale),
        seed=np.array(seed),
    )
    size_mb = Path(out_path).stat().st_size / 1e6
    print(f"Saved. File size: {size_mb:.1f} MB")
    print(f"Pool shape: series={pool_series.shape}  switch_times={pool_switch.shape}")
    print(f"Switch time range: min={pool_switch.min()}  max={pool_switch.max()}  "
          f"mean={pool_switch.mean():.1f}")


def main():
    ap = argparse.ArgumentParser(
        description="Generate single-switch AR pool for detection training."
    )
    ap.add_argument("--n_series",       type=int,   default=200_000)
    ap.add_argument("--series_len",     type=int,   default=512,
                    help="Usable series length after burn-in")
    ap.add_argument("--burn_in",        type=int,   default=100,
                    help="Burn-in steps discarded during simulation")
    ap.add_argument("--ar_order",       type=int,   default=2,
                    help="AR order (default 2 to match eval datasets)")
    ap.add_argument("--ar_coeff_scale", type=float, default=1.2,
                    help="AR coefficient scale (matches generate_pool.py default)")
    ap.add_argument("--sigma_lo",       type=float, default=0.15)
    ap.add_argument("--sigma_hi",       type=float, default=0.70)
    ap.add_argument("--seed",           type=int,   default=42)
    ap.add_argument("--out",            type=str,   default="switch_pool.npz")
    ap.add_argument("--chunk_size",     type=int,   default=512)
    args = ap.parse_args()

    generate_pool(
        n_series=args.n_series,
        series_len=args.series_len,
        burn_in=args.burn_in,
        ar_order=args.ar_order,
        ar_coeff_scale=args.ar_coeff_scale,
        sigma_lo=args.sigma_lo,
        sigma_hi=args.sigma_hi,
        seed=args.seed,
        out_path=args.out,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()

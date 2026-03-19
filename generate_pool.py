# generate_pool.py
"""
Pre-generate a pool of synthetic MSAR series and save to disk.
Run this on the LOGIN NODE (no GPU needed) before requesting a GPU allocation.
Training then loads series from the pool instead of generating on the fly,
removing all CPU generation overhead from the GPU training loop.

Usage:
    python generate_pool.py                        # defaults: 500k series, series_len=512
    python generate_pool.py --n_series 200000      # smaller pool, faster to generate
    python generate_pool.py --out pool_large.npz   # custom output path

Typical time: ~10-20 minutes for 500k series on a login node CPU.
Output file size: ~500k * 412 * 4 bytes ≈ ~800 MB for default settings.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from data.msar_sampler import MSARBatchSampler, MSARSamplerConfig


def generate_pool(
    n_series: int,
    series_len: int,
    burn_in: int,
    seed: int,
    out_path: str,
    chunk_size: int = 512,   # generate this many series at a time
) -> None:
    """
    Generate n_series synthetic series and save to a single .npz file.

    The pool is a (n_series, usable_len) float32 array where
    usable_len = series_len - burn_in (burn-in already discarded by sampler).

    Args:
        n_series:   total number of series to generate
        series_len: length of each series before burn-in discard
        burn_in:    burn-in steps discarded inside each simulator
        seed:       RNG seed for reproducibility
        out_path:   path to save the .npz file
        chunk_size: number of series to generate per batch (controls memory)
    """
    cfg = MSARSamplerConfig(
        series_len=series_len,
        burn_in=burn_in,
        k_regimes=2,
        ar_coeff_scale=0.6,
        ma_coeff_scale=0.4,
        sar_coeff_scale=0.35,
        sigma_lo=0.15,
        sigma_hi=0.70,
        persistence_lo=0.85,
        persistence_hi=0.98,
        mix_ar=0.22,
        mix_ar_near_unit=0.05,
        mix_ar_no_switch=0.05,
        mix_arma=0.13,
        mix_arima1=0.13,
        mix_arima2=0.07,
        mix_seasonal=0.12,
        mix_exog_const=0.08,
        mix_exog_sine=0.08,
        mix_exog_seasonal=0.07,
    )
    sampler = MSARBatchSampler(cfg, seed=seed)

    usable_len = series_len - burn_in  # each simulator discards burn_in internally
    pool = np.empty((n_series, usable_len), dtype=np.float32)

    n_generated = 0
    t0 = time.time()

    print(f"Generating {n_series:,} series  (series_len={series_len}, burn_in={burn_in})")
    print(f"Each series has {usable_len} usable timesteps after burn-in.")
    print(f"Output: {out_path}")
    print()

    while n_generated < n_series:
        this_chunk = min(chunk_size, n_series - n_generated)

        # Generate chunk_size series at once using vectorized batch simulators.
        # We call _simulate_batch directly since we want raw series not windows.
        while True:
            family = sampler.rng.choice(sampler._N_FAMILIES, p=sampler._mix_weights)
            candidate = sampler._simulate_batch(family, this_chunk + 8)
            valid = (
                np.isfinite(candidate).all(axis=1) &
                (candidate.std(axis=1) > 1e-6)
            )
            if valid.sum() >= this_chunk:
                chunk = candidate[valid][:this_chunk]
                break

        # Trim or pad to exactly usable_len (seasonal simulators add extra steps)
        if chunk.shape[1] > usable_len:
            chunk = chunk[:, :usable_len]
        elif chunk.shape[1] < usable_len:
            # pad with last value (rare edge case for very short seasonal series)
            pad = np.repeat(chunk[:, -1:], usable_len - chunk.shape[1], axis=1)
            chunk = np.concatenate([chunk, pad], axis=1)

        pool[n_generated: n_generated + this_chunk] = chunk.astype(np.float32)
        n_generated += this_chunk

        elapsed = time.time() - t0
        rate = n_generated / elapsed
        eta = (n_series - n_generated) / rate if rate > 0 else 0
        pct = 100 * n_generated / n_series
        print(
            f"\r  {n_generated:>7,}/{n_series:,}  ({pct:.1f}%)  "
            f"{rate:.0f} series/s  ETA {eta:.0f}s",
            end="", flush=True,
        )

    elapsed = time.time() - t0
    print(f"\n\nGeneration complete in {elapsed:.1f}s  ({n_series/elapsed:.0f} series/s)")

    # Save pool and metadata so the sampler can verify compatibility at load time
    print(f"Saving to {out_path} ...")
    np.savez_compressed(
        out_path,
        series=pool,
        n_series=np.array(n_series),
        series_len=np.array(series_len),
        burn_in=np.array(burn_in),
        seed=np.array(seed),
    )

    size_mb = Path(out_path).stat().st_size / 1e6
    print(f"Saved. File size: {size_mb:.1f} MB")
    print(f"Pool shape: {pool.shape}  dtype: {pool.dtype}")


def main():
    ap = argparse.ArgumentParser(
        description="Pre-generate synthetic MSAR series pool for training."
    )
    ap.add_argument(
        "--n_series", type=int, default=500_000,
        help="Number of series to generate (default: 500000)"
    )
    ap.add_argument(
        "--series_len", type=int, default=512,
        help="Length of each series before burn-in (default: 512)"
    )
    ap.add_argument(
        "--burn_in", type=int, default=100,
        help="Burn-in steps discarded per series (default: 100)"
    )
    ap.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for reproducibility (default: 42)"
    )
    ap.add_argument(
        "--out", type=str, default="series_pool.npz",
        help="Output file path (default: series_pool.npz)"
    )
    ap.add_argument(
        "--chunk_size", type=int, default=512,
        help="Series generated per internal batch (default: 512)"
    )
    args = ap.parse_args()

    generate_pool(
        n_series=args.n_series,
        series_len=args.series_len,
        burn_in=args.burn_in,
        seed=args.seed,
        out_path=args.out,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
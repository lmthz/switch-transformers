# plot_training_samples.py
"""
Plot example series from each sampler family to visually inspect
the training data distribution and check for explosions.

Usage:
    python plot_training_samples.py                    # saves to training_samples.png
    python plot_training_samples.py --out my_plot.png
    python plot_training_samples.py --n_per_family 3   # show 3 series per family
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from data.msar_sampler import MSARBatchSampler, MSARSamplerConfig


FAMILY_NAMES = [
    "AR switching\n(A1-A3, B1-B2, S1-S2, H1)",
    "AR near-unit-root\n(H2)",
    "AR no-switch\n(NS0, NS1, SW1)",
    "ARMA switching\n(C1)",
    "ARIMA d=1\n(D1, D3)",
    "ARIMA d=2\n(D2)",
    "Seasonal SARMA\n(F1)",
    "Exog constant\n(E1, E2)",
    "Exog sine\n(G1)",
    "Exog seasonal\n(F2)",
]

N_FAMILIES = 10


def plot_training_samples(
    n_per_family: int = 2,
    seed: int = 0,
    out_path: str = "training_samples.png",
) -> None:

    cfg = MSARSamplerConfig(
        series_len=512,
        burn_in=100,
        ar_coeff_scale=0.6,
    )
    sampler = MSARBatchSampler(cfg, seed=seed)

    fig = plt.figure(figsize=(18, 3.5 * N_FAMILIES))
    gs  = gridspec.GridSpec(N_FAMILIES, n_per_family, figure=fig,
                            hspace=0.6, wspace=0.3)

    for fam_idx in range(N_FAMILIES):
        # Generate a batch and take first n_per_family valid series
        attempts = 0
        collected = []
        while len(collected) < n_per_family and attempts < 20:
            candidate = sampler._simulate_batch(fam_idx, n_per_family + 8)
            valid = (
                np.isfinite(candidate).all(axis=1) &
                (candidate.std(axis=1) > 1e-6)
            )
            collected.extend(candidate[valid].tolist())
            attempts += 1

        series_list = [np.array(s) for s in collected[:n_per_family]]

        for col, series in enumerate(series_list):
            ax = fig.add_subplot(gs[fam_idx, col])
            T  = len(series)
            t  = np.arange(T)

            # Colour by rough regime: above/below rolling mean as proxy
            window = max(1, T // 20)
            rolling = np.convolve(series, np.ones(window) / window, mode="same")
            colour  = (series > rolling).astype(float)

            ax.scatter(t, series, c=colour, cmap="coolwarm", s=2, alpha=0.7, linewidths=0)
            ax.plot(t, series, color="gray", linewidth=0.4, alpha=0.5)

            # Annotate with stats
            mn, mx = series.min(), series.max()
            std    = series.std()
            ax.set_title(
                f"family {fam_idx}  series {col+1}\n"
                f"min={mn:.2f}  max={mx:.2f}  std={std:.2f}",
                fontsize=7, pad=3,
            )
            ax.set_xlabel("timestep", fontsize=6)
            ax.tick_params(labelsize=6)

            # Highlight if series looks explosive (std > 10 after standardisation)
            if std > 10 or not np.isfinite(series).all():
                ax.set_facecolor("#ffe0e0")
                ax.set_title(ax.get_title() + "\n⚠ POSSIBLE EXPLOSION", fontsize=7,
                             color="red", pad=3)

            # Add family label on leftmost column
            if col == 0:
                ax.set_ylabel(FAMILY_NAMES[fam_idx], fontsize=7, labelpad=4)

    fig.suptitle(
        f"Training data samples — {n_per_family} series per family\n"
        f"ar_coeff_scale=0.6  series_len=512  burn_in=100  seed={seed}",
        fontsize=10, y=1.001,
    )

    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"Saved to {out_path}")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_per_family", type=int, default=2)
    ap.add_argument("--seed",         type=int, default=0)
    ap.add_argument("--out",          type=str, default="training_samples.png")
    args = ap.parse_args()
    plot_training_samples(
        n_per_family=args.n_per_family,
        seed=args.seed,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
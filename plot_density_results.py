# plot_density_results.py
"""
Generate plots for all density experiments (A, B1, B2, B3, C, D, E).

Usage:
    python plot_density_results.py                    # plots all available
    python plot_density_results.py --experiments A D E
    python plot_density_results.py --out_dir figures/
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":       150,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "font.size":        10,
})

MSAR_COLOR  = "#e74c3c"
TR_COLOR    = "#2980b9"
NOISE_COLOR = "#27ae60"
RIDGE_COLOR = "#8e44ad"

FAMILY_COLORS = {
    "mean_ar":       "#2980b9",
    "mean_arima":    "#e67e22",
    "mean_seasonal": "#27ae60",
    "mean_exog":     "#8e44ad",
    "mean_all":      "#2c3e50",
}
FAMILY_LABELS = {
    "mean_ar":       "AR",
    "mean_arima":    "ARIMA",
    "mean_seasonal": "Seasonal",
    "mean_exog":     "Exogenous",
    "mean_all":      "All datasets",
}

PRESET_COLORS = {
    "ar_only":       "#e74c3c",
    "ar_arma":       "#e67e22",
    "ar_arma_arima": "#2980b9",
    "full":          "#27ae60",
}
PRESET_LABELS = {
    "ar_only":       "AR only",
    "ar_arma":       "AR + ARMA",
    "ar_arma_arima": "AR + ARMA + ARIMA",
    "full":          "Full mixture",
}


def load_msar(msar_csv: str) -> Optional[pd.DataFrame]:
    p = Path(msar_csv)
    if not p.exists():
        return None
    return pd.read_csv(p).set_index("dataset")


def save(fig, path: Path):
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ================================================================
# EXPERIMENT A — OOD RMSE vs log(M)
# ================================================================

def plot_experiment_a(df: pd.DataFrame, out_dir: Path):
    print("Plotting Experiment A...")
    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(df["M"], df["ood_rmse"],
            "o-", color=TR_COLOR, label="Transformer (OOD RMSE)", lw=2)
    ax.axhline(df["noise_floor"].iloc[0], color=NOISE_COLOR,
               linestyle="--", lw=1.5, label=f"Noise floor σ={df['noise_floor'].iloc[0]:.2f}")
    if "ridge_rmse" in df.columns:
        ax.plot(df["M"], df["ridge_rmse"],
                "s--", color=RIDGE_COLOR, label="Ridge regression (approx)", lw=1.5)

    ax.set_xscale("log", base=2)
    ax.set_xlabel("M — number of distinct training tasks (log scale)")
    ax.set_ylabel("OOD RMSE")
    ax.set_title("Exp A: Linear regression task diversity\n(Raventós et al. 2023 replication)")
    ax.legend()
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"2$^{{{int(np.log2(x))}}}$" if x > 0 else ""))
    save(fig, out_dir / "exp_a_ood_rmse_vs_M.png")


# ================================================================
# EXPERIMENT B1 — Family coverage grouped bars + heatmap
# ================================================================

def plot_experiment_b1(df: pd.DataFrame, out_dir: Path):
    print("Plotting Experiment B1...")

    presets = df["family_preset"].tolist()
    families = ["mean_ar", "mean_arima", "mean_seasonal", "mean_exog"]
    available = [f for f in families if f in df.columns]

    # ── Grouped bar chart ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(presets))
    w = 0.8 / len(available)
    for i, fam in enumerate(available):
        vals = df[fam].tolist()
        offset = (i - len(available)/2 + 0.5) * w
        bars = ax.bar(x + offset, vals, w * 0.9,
                      label=FAMILY_LABELS[fam],
                      color=FAMILY_COLORS[fam], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([PRESET_LABELS.get(p, p) for p in presets], rotation=15, ha="right")
    ax.set_ylabel("Mean val RMSE")
    ax.set_title("Exp B1: Process family coverage\nPerformance by family vs training preset")
    ax.legend(loc="upper right")
    save(fig, out_dir / "exp_b1_grouped_bars.png")

    # ── Per-dataset heatmap ──────────────────────────────────────
    dataset_cols = [c for c in df.columns if c.startswith("val_") or
                    (c not in ["family_preset","steps","mean_all","mean_ar",
                               "mean_arima","mean_seasonal","mean_exog",
                               "mean_gap_vs_msar"] and "_" in c and "rmse" not in c)]
    # Fall back to dataset-specific columns
    ds_cols = [c for c in df.columns if c.startswith("A") or c.startswith("B") or
               c.startswith("C") or c.startswith("D") or c.startswith("E") or
               c.startswith("F") or c.startswith("G") or c.startswith("H") or
               c.startswith("S") or c.startswith("N")]
    ds_cols = [c for c in ds_cols if c in df.columns]

    if ds_cols:
        heat = df.set_index("family_preset")[ds_cols].T
        fig, ax = plt.subplots(figsize=(7, 10))
        im = ax.imshow(heat.values, aspect="auto", cmap="RdYlGn_r",
                       vmin=0, vmax=1.2)
        ax.set_xticks(range(len(heat.columns)))
        ax.set_xticklabels([PRESET_LABELS.get(c, c) for c in heat.columns],
                           rotation=20, ha="right", fontsize=8)
        ax.set_yticks(range(len(heat.index)))
        ax.set_yticklabels(heat.index, fontsize=7)
        plt.colorbar(im, ax=ax, label="Val RMSE")
        ax.set_title("Exp B1: Per-dataset RMSE heatmap\n(green=low RMSE, red=high)")
        save(fig, out_dir / "exp_b1_heatmap.png")


# ================================================================
# EXPERIMENT B2 — H1 vs A1 by AR order
# ================================================================

def plot_experiment_b2(df: pd.DataFrame, out_dir: Path):
    print("Plotting Experiment B2...")
    fig, ax = plt.subplots(figsize=(7, 4.5))

    order_hi = df["ar_order_hi"].tolist()
    if "H1_ar10_coeffs" in df.columns:
        ax.plot(order_hi, df["H1_ar10_coeffs"], "o-", color="#e74c3c",
                label="H1 AR(10) — OOD", lw=2)
    if "A1_ar2_coeffs_easy" in df.columns:
        ax.plot(order_hi, df["A1_ar2_coeffs_easy"], "s--", color=TR_COLOR,
                label="A1 AR(2) — in-distribution", lw=2)
    if "mean_all" in df.columns:
        ax.plot(order_hi, df["mean_all"], "^:", color="#2c3e50",
                label="Mean all datasets", lw=1.5, alpha=0.7)

    ax.set_xlabel("Max AR order in training pool (ar_order_hi)")
    ax.set_ylabel("Val RMSE")
    ax.set_title("Exp B2: AR order coverage\nDoes restricting training order hurt AR(10)?")
    ax.legend()
    save(fig, out_dir / "exp_b2_order_coverage.png")


# ================================================================
# EXPERIMENT B3 — Coefficient magnitude
# ================================================================

def plot_experiment_b3(df: pd.DataFrame, out_dir: Path):
    print("Plotting Experiment B3...")
    fig, ax = plt.subplots(figsize=(7, 4.5))

    scales = df["ar_coeff_scale"].tolist()
    if "mean_all" in df.columns:
        ax.plot(scales, df["mean_all"], "o-", color="#2c3e50",
                label="Mean all datasets", lw=2)
    if "mean_ar" in df.columns:
        ax.plot(scales, df["mean_ar"], "s-", color=TR_COLOR,
                label="AR datasets", lw=2)
    if "A1_ar2_coeffs_easy" in df.columns:
        ax.plot(scales, df["A1_ar2_coeffs_easy"], "^--", color="#e74c3c",
                label="A1 (has coeff 1.2)", lw=1.5)

    ax.axvline(0.6, color="gray", linestyle=":", lw=1, label="Default scale=0.6")
    ax.set_xlabel("ar_coeff_scale (training coefficient range)")
    ax.set_ylabel("Val RMSE")
    ax.set_title("Exp B3: AR coefficient magnitude\nEffect of restricting coefficient range")
    ax.legend()
    save(fig, out_dir / "exp_b3_coeff_magnitude.png")


# ================================================================
# EXPERIMENT C — Steps sweep
# ================================================================

def plot_experiment_c(df: pd.DataFrame, out_dir: Path,
                      msar_df: Optional[pd.DataFrame] = None):
    print("Plotting Experiment C...")
    fig, ax = plt.subplots(figsize=(8, 5))

    steps = df["steps"].tolist()
    families = ["mean_all", "mean_ar", "mean_arima", "mean_seasonal"]
    for fam in [f for f in families if f in df.columns]:
        ax.plot(steps, df[fam], "o-", color=FAMILY_COLORS[fam],
                label=FAMILY_LABELS[fam], lw=2)

    # MSAR reference lines
    if msar_df is not None:
        ar_ds = ["A1_ar2_coeffs_easy","A2_ar2_coeffs_hard","A3_ar2_coeffs_plus_var",
                 "B1_ar2_variance","B2_ar2_variance_big"]
        arima_ds = ["D1_arima211","D2_arima221","D3_arima210"]
        seasonal_ds = ["F1_seasonal_sarimax","F2_seasonal_exog"]
        for ds_list, color, label in [
            (ar_ds,      FAMILY_COLORS["mean_ar"],       "MSAR AR"),
            (arima_ds,   FAMILY_COLORS["mean_arima"],    "MSAR ARIMA"),
            (seasonal_ds,FAMILY_COLORS["mean_seasonal"], "MSAR Seasonal"),
        ]:
            vals = [float(msar_df.loc[d,"msar_val_rmse"]) for d in ds_list
                    if d in msar_df.index and not np.isnan(float(msar_df.loc[d,"msar_val_rmse"]))]
            if vals:
                ax.axhline(np.mean(vals), color=color, linestyle="--",
                           lw=1, alpha=0.6, label=label)

    ax.set_xscale("log")
    ax.set_xlabel("Training steps (log scale)")
    ax.set_ylabel("Val RMSE")
    ax.set_title("Exp C: Training steps sweep\nMinimum data needed to learn different dynamics")
    ax.legend(fontsize=8, ncol=2)
    save(fig, out_dir / "exp_c_steps_sweep.png")


# ================================================================
# EXPERIMENT D — Task diversity sweep
# ================================================================

def plot_experiment_d(df: pd.DataFrame, out_dir: Path,
                      msar_df: Optional[pd.DataFrame] = None):
    print("Plotting Experiment D...")
    fig, ax = plt.subplots(figsize=(8, 5))

    M = df["M"].tolist()
    families = ["mean_all", "mean_ar", "mean_arima", "mean_seasonal"]
    for fam in [f for f in families if f in df.columns]:
        ax.plot(M, df[fam], "o-", color=FAMILY_COLORS[fam],
                label=FAMILY_LABELS[fam], lw=2)

    if msar_df is not None:
        msar_mean = msar_df["msar_val_rmse"].dropna().mean()
        ax.axhline(msar_mean, color=MSAR_COLOR, linestyle="--",
                   lw=1.5, label=f"MSAR mean ({msar_mean:.3f})")

    ax.set_xscale("log", base=2)
    ax.set_xlabel("M — pool size / number of distinct training series (log scale)")
    ax.set_ylabel("Val RMSE")
    ax.set_title("Exp D: Task diversity sweep\n(Raventós replication for switching time series)")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"2$^{{{int(np.log2(x))}}}$" if x > 1 else str(int(x))))
    save(fig, out_dir / "exp_d_task_diversity.png")


# ================================================================
# EXPERIMENT E — Task diversity × model class
# ================================================================

def plot_experiment_e(df: pd.DataFrame, out_dir: Path,
                      msar_df: Optional[pd.DataFrame] = None):
    print("Plotting Experiment E...")

    presets = df["family_preset"].unique().tolist()

    # ── Panel 1: mean_all for both presets ──────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    for preset in presets:
        sub = df[df["family_preset"] == preset].sort_values("M")
        ax.plot(sub["M"], sub["mean_all"], "o-",
                color=PRESET_COLORS.get(preset, "gray"),
                label=PRESET_LABELS.get(preset, preset), lw=2)

    if msar_df is not None:
        msar_mean = msar_df["msar_val_rmse"].dropna().mean()
        ax.axhline(msar_mean, color=MSAR_COLOR, linestyle="--",
                   lw=1.5, label=f"MSAR mean ({msar_mean:.3f})")

    ax.set_xscale("log", base=2)
    ax.set_xlabel("M — pool size (log scale)")
    ax.set_ylabel("Mean val RMSE (all datasets)")
    ax.set_title("Exp E: Task diversity × model class\nDoes more data overcome restricted training families?")
    ax.legend()
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"2$^{{{int(np.log2(x))}}}$" if x > 1 else str(int(x))))
    save(fig, out_dir / "exp_e_diversity_x_class_all.png")

    # ── Panel 2: mean_arima for both presets ─────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    for preset in presets:
        sub = df[df["family_preset"] == preset].sort_values("M")
        if "mean_arima" in sub.columns:
            ax.plot(sub["M"], sub["mean_arima"], "o-",
                    color=PRESET_COLORS.get(preset, "gray"),
                    label=PRESET_LABELS.get(preset, preset), lw=2)

    if msar_df is not None:
        arima_ds = ["D1_arima211","D2_arima221","D3_arima210"]
        vals = [float(msar_df.loc[d,"msar_val_rmse"]) for d in arima_ds
                if d in msar_df.index and not np.isnan(float(msar_df.loc[d,"msar_val_rmse"]))]
        if vals:
            ax.axhline(np.mean(vals), color=MSAR_COLOR, linestyle="--",
                       lw=1.5, label=f"MSAR ARIMA mean ({np.mean(vals):.3f})")

    ax.set_xscale("log", base=2)
    ax.set_xlabel("M — pool size (log scale)")
    ax.set_ylabel("Mean val RMSE (ARIMA datasets)")
    ax.set_title("Exp E: ARIMA performance vs task diversity\nAR-only vs full training mixture")
    ax.legend()
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"2$^{{{int(np.log2(x))}}}$" if x > 1 else str(int(x))))
    save(fig, out_dir / "exp_e_diversity_x_class_arima.png")


# ================================================================
# Main
# ================================================================

def main():
    ap = argparse.ArgumentParser(description="Plot density experiment results.")
    ap.add_argument("--experiments", nargs="+",
                    default=["A","B1","B2","B3","C","D","E"],
                    choices=["A","B1","B2","B3","C","D","E"])
    ap.add_argument("--results_dir", type=str, default=".",
                    help="Directory containing results_density_exp_*.csv files.")
    ap.add_argument("--msar_csv",    type=str, default="msar_results.csv")
    ap.add_argument("--out_dir",     type=str, default="figures")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    results_dir = Path(args.results_dir)

    msar_df = load_msar(args.msar_csv)
    if msar_df is None:
        print(f"[warning] {args.msar_csv} not found — MSAR reference lines will be omitted")

    exp_map = {
        "A":  ("results_density_exp_a.csv",  plot_experiment_a,  False),
        "B1": ("results_density_exp_b1.csv", plot_experiment_b1, False),
        "B2": ("results_density_exp_b2.csv", plot_experiment_b2, False),
        "B3": ("results_density_exp_b3.csv", plot_experiment_b3, False),
        "C":  ("results_density_exp_c.csv",  plot_experiment_c,  True),
        "D":  ("results_density_exp_d.csv",  plot_experiment_d,  True),
        "E":  ("results_density_exp_e.csv",  plot_experiment_e,  True),
    }

    for exp in args.experiments:
        fname, plot_fn, needs_msar = exp_map[exp]
        csv_path = results_dir / fname
        if not csv_path.exists():
            print(f"[skip] {csv_path} not found")
            continue
        df = pd.read_csv(csv_path)
        if needs_msar:
            plot_fn(df, out_dir, msar_df)
        else:
            plot_fn(df, out_dir)

    print(f"\nAll plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
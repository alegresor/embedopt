#!/usr/bin/env python3
"""
Step 12: Plot CC_mask and MolProbity vs. learning rate (Figure 2C from the EmbedOpt paper).

Side-by-side boxplots per learning rate (EmbedOpt vs. DPS) with individual data
points (jittered) and median curves overlaid.

Input:  benchmark_summary.csv (from step 10)
Output: learning_rate_sensitivity_5A.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent.resolve()

COLOR_EMBEDOPT = "#FF6347"
COLOR_DPS = "#2E8B57"


def format_lr(lr: float) -> str:
    if lr < 0.01:
        return f"{lr:.4f}"
    elif lr < 0.1:
        return f"{lr:.3f}"
    else:
        return f"{lr:.2f}"


def plot_metric(ax, df: pd.DataFrame, metric: str, lrs: list, ylabel: str):
    positions_emb = np.arange(len(lrs)) * 3.0
    positions_dps = positions_emb + 1.0
    rng = np.random.default_rng(42)

    for i, lr in enumerate(lrs):
        for pos, algo, color in [
            (positions_emb[i], "embedopt", COLOR_EMBEDOPT),
            (positions_dps[i], "dps", COLOR_DPS),
        ]:
            vals = df[(df["LR"] == lr) & (df["Algorithm"] == algo)][metric].dropna().values
            if len(vals) == 0:
                continue
            ax.boxplot(vals, positions=[pos], widths=0.6, patch_artist=True,
                            medianprops=dict(color="black", lw=2),
                            whiskerprops=dict(color=color),
                            capprops=dict(color=color),
                            flierprops=dict(marker=""),
                            boxprops=dict(facecolor=color, alpha=0.4))
            jitter = rng.uniform(-0.2, 0.2, size=len(vals))
            ax.scatter(pos + jitter, vals, c=color, alpha=0.15, s=12, zorder=3)

    # Median curves
    for algo, color, label in [("embedopt", COLOR_EMBEDOPT, "EmbedOpt"), ("dps", COLOR_DPS, "DPS")]:
        medians = []
        x_pos = positions_emb if algo == "embedopt" else positions_dps
        for lr in lrs:
            vals = df[(df["LR"] == lr) & (df["Algorithm"] == algo)][metric].dropna().values
            medians.append(np.median(vals) if len(vals) > 0 else np.nan)
        ax.plot(x_pos, medians, "-o", color=color, lw=2, ms=6, label=label, zorder=4)

    tick_pos = (positions_emb + positions_dps) / 2
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([format_lr(lr) for lr in lrs], rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=9)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=HERE / "benchmark_summary.csv",
                        help="Input CSV (default: benchmark_summary.csv)")
    parser.add_argument("--output", type=Path, default=HERE / "learning_rate_sensitivity_5A.png",
                        help="Output PNG (default: learning_rate_sensitivity_5A.png)")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df = df[(df["Resolution"] == 5.0) & df["Algorithm"].isin(["embedopt", "dps"])].copy()
    df = df.dropna(subset=["LR"])

    lrs = sorted(df["LR"].unique())
    print(f"Learning rates: {lrs}")
    print(f"Records: {len(df)}")

    _, (ax_cc, ax_mp) = plt.subplots(2, 1, figsize=(max(8, len(lrs) * 1.5), 9))

    plot_metric(ax_cc, df, "CC_mask", lrs, "CC_mask")
    ax_cc.set_title("Map correlation vs. learning rate (5 Å)")

    plot_metric(ax_mp, df, "MolProbity_Score", lrs, "MolProbity Score")
    ax_mp.set_xlabel("Learning rate / guided scale")
    ax_mp.set_title("MolProbity score vs. learning rate (5 Å)")

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    plt.close()
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()

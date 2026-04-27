#!/usr/bin/env python3
"""
Step 11: Plot task difficulty vs. final CC (Figure 2B from the EmbedOpt paper).

For each system, difficulty = 1 - mean(CC_mask of base/unconditional runs).
Best performance per method per seed = max CC_mask over all learning rates.

Two-panel figure:
  Top:    final CC_mask vs. difficulty, scatter + binned mean ± std
  Bottom: CC difference (EmbedOpt - DPS) vs. difficulty + zero line

Input:  benchmark_summary.csv (from step 10)
Output: difficulty_vs_performance_5A.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent.resolve()

COLOR_EMBEDOPT = "#FF6347"
COLOR_DPS = "#2E8B57"


def compute_binned_stats(x: np.ndarray, y: np.ndarray, n_bins: int = 6):
    bins = np.linspace(x.min(), x.max(), n_bins + 1)
    centers, means, stds = [], [], []
    for i in range(n_bins):
        mask = (x >= bins[i]) & (x < bins[i + 1])
        if mask.sum() >= 2:
            centers.append((bins[i] + bins[i + 1]) / 2)
            means.append(y[mask].mean())
            stds.append(y[mask].std())
    return np.array(centers), np.array(means), np.array(stds)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=HERE / "benchmark_summary.csv",
                        help="Input CSV (default: benchmark_summary.csv)")
    parser.add_argument("--output", type=Path, default=HERE / "difficulty_vs_performance_5A.png",
                        help="Output PNG (default: difficulty_vs_performance_5A.png)")
    parser.add_argument("--n-bins", type=int, default=6, help="Number of bins for trend line (default: 6)")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df = df[df["Resolution"] == 5.0].copy()

    # Difficulty: 1 - mean CC_mask of base runs per system
    base = df[df["Algorithm"] == "base"].groupby("System")["CC_mask"].mean()
    difficulty = (1 - base).rename("difficulty")

    systems = difficulty.index.tolist()
    print(f"Systems with base results: {len(systems)}")

    # Best CC per system/seed: max over all LRs
    records = []
    for method in ["embedopt", "dps"]:
        mdf = df[df["Algorithm"] == method]
        best = mdf.groupby(["System", "Seed"])["CC_mask"].max().reset_index()
        best["Algorithm"] = method
        records.append(best)
    best_df = pd.concat(records)
    best_df = best_df.merge(difficulty.reset_index(), on="System")
    best_df = best_df.dropna(subset=["CC_mask", "difficulty"])

    emb = best_df[best_df["Algorithm"] == "embedopt"].set_index(["System", "Seed"])
    dps = best_df[best_df["Algorithm"] == "dps"].set_index(["System", "Seed"])
    common = emb.index.intersection(dps.index)
    diff_df = pd.DataFrame({
        "difficulty": emb.loc[common, "difficulty"],
        "delta": emb.loc[common, "CC_mask"].values - dps.loc[common, "CC_mask"].values,
    })

    _, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(7, 8), sharex=True)

    for method, color, label in [
        ("embedopt", COLOR_EMBEDOPT, "EmbedOpt"),
        ("dps", COLOR_DPS, "DPS"),
    ]:
        sub = best_df[best_df["Algorithm"] == method]
        x, y = sub["difficulty"].values, sub["CC_mask"].values
        ax_top.scatter(x, y, c=color, alpha=0.3, s=20, label=f"{label} (per seed)")
        cx, cm, cs = compute_binned_stats(x, y, args.n_bins)
        ax_top.plot(cx, cm, "-o", color=color, lw=2, label=f"{label} (mean)")
        ax_top.fill_between(cx, cm - cs, cm + cs, color=color, alpha=0.15)

    ax_top.set_ylabel("Final CC_mask")
    ax_top.legend(fontsize=8, ncol=2)
    ax_top.set_title("Task difficulty vs. final map correlation (5 Å)")

    x_diff = diff_df["difficulty"].values
    y_diff = diff_df["delta"].values
    ax_bot.scatter(x_diff, y_diff, c="gray", alpha=0.3, s=20)
    cx, cm, cs = compute_binned_stats(x_diff, y_diff, args.n_bins)
    ax_bot.plot(cx, cm, "-o", color="black", lw=2)
    ax_bot.fill_between(cx, cm - cs, cm + cs, color="gray", alpha=0.2)
    ax_bot.axhline(0, color="black", lw=1, ls="--")
    ax_bot.set_xlabel("Difficulty (1 − baseline CC_mask)")
    ax_bot.set_ylabel("CC difference\n(EmbedOpt − DPS)")

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    plt.close()
    print(f"Saved: {args.output}")

    # Print statistics
    wins_emb = (diff_df["delta"] > 0).sum()
    wins_dps = (diff_df["delta"] < 0).sum()
    print(f"\nEmbedOpt wins: {wins_emb}/{len(diff_df)} ({100*wins_emb/len(diff_df):.1f}%)")
    print(f"DPS wins:      {wins_dps}/{len(diff_df)} ({100*wins_dps/len(diff_df):.1f}%)")
    print(f"Mean CC diff (EmbedOpt - DPS): {diff_df['delta'].mean():.4f}")


if __name__ == "__main__":
    main()

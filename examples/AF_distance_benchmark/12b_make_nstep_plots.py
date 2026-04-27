#!/usr/bin/env python3
"""
Visualize constraint satisfaction + MolProbity metrics across N_step values.

Mirror of 12_make_plots.py but x-axis is N_step (with paired lr s.t. N_step * lr = 20).

Optionally merges in the lr=0.1 baseline from the LR-sweep run (200 * 0.1 = 20)
so the 200-step point appears on the same plot.

Usage:
    python 12b_make_nstep_plots.py
    python 12b_make_nstep_plots.py --constraint-total 20 --output my_plot.png
"""

import argparse
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================================
# CONFIGURATION
# ============================================================================
DEFAULT_NSTEP_FOLDER = "analysis_nstep_k20_relaxed"
DEFAULT_BASELINE_FOLDER = "analysis_k20_relaxed"  # Pull lr=0.1 → label as nstep=200
DEFAULT_CONSTRAINT_TOTAL = 20
DEFAULT_BASELINE_LR = 0.1
DEFAULT_BASELINE_NSTEP = 200
DEFAULT_OUTPUT_PATH = f"./{DEFAULT_NSTEP_FOLDER}/{DEFAULT_CONSTRAINT_TOTAL}_constraints_nstep.png"
DEFAULT_CENTRAL_METRIC = "median"
# ============================================================================


def load_all_csv_files(folder_path: Path) -> pd.DataFrame:
    csv_files = glob.glob(str(folder_path / "*_analysis.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No *_analysis.csv files found in {folder_path}")
    print(f"Found {len(csv_files)} CSV files in {folder_path.name}")
    df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    print(f"  Loaded {len(df)} total records")
    return df


def load_and_merge_nstep_data(
    nstep_folder: Path,
    baseline_folder: Path,
    baseline_lr: float,
    baseline_nstep: int,
) -> pd.DataFrame:
    print("Loading N_step experiment data...")
    df_nstep = load_all_csv_files(nstep_folder)
    if "nstep" not in df_nstep.columns:
        raise ValueError(f"'nstep' column not found in {nstep_folder} CSVs")

    if baseline_folder.exists():
        print(f"\nLoading baseline data (lr={baseline_lr} → nstep={baseline_nstep})...")
        df_base = load_all_csv_files(baseline_folder)
        # lr stored as float in nstep CSVs but may be string in baseline; normalize
        df_base["lr"] = pd.to_numeric(df_base["lr"], errors="coerce")
        df_base = df_base[df_base["lr"] == baseline_lr].copy()
        df_base["nstep"] = baseline_nstep
        # Restrict baseline to systems present in the nstep run (smoke-test friendly)
        nstep_systems = set(df_nstep["system"].unique())
        df_base = df_base[df_base["system"].isin(nstep_systems)]
        print(f"  Baseline filtered to {len(df_base)} records (systems: {sorted(nstep_systems)})")
        df_combined = pd.concat([df_nstep, df_base], ignore_index=True)
    else:
        print(f"  Warning: baseline folder not found: {baseline_folder}")
        df_combined = df_nstep
    return df_combined


def plot_metrics_comparison(
    df: pd.DataFrame,
    constraint_total: int,
    output_path: str = None,
    central_metric: str = "median",
) -> None:
    metrics = [
        ("constraint_passed", "Constraints Passed"),
        ("total_distance_diff", "Total Distance Diff (Å)"),
        ("clashscore", "Clashscore"),
        ("rama_outliers", "Ramachandran Outliers (%)"),
        ("molprobity_score", "MolProbity Score"),
        ("rmse_pd", "RMSE Pairwise Distance (Å)"),
    ]

    df_dps = df[df["algorithm"] == "dps"].copy()
    df_ebd = df[df["algorithm"] == "embedopt"].copy()
    if len(df_dps) == 0 and len(df_ebd) == 0:
        print(f"Warning: no data for constraint_total={constraint_total}")
        return

    nstep_values = sorted(df["nstep"].dropna().unique())
    print(f"N_step values: {nstep_values}")
    print(f"DPS samples: {len(df_dps)}, EmbedOpt samples: {len(df_ebd)}")

    color_dps = "#2E8B57"
    color_ebd = "#FF6347"

    fig, axes = plt.subplots(1, 6, figsize=(29, 5))
    fig.suptitle(
        f"Metric Comparison Across N_step (Constraint Total = {constraint_total}, "
        f"N_step × lr = 20)",
        fontsize=16, fontweight="bold", y=1.02,
    )
    metric_func = np.mean if central_metric == "mean" else np.median

    for ax_idx, (metric_col, metric_label) in enumerate(metrics):
        ax = axes[ax_idx]
        if metric_col not in df.columns:
            ax.text(0.5, 0.5, f"{metric_col}\nNot Available", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(metric_label, fontweight="bold")
            continue

        centrals_dps, centrals_ebd = [], []
        for i, nstep in enumerate(nstep_values):
            data_dps = df_dps[df_dps["nstep"] == nstep][metric_col].dropna()
            data_ebd = df_ebd[df_ebd["nstep"] == nstep][metric_col].dropna()

            if metric_col in ["total_distance_diff", "clashscore"]:
                data_dps = data_dps + 0.1
                data_ebd = data_ebd + 0.1

            pos_dps = i - 0.15
            pos_ebd = i + 0.15

            if len(data_dps) > 0:
                ax.boxplot([data_dps], positions=[pos_dps], widths=0.25,
                           patch_artist=True, showfliers=False,
                           boxprops=dict(facecolor=color_dps, alpha=0.5, linewidth=1.5),
                           medianprops=dict(color=color_dps, linewidth=2),
                           whiskerprops=dict(color=color_dps, linewidth=1.5),
                           capprops=dict(color=color_dps, linewidth=1.5))
                x_jitter = pos_dps + np.random.normal(0, 0.03, len(data_dps))
                ax.scatter(x_jitter, data_dps, alpha=0.3, s=20, color=color_dps, zorder=1)
                centrals_dps.append(metric_func(data_dps))
            else:
                centrals_dps.append(np.nan)

            if len(data_ebd) > 0:
                ax.boxplot([data_ebd], positions=[pos_ebd], widths=0.25,
                           patch_artist=True, showfliers=False,
                           boxprops=dict(facecolor=color_ebd, alpha=0.5, linewidth=1.5),
                           medianprops=dict(color=color_ebd, linewidth=2),
                           whiskerprops=dict(color=color_ebd, linewidth=1.5),
                           capprops=dict(color=color_ebd, linewidth=1.5))
                x_jitter = pos_ebd + np.random.normal(0, 0.03, len(data_ebd))
                ax.scatter(x_jitter, data_ebd, alpha=0.3, s=20, color=color_ebd, zorder=1)
                centrals_ebd.append(metric_func(data_ebd))
            else:
                centrals_ebd.append(np.nan)

        positions = np.arange(len(nstep_values))
        valid_dps = ~np.isnan(centrals_dps)
        valid_ebd = ~np.isnan(centrals_ebd)
        if np.any(valid_dps):
            ax.plot(positions[valid_dps] - 0.15, np.array(centrals_dps)[valid_dps],
                    marker="o", markersize=10, linewidth=3, color=color_dps,
                    label=f"DPS ({central_metric})", zorder=10,
                    markeredgecolor="white", markeredgewidth=2)
        if np.any(valid_ebd):
            ax.plot(positions[valid_ebd] + 0.15, np.array(centrals_ebd)[valid_ebd],
                    marker="s", markersize=10, linewidth=3, color=color_ebd,
                    label=f"EmbedOpt ({central_metric})", zorder=10,
                    markeredgecolor="white", markeredgewidth=2)

        ax.set_xticks(positions)
        ax.set_xticklabels([f"{int(n)}" for n in nstep_values], rotation=45, ha="right", fontsize=10)
        ax.set_xlabel("N_step", fontsize=11, fontweight="bold")

        y_label = f"{metric_label} + 0.1" if metric_col in ["total_distance_diff", "clashscore"] else metric_label
        ax.set_ylabel(y_label, fontsize=11, fontweight="bold")
        ax.set_title(metric_label, fontsize=12, fontweight="bold")
        ax.grid(True, axis="y", ls="--", alpha=0.3)
        if metric_col in ["total_distance_diff", "clashscore"]:
            ax.set_yscale("log")
        if ax_idx == 0:
            ax.legend(fontsize=9, loc="best", framealpha=0.9)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nstep-folder", type=str, default=DEFAULT_NSTEP_FOLDER)
    parser.add_argument("--baseline-folder", type=str, default=DEFAULT_BASELINE_FOLDER)
    parser.add_argument("--constraint-total", type=int, default=DEFAULT_CONSTRAINT_TOTAL, choices=[1, 5, 20])
    parser.add_argument("--baseline-lr", type=float, default=DEFAULT_BASELINE_LR)
    parser.add_argument("--baseline-nstep", type=int, default=DEFAULT_BASELINE_NSTEP)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--central-metric", type=str, default=DEFAULT_CENTRAL_METRIC, choices=["mean", "median"])
    args = parser.parse_args()

    base_dir = Path(__file__).parent
    nstep_path = base_dir / args.nstep_folder
    baseline_path = base_dir / args.baseline_folder

    if not nstep_path.exists():
        print(f"Error: nstep folder not found: {nstep_path}")
        return

    print("=" * 80)
    print("N_step Robustness Visualization")
    print("=" * 80)
    print(f"Nstep folder:     {args.nstep_folder}")
    print(f"Baseline folder:  {args.baseline_folder}")
    print(f"Baseline lr:      {args.baseline_lr} → labelled nstep={args.baseline_nstep}")
    print(f"Constraint Total: {args.constraint_total}")
    print(f"Central Metric:   {args.central_metric}")
    print("=" * 80)
    print()

    df_all = load_and_merge_nstep_data(
        nstep_folder=nstep_path,
        baseline_folder=baseline_path,
        baseline_lr=args.baseline_lr,
        baseline_nstep=args.baseline_nstep,
    )
    df_filtered = df_all[df_all["constraint_total"] == args.constraint_total].copy()
    print(f"\nFiltered to {len(df_filtered)} records with constraint_total={args.constraint_total}")
    print(f"Unique systems:     {df_filtered['system'].nunique()}")
    print(f"Unique algorithms:  {df_filtered['algorithm'].unique()}")
    print(f"Unique N_step:      {sorted(df_filtered['nstep'].dropna().unique())}")
    print()

    if len(df_filtered) == 0:
        print(f"Error: no data for constraint_total={args.constraint_total}")
        return

    plot_metrics_comparison(
        df=df_filtered,
        constraint_total=args.constraint_total,
        output_path=args.output,
        central_metric=args.central_metric,
    )
    print("\nDone!")


if __name__ == "__main__":
    main()

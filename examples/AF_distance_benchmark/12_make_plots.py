#!/usr/bin/env python3
"""
Visualize constraint satisfaction and MolProbity metrics across systems.

This script creates multi-panel plots comparing the 'dps' and 'embedopt' samplers
across different learning rates for a specified constraint_total value.

Usage:
    1. Edit configuration variables below and run: python 12_make_plots.py
    2. Or use command-line arguments: python 12_make_plots.py --folder analysis_1_5_20 --constraint-total 5
    3. Save to file: python 12_make_plots.py --folder analysis_1_5_20 --constraint-total 1 --output my_plot.png
"""

import argparse
import glob
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================================
# CONFIGURATION - Edit these variables to customize the plot
# ============================================================================
DEFAULT_FOLDER = "analysis_k20_relaxed"  # Folder containing CSV files
DEFAULT_CONSTRAINT_TOTAL = 20                 # Constraint total: 1, 5, or 20
DEFAULT_OUTPUT_PATH = f"./{DEFAULT_FOLDER}/{DEFAULT_CONSTRAINT_TOTAL}_constraints.png"                    # Output file path (None = display only)
DEFAULT_CENTRAL_METRIC = "median"              # Central tendency metric: "mean" or "median"
# ============================================================================


def load_all_csv_files(folder_path: Path) -> pd.DataFrame:
    """
    Load and concatenate all CSV files from the specified folder.
    
    Args:
        folder_path: Path to folder containing *_analysis.csv files
        
    Returns:
        Combined DataFrame with all systems
    """
    csv_files = glob.glob(str(folder_path / "*_analysis.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No *_analysis.csv files found in {folder_path}")
    
    print(f"Found {len(csv_files)} CSV files")
    
    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(combined_df)} total records from {len(csv_files)} systems")
    
    return combined_df


def plot_metrics_comparison(
    df: pd.DataFrame,
    constraint_total: int,
    output_path: str = None,
    central_metric: str = "mean"
) -> None:
    """
    Create a 1x5 plot comparing metrics across learning rates.
    
    Args:
        df: DataFrame with all data (already filtered by constraint_total)
        constraint_total: The constraint_total value being plotted
        output_path: Optional path to save the figure
        central_metric: Central tendency metric to use ("mean" or "median")
    """
    # Define metrics to plot
    metrics = [
        ('constraint_passed', 'Constraints Passed'),
        ('total_distance_diff', 'Total Distance Diff (Å)'),
        ('clashscore', 'Clashscore'),
        ('rama_outliers', 'Ramachandran Outliers (%)'),
        ('molprobity_score', 'MolProbity Score'),
        ('rmse_pd', 'RMSE Pairwise Distance (Å)')
    ]
    
    # Filter data for dps and embedopt samplers only
    df_dps = df[df['algorithm'] == 'dps'].copy()
    df_ebd = df[df['algorithm'] == 'embedopt'].copy()

    if len(df_dps) == 0 and len(df_ebd) == 0:
        print(f"Warning: No data found for constraint_total={constraint_total}")
        return
    
    # Get unique learning rates (sorted)
    learning_rates = sorted(df['lr'].unique())
    learning_rates = [lr for lr in learning_rates if lr > 0]  # Exclude base (lr=0)
    
    if len(learning_rates) == 0:
        print("Warning: No learning rate data found (excluding base)")
        return
    
    print(f"Learning rates: {learning_rates}")
    print(f"DPS samples: {len(df_dps)}, EmbedOpt samples: {len(df_ebd)}")

    # Colors
    color_dps = '#2E8B57'  # SeaGreen (DPS)
    color_ebd = '#FF6347'  # Tomato (EmbedOpt)
    
    # Create figure
    fig, axes = plt.subplots(1, 6, figsize=(29, 5))
    fig.suptitle(f'Metric Comparison (Constraint Total = {constraint_total})', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    # Plot each metric
    for ax_idx, (metric_col, metric_label) in enumerate(metrics):
        ax = axes[ax_idx]
        
        # Check if metric exists in dataframe
        if metric_col not in df.columns:
            ax.text(0.5, 0.5, f'{metric_col}\nNot Available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(metric_label, fontweight='bold')
            continue
        
        # Collect central metric values for line plots
        centrals_dps = []
        centrals_ebd = []
        
        # Determine metric function
        metric_func = np.mean if central_metric == "mean" else np.median
        metric_label_text = central_metric.capitalize()
        
        # Plot for each learning rate
        for i, lr in enumerate(learning_rates):
            # Get data for this learning rate
            data_dps = df_dps[df_dps['lr'] == lr][metric_col].dropna()
            data_ebd = df_ebd[df_ebd['lr'] == lr][metric_col].dropna()
            
            # Transform total_distance_diff and clashscore by adding 0.1 for better log scale visualization
            if metric_col in ['total_distance_diff', 'clashscore']:
                data_dps = data_dps + 0.1
                data_ebd = data_ebd + 0.1
            
            # Offset positions
            pos_dps = i - 0.15
            pos_ebd = i + 0.15
            
            # Plot dps
            if len(data_dps) > 0:
                # Box plot
                bp_dps = ax.boxplot([data_dps], positions=[pos_dps], widths=0.25,
                                      patch_artist=True, showfliers=False,
                                      boxprops=dict(facecolor=color_dps, alpha=0.5, linewidth=1.5),
                                      medianprops=dict(color=color_dps, linewidth=2),
                                      whiskerprops=dict(color=color_dps, linewidth=1.5),
                                      capprops=dict(color=color_dps, linewidth=1.5))
                
                # Scatter points (small, transparent)
                x_jitter = pos_dps + np.random.normal(0, 0.03, len(data_dps))
                ax.scatter(x_jitter, data_dps, alpha=0.3, s=20, color=color_dps, zorder=1)
                
                # Store central metric
                centrals_dps.append(metric_func(data_dps))
            else:
                centrals_dps.append(np.nan)
            
            # Plot embedopt
            if len(data_ebd) > 0:
                # Box plot
                bp_ebd = ax.boxplot([data_ebd], positions=[pos_ebd], widths=0.25,
                                   patch_artist=True, showfliers=False,
                                   boxprops=dict(facecolor=color_ebd, alpha=0.5, linewidth=1.5),
                                   medianprops=dict(color=color_ebd, linewidth=2),
                                   whiskerprops=dict(color=color_ebd, linewidth=1.5),
                                   capprops=dict(color=color_ebd, linewidth=1.5))
                
                # Scatter points (small, transparent)
                x_jitter = pos_ebd + np.random.normal(0, 0.03, len(data_ebd))
                ax.scatter(x_jitter, data_ebd, alpha=0.3, s=20, color=color_ebd, zorder=1)
                
                # Store central metric
                centrals_ebd.append(metric_func(data_ebd))
            else:
                centrals_ebd.append(np.nan)
        
        # Plot central metric lines (highlighted)
        central_positions = np.arange(len(learning_rates))
        
        # Filter out NaN values for line plotting
        valid_dps = ~np.isnan(centrals_dps)
        valid_ebd = ~np.isnan(centrals_ebd)
        
        if np.any(valid_dps):
            ax.plot(central_positions[valid_dps] - 0.15, 
                   np.array(centrals_dps)[valid_dps],
                   marker='o', markersize=10, linewidth=3, color=color_dps,
                   label=f'DPS ({metric_label_text})', zorder=10, 
                   markeredgecolor='white', markeredgewidth=2)
        
        if np.any(valid_ebd):
            ax.plot(central_positions[valid_ebd] + 0.15, 
                   np.array(centrals_ebd)[valid_ebd],
                   marker='s', markersize=10, linewidth=3, color=color_ebd,
                   label=f'EmbedOpt ({metric_label_text})', zorder=10,
                   markeredgecolor='white', markeredgewidth=2)
        
        # Formatting
        ax.set_xticks(central_positions)
        ax.set_xticklabels([f'{lr:.3f}' for lr in learning_rates], 
                          rotation=45, ha='right', fontsize=9)
        ax.set_xlabel('Learning Rate (lr)', fontsize=11, fontweight='bold')
        
        # Update y-axis label for transformed metrics
        if metric_col in ['total_distance_diff', 'clashscore']:
            y_label = f"{metric_label} + 0.1"
        else:
            y_label = metric_label
        ax.set_ylabel(y_label, fontsize=11, fontweight='bold')
        ax.set_title(metric_label, fontsize=12, fontweight='bold')
        ax.grid(True, axis='y', ls='--', alpha=0.3)
        
        # Set log scale for total_distance_diff and clashscore
        if metric_col in ['total_distance_diff', 'clashscore']:
            ax.set_yscale('log')
        
        # Only show legend on first panel
        if ax_idx == 0:
            ax.legend(fontsize=9, loc='best', framealpha=0.9)
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()


def main():
    """Main function to create visualization plots."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--folder',
        type=str,
        default=DEFAULT_FOLDER,
        help=f'Folder name containing CSV files (default: {DEFAULT_FOLDER})'
    )
    parser.add_argument(
        '--constraint-total',
        type=int,
        default=DEFAULT_CONSTRAINT_TOTAL,
        choices=[1, 5, 20],
        help=f'Constraint total value to plot (default: {DEFAULT_CONSTRAINT_TOTAL})'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help='Output file path for the plot (default: display only)'
    )
    parser.add_argument(
        '--central-metric',
        type=str,
        default=DEFAULT_CENTRAL_METRIC,
        choices=['mean', 'median'],
        help=f'Central tendency metric to highlight (default: {DEFAULT_CENTRAL_METRIC})'
    )
    
    args = parser.parse_args()
    
    # Get base directory
    base_dir = Path(__file__).parent
    folder_path = base_dir / args.folder
    
    if not folder_path.exists():
        print(f"Error: Folder not found: {folder_path}")
        return
    
    print("=" * 80)
    print("Constraint Satisfaction & MolProbity Visualization")
    print("=" * 80)
    print(f"Folder:           {args.folder}")
    print(f"Constraint Total: {args.constraint_total}")
    print(f"Central Metric:   {args.central_metric}")
    print("=" * 80)
    print()
    
    # Load all CSV files
    df_all = load_all_csv_files(folder_path)
    
    # Filter by constraint_total
    df_filtered = df_all[df_all['constraint_total'] == args.constraint_total].copy()
    
    print(f"\nFiltered to {len(df_filtered)} records with constraint_total={args.constraint_total}")
    print(f"Unique systems: {df_filtered['system'].nunique()}")
    print(f"Unique algorithms: {df_filtered['algorithm'].unique()}")
    print(f"Unique learning rates: {sorted(df_filtered['lr'].unique())}")
    print()
    
    if len(df_filtered) == 0:
        print(f"Error: No data found for constraint_total={args.constraint_total}")
        return
    
    # Create plot
    plot_metrics_comparison(
        df=df_filtered,
        constraint_total=args.constraint_total,
        output_path=args.output,
        central_metric=args.central_metric
    )
    
    print("\nDone!")


if __name__ == "__main__":
    main()

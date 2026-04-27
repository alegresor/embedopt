#!/usr/bin/env python3
"""
Generate distance constraint dataframes for all systems.

For each system:
1. Load GT and base distance matrices
2. Find top K largest differences
3. Save results as CSV

Usage:
    python 07_make_constraint_dataframes.py [--k NUM_CONSTRAINTS]
"""

import numpy as np
import pandas as pd
from pathlib import Path
import argparse


def get_top_k_diff(diff, gt_matrix, proteinix_matrix, k=10):
    """
    Finds the top k largest values in the upper triangle of 'diff',
    ignoring NaNs, and returns corresponding values from A and B.

    Args:
        diff (np.ndarray): The symmetric difference matrix (NxN).
        gt_matrix (np.ndarray): Ground truth distance matrix (NxN).
        proteinix_matrix (np.ndarray): Base prediction matrix (NxN).
        k (int): Number of top elements to retrieve.

    Returns:
        pd.DataFrame: DataFrame containing Row, Col, Diff, Val_A, Val_B.
    """
    # 1. Create a mask for valid entries (Upper Triangle AND Not NaN)
    # k=1 excludes the main diagonal. Use k=0 if you want to include diagonal.
    mask = np.triu(np.ones_like(diff, dtype=bool), k=1) & ~np.isnan(diff)

    # 2. Replace invalid entries with -infinity so they sort to the bottom
    # .ravel() flattens the array to 1D
    clean_diff_flat = np.where(mask, diff, -np.inf).ravel()

    # 3. Handle edge case: if k is larger than total elements
    k = min(k, clean_diff_flat.size)

    # 4. Get indices of the top K values efficiently
    # argpartition is O(n), significantly faster than full sort O(n log n) for large matrices
    if k > 0:
        flat_indices = np.argpartition(clean_diff_flat, -k)[-k:]
    else:
        return pd.DataFrame()

    # 5. Sort the top K specifically (descending order)
    # argpartition does not guarantee order, so we sort the small subset
    top_values = clean_diff_flat[flat_indices]
    sorted_order = np.argsort(top_values)[::-1]
    final_indices = flat_indices[sorted_order]

    # 6. Convert flat indices back to (row, col) coordinates
    rows, cols = np.unravel_index(final_indices, diff.shape)

    # 7. Extract values and build DataFrame
    df = pd.DataFrame(
        {
            "Residue 1": rows + 1,
            "Residue 2": cols + 1,
            "PDB distance (A)": gt_matrix[rows, cols],
            "Protenix distance (A)": proteinix_matrix[rows, cols],
            "Diff (A)": diff[rows, cols],
        }
    )

    # 8. Final clean up: Remove any -inf placeholders
    # (This happens if actual valid entries < k)
    return df[df["Diff (A)"] > -np.inf].reset_index(drop=True)


def process_system(system_name, gt_dir, base_dir, output_dir, k=100):
    """
    Process a single system and save constraint dataframe.

    Args:
        system_name: Name of the system (e.g., "4EGL_A")
        gt_dir: Directory with ground truth distance matrices
        base_dir: Directory with base predictions
        output_dir: Directory to save CSV outputs
        k: Number of constraints to generate
    """
    print(f"Processing {system_name}...")

    # Load matrices
    gt_file = gt_dir / f"{system_name}.npy"
    base_file = base_dir / f"{system_name}.npy"

    if not gt_file.exists():
        print(f"  Warning: GT file not found: {gt_file}")
        return

    if not base_file.exists():
        print(f"  Warning: Base file not found: {base_file}")
        return

    gt_matrix = np.load(gt_file)
    proteinix_matrix = np.load(base_file)

    # Compute difference
    diff_matrix = np.abs(gt_matrix - proteinix_matrix)

    # Get top K differences
    df = get_top_k_diff(diff_matrix, gt_matrix, proteinix_matrix, k=k)

    if df.empty:
        print(f"  Warning: No valid constraints found for {system_name}")
        return

    # Save to CSV
    output_file = output_dir / f"{system_name}_constraint.csv"
    df.to_csv(output_file, index=False)

    print(f"  Saved {len(df)} constraints -> {output_file}")
    print(f"    Distance range: {df['PDB distance (A)'].min():.1f} - {df['PDB distance (A)'].max():.1f} Å")
    print(f"    Mean difference: {df['Diff (A)'].mean():.2f} Å")


def main():
    """Main function to process all systems."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--k", type=int, default=100, help="Number of distance constraints per system (default: 100)")
    args = parser.parse_args()

    # Set up paths
    script_dir = Path(__file__).parent
    gt_dir = script_dir / "pair_dist_gt"
    base_dir = script_dir / "pair_dist_base"
    output_dir = script_dir / "distance_constraint_df"

    # Create output directory
    output_dir.mkdir(exist_ok=True)

    # Find all systems
    systems = sorted([f.stem for f in gt_dir.glob("*.npy")])

    print("=" * 80)
    print("Distance Constraint DataFrame Generator")
    print("=" * 80)
    print(f"Number of constraints (K): {args.k}")
    print("Output directory: distance_constraint_df/")
    print(f"Found {len(systems)} systems to process")
    print("=" * 80)
    print()

    # Process each system
    for system_name in systems:
        try:
            process_system(system_name=system_name, gt_dir=gt_dir, base_dir=base_dir, output_dir=output_dir, k=args.k)
        except Exception as e:
            print(f"  Error processing {system_name}: {e}")
            import traceback

            traceback.print_exc()
            continue

    print("\n" + "=" * 80)
    print("Done! Saved constraint dataframes in distance_constraint_df/")
    print("=" * 80)


if __name__ == "__main__":
    main()

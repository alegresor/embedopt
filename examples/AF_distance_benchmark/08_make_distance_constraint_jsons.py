#!/usr/bin/env python3
"""
Script to generate distance constraint benchmark configurations.

For each system:
1. Load base and ground truth distance matrices
2. Find top K largest differences
3. Generate distance constraints based on PDB distances
4. Create multiple jobs (embedopt and dps samplers with various learning rates)
   sharing the same constraint set
5. Save to benchmark_cfgs/{system}_benchmark.json

Usage:
    python 08_make_distance_constraint_jsons.py [system_name] [--k NUM_CONSTRAINTS] [--lrs LR ...] [--output-folder FOLDER]

Examples:
    python 08_make_distance_constraint_jsons.py                                # Process all systems, K=20, default LR sweep
    python 08_make_distance_constraint_jsons.py 4EGL_A                         # Process single system
    python 08_make_distance_constraint_jsons.py --k 10                         # Use 10 constraints
    python 08_make_distance_constraint_jsons.py 4EGL_A --k 15                  # Single system, 15 constraints
    python 08_make_distance_constraint_jsons.py --lrs 0.05 0.1                 # Smaller 2-LR sweep instead of default 6
    python 08_make_distance_constraint_jsons.py --output-folder my_configs     # Custom output folder
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import argparse
import copy


# Configuration
DEFAULT_LEARNING_RATES = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
ALGORITHMS = ["embedopt", "dps"]
DEFAULT_K = 20  # Default number of distance constraints
DISTANCE_MARGIN = 2.0  # +/- margin for min/max distance
# Output folder defaults to benchmark_cfgs_k{k} (set in main() so it tracks args.k)


def get_top_k_diff(diff, gt_matrix, base_matrix, k=10) -> pd.DataFrame:
    """
    Finds the top k largest values in the upper triangle of 'diff',
    ignoring NaNs, and returns corresponding values from ground truth and base.

    Args:
        diff (np.ndarray): The symmetric difference matrix (NxN).
        gt_matrix (np.ndarray): Ground truth distance matrix (NxN).
        base_matrix (np.ndarray): Base prediction matrix (NxN).
        k (int): Number of top elements to retrieve.

    Returns:
        pd.DataFrame: DataFrame containing residue pairs and distances.
    """
    # Create a mask for valid entries (Upper Triangle AND Not NaN)
    mask = np.triu(np.ones_like(diff, dtype=bool), k=1) & ~np.isnan(diff)

    # Replace invalid entries with -infinity
    clean_diff_flat = np.where(mask, diff, -np.inf).ravel()

    # Handle edge case: if k is larger than total elements
    k = min(k, clean_diff_flat.size)

    # Get indices of the top K values
    if k > 0:
        flat_indices = np.argpartition(clean_diff_flat, -k)[-k:]
    else:
        return pd.DataFrame()

    # Sort the top K (descending order)
    top_values = clean_diff_flat[flat_indices]
    sorted_order = np.argsort(top_values)[::-1]
    final_indices = flat_indices[sorted_order]

    # Convert flat indices back to (row, col) coordinates
    rows, cols = np.unravel_index(final_indices, diff.shape)

    # Extract values and build DataFrame
    df = pd.DataFrame(
        {
            "Residue_1": rows + 1,  # 1-indexed
            "Residue_2": cols + 1,  # 1-indexed
            "PDB_distance": gt_matrix[rows, cols],
            "Protenix_distance": base_matrix[rows, cols],
            "Diff": diff[rows, cols],
        }
    )

    # Remove any -inf placeholders
    result = df[df["Diff"] > -np.inf].reset_index(drop=True)
    return result  # type: ignore[return-value]


def create_constraint_contacts(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Convert DataFrame of residue pairs to constraint contact format.

    Args:
        df: DataFrame with columns Residue_1, Residue_2, PDB_distance

    Returns:
        List of contact constraint dictionaries
    """
    contacts = []
    for _, row in df.iterrows():
        contact = {
            "entity1": 1,
            "copy1": 1,
            "position1": int(row["Residue_1"]),
            "entity2": 1,
            "copy2": 1,
            "position2": int(row["Residue_2"]),
            "max_distance": float(row["PDB_distance"] + DISTANCE_MARGIN),
            "min_distance": float(row["PDB_distance"] - DISTANCE_MARGIN),
        }
        contacts.append(contact)

    return contacts


def create_reward_config(contacts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Create reward configuration with distance constraints.

    Args:
        contacts: List of contact constraint dictionaries

    Returns:
        Reward configuration dictionary
    """
    return {"type": "constraint", "params": {"distance_atom_type": "center", "context": {"contact": contacts}}}


def create_job_config(
    base_config: Dict[str, Any],
    system_name: str,
    algorithm_type: str,
    algorithm_params: Dict[str, Any],
    reward_config: Dict[str, Any],
    k: int,
) -> Dict[str, Any]:
    """Create a single job configuration."""
    job = copy.deepcopy(base_config)

    # Set name based on algorithm and parameters
    if algorithm_type == "embedopt":
        lr = algorithm_params["reward_scale_s"]
        job["name"] = f"{system_name}_embedopt_{lr}_k{k}"
    elif algorithm_type == "dps":
        scale = algorithm_params["guided_scale"]
        job["name"] = f"{system_name}_dps_{scale}_k{k}"

    # Set reward
    job["reward"] = reward_config

    # Set algorithm
    job["algorithm"] = {"type": algorithm_type, "params": algorithm_params}

    return job


def generate_benchmark_config(
    system_name: str,
    k: int,
    base_dir: Path,
    gt_dir: Path,
    minimal_json_dir: Path,
    output_dir: Path,
    learning_rates: List[float],
) -> None:
    """
    Generate benchmark configuration for a single system.

    Args:
        system_name: System name (e.g., "4EGL_A")
        k: Number of distance constraints to generate
        base_dir: Directory with base predictions
        gt_dir: Directory with ground truth distances
        minimal_json_dir: Directory with minimal JSON configs
        output_dir: Output directory for benchmark configs
        learning_rates: LR sweep — generates one job per (algorithm × lr)
    """
    print(f"Processing {system_name}...")

    # Load distance matrices
    base_file = base_dir / f"{system_name}.npy"
    gt_file = gt_dir / f"{system_name}.npy"

    if not base_file.exists():
        print(f"  Warning: Base file not found: {base_file}")
        return

    if not gt_file.exists():
        print(f"  Warning: Ground truth file not found: {gt_file}")
        return

    base_matrix = np.load(base_file)
    gt_matrix = np.load(gt_file)

    # Compute difference
    diff_matrix = np.abs(gt_matrix - base_matrix)

    # Get top K differences
    df = get_top_k_diff(diff_matrix, gt_matrix, base_matrix, k=k)

    if df.empty:
        print(f"  Warning: No valid constraints found for {system_name}")
        return

    print(f"  Found {len(df)} constraints (requested {k})")

    # Create constraint contacts
    contacts = create_constraint_contacts(df)
    reward_config = create_reward_config(contacts)

    # Load minimal JSON config
    minimal_json_file = minimal_json_dir / f"{system_name}_minimal-update-msa.json"

    if not minimal_json_file.exists():
        print(f"  Warning: Minimal JSON not found: {minimal_json_file}")
        return

    with open(minimal_json_file, "r") as f:
        minimal_config = json.load(f)

    if not minimal_config or len(minimal_config) == 0:
        print(f"  Warning: Empty config in {minimal_json_file}")
        return

    base_config = minimal_config[0]

    # Generate all jobs (same reward, different algorithms/learning rates)
    jobs = []

    # embedopt jobs
    for lr in learning_rates:
        job = create_job_config(
            base_config=base_config,
            system_name=system_name,
            algorithm_type="embedopt",
            algorithm_params={"reward_scale_s": lr, "reward_scale_z": lr},
            reward_config=reward_config,
            k=k,
        )
        jobs.append(job)

    # dps jobs
    for scale in learning_rates:
        job = create_job_config(
            base_config=base_config,
            system_name=system_name,
            algorithm_type="dps",
            algorithm_params={"guided_scale": scale, "normalize_gradients": True},
            reward_config=reward_config,
            k=k,
        )
        jobs.append(job)

    # Write output
    output_file = output_dir / f"{system_name}_benchmark.json"
    with open(output_file, "w") as f:
        json.dump(jobs, f, indent=4)

    print(f"  Generated {len(jobs)} jobs -> {output_file}")
    print("  Constraint summary:")
    print(f"    Min distance range: {df['PDB_distance'].min():.1f} - {df['PDB_distance'].max():.1f} Å")
    print(f"    Mean difference: {df['Diff'].mean():.2f} Å")


def main():
    """Main function to process systems and generate benchmark configs."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("system", nargs="?", help="System name (e.g., 4EGL_A). If not provided, process all systems.")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help=f"Number of distance constraints per system (default: {DEFAULT_K})")
    parser.add_argument(
        "--lrs", type=float, nargs="+", default=DEFAULT_LEARNING_RATES,
        help=f"Learning rates (one job per algorithm × lr). Default: {DEFAULT_LEARNING_RATES}",
    )
    parser.add_argument(
        "--output-folder", type=str, default=None,
        help="Output folder name (default: benchmark_cfgs_k{k})"
    )
    args = parser.parse_args()
    if args.output_folder is None:
        args.output_folder = f"benchmark_cfgs_k{args.k}"

    # Set up paths
    script_dir = Path(__file__).parent
    base_dir = script_dir / "pair_dist_base"
    gt_dir = script_dir / "pair_dist_gt"
    minimal_json_dir = script_dir / "json_minimal"
    output_dir = script_dir / args.output_folder

    # Create output directory
    output_dir.mkdir(exist_ok=True)

    # Determine which systems to process
    if args.system:
        systems = [args.system]
    else:
        # Find all systems in base directory
        systems = sorted([f.stem for f in base_dir.glob("*.npy")])

    print("=" * 80)
    print("Distance Constraint Benchmark Configuration Generator")
    print("=" * 80)
    print(f"Output folder:         {args.output_folder}")
    print(f"Number of constraints (K): {args.k}")
    print(f"Learning rates:        {args.lrs}")
    print(f"Algorithms:            {ALGORITHMS}")
    print(f"Distance margin:       ±{DISTANCE_MARGIN} Å")
    print(f"Found {len(systems)} systems to process")
    print("=" * 80)
    print()

    # Process each system
    for system_name in systems:
        try:
            generate_benchmark_config(
                system_name=system_name,
                k=args.k,
                base_dir=base_dir,
                gt_dir=gt_dir,
                minimal_json_dir=minimal_json_dir,
                output_dir=output_dir,
                learning_rates=args.lrs,
            )
        except Exception as e:
            print(f"  Error processing {system_name}: {e}")
            import traceback

            traceback.print_exc()
            continue

    print("\n" + "=" * 80)
    print(f"Done! Generated benchmark configs in {args.output_folder}/")
    print(f"Each config contains {len(args.lrs) * len(ALGORITHMS)} jobs")
    print(f"  - {len(args.lrs)} embedopt jobs (learning rates: {args.lrs})")
    print(f"  - {len(args.lrs)} dps jobs (scales: {args.lrs})")
    print("=" * 80)


if __name__ == "__main__":
    main()

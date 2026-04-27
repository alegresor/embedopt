#!/usr/bin/env python3
"""
Check constraint satisfaction for distance constraint benchmark.

For each system:
1. Load constraint definitions and ground truth distance matrices
2. Process base predictions and compute constraint satisfaction
3. Process conditional predictions (K=1,5,20) with various algorithms/learning rates
4. Compute RMSE of pairwise distances vs ground truth
5. Save analysis results to CSV

Usage:
    python check_constriant_satisfication.py                    # Process all systems
    python check_constriant_satisfication.py 1IXC_A             # Process single system
    python check_constriant_satisfication.py --output-dir results  # Custom output directory
    python check_constriant_satisfication.py --pattern "trajectory/step_99_relaxed.pdb"  # Custom pattern
    python check_constriant_satisfication.py --molprobity molprobity_relaxed.out  # Custom MolProbity report

Examples:
    python check_constriant_satisfication.py
    python check_constriant_satisfication.py 4EGL_A
    python check_constriant_satisfication.py --k-values 1 5 10 20
    python check_constriant_satisfication.py --pattern "trajectory/step_99_relaxed.pdb"
    python check_constriant_satisfication.py --pattern "predictions/*.cif" --molprobity molprobity_relaxed.out
"""

import argparse
import glob
import re
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
from SFC_Torch import PDBParser


# Configuration
DEFAULT_K_VALUES = [20]
DISTANCE_TOLERANCE = 2.0  # Angstroms tolerance for constraint satisfaction
DEFAULT_OUTPUT_DIR = "analysis_k20_relaxed"
# Defaults match step 10's relaxed-mode defaults (trajectory subdir, reward-written
# *_relaxed.pdb file, and molprobity_relaxed.{out,log}). Override on the CLI for
# unrelaxed analysis: --pattern 'predictions/*sample_0.cif' --molprobity molprobity_unrelaxed.out
DEFAULT_PREDICTION_PATTERN = "trajectory/step_*_relaxed.pdb"
DEFAULT_MOLPROBITY_REPORT = "molprobity_relaxed.out"


def compute_pairwise_distance(x: np.ndarray) -> np.ndarray:
    """
    Compute pairwise distance between the atoms in the array.

    Args:
        x: np.array of shape (N, 3)

    Returns:
        pairwise_distance: np.array of shape (N, N)
    """
    pairwise_distance = np.linalg.norm(x[None, :, :] - x[:, None, :], axis=-1)
    return pairwise_distance


def compute_rmse_pairwise_distance(pred_dist: np.ndarray, gt_dist: np.ndarray) -> float:
    """
    Compute RMSE of pairwise distances, ignoring NaN values.

    Args:
        pred_dist: Predicted pairwise distance matrix
        gt_dist: Ground truth pairwise distance matrix

    Returns:
        RMSE value
    """
    N = len(gt_dist)
    rmse = np.sqrt(np.nansum((pred_dist - gt_dist) ** 2) / (N**2 - N))
    return rmse


def extract_molprobity_score(log_file: Path) -> Optional[float]:
    """Extract MolProbity score from molprobity log"""
    try:
        with open(log_file, "r") as f:
            for line in f:
                if line.strip().startswith("MolProbity score"):
                    match = re.search(r"=\s*([-0-9.]+)", line)
                    if match:
                        return float(match.group(1))
    except Exception as e:
        print(f"    Warning: Could not extract MolProbity score from {log_file}: {e}")
    return None


def extract_rama_outliers(log_file: Path) -> Optional[float]:
    """Extract Ramachandran outliers % from molprobity log"""
    try:
        with open(log_file, "r") as f:
            for line in f:
                if line.strip().startswith("Ramachandran outliers"):
                    match = re.search(r"=\s*([-0-9.]+)\s*%", line)
                    if match:
                        return float(match.group(1))
    except Exception as e:
        print(f"    Warning: Could not extract Ramachandran outliers from {log_file}: {e}")
    return None


def extract_clashscore(log_file: Path) -> Optional[float]:
    """Extract Clashscore from molprobity log"""
    try:
        with open(log_file, "r") as f:
            for line in f:
                if line.strip().startswith("Clashscore"):
                    match = re.search(r"=\s*([-0-9.]+)", line)
                    if match:
                        return float(match.group(1))
    except Exception as e:
        print(f"    Warning: Could not extract Clashscore from {log_file}: {e}")
    return None


def check_constraints(
    pair_dist: np.ndarray,
    df_constraint: pd.DataFrame,
    k: int,
    tolerance: float = DISTANCE_TOLERANCE,
) -> tuple[int, float]:
    """
    Check how many constraints are satisfied and compute total distance difference.

    Args:
        pair_dist: Pairwise distance matrix from prediction
        df_constraint: DataFrame with constraint definitions
        k: Number of constraints to check
        tolerance: Distance tolerance for satisfaction

    Returns:
        Tuple of (number of constraints passed, total distance difference)
    """
    constraint_passed = 0
    total_distance_diff = 0.0
    for i in range(min(k, len(df_constraint))):
        constraint = df_constraint.iloc[i]
        resid1 = int(constraint["Residue 1"]) - 1  # Convert to 0-indexed
        resid2 = int(constraint["Residue 2"]) - 1
        expected_distance = float(constraint["PDB distance (A)"])
        actual_distance = pair_dist[resid1, resid2]
        distance_diff = np.abs(actual_distance - expected_distance)
        if distance_diff <= tolerance:
            constraint_passed += 1
        else:
            total_distance_diff += distance_diff - tolerance
    return constraint_passed, total_distance_diff


def get_seed_dir_from_prediction(prediction_path: str) -> Path:
    """Extract seed directory path from prediction file path."""
    parts = Path(prediction_path).parts
    # Find the seed_* directory in the path
    for i, part in enumerate(parts):
        if part.startswith("seed_"):
            return Path(*parts[:i+1])
    raise ValueError(f"Could not find seed directory in path: {prediction_path}")


def parse_base_path(cif_path: str) -> Dict[str, str]:
    """Parse metadata from base prediction path.

    Path layout: base_output/{system}/{system}_minimal/seed_{N}/{pattern}
    where {pattern} may be e.g. trajectory/step_*_relaxed.pdb (1 subdir deep).
    parts[-3] is the seed_{N} dir.
    """
    parts = cif_path.split("/")
    seed = parts[-3].split("_")[1]
    return {
        "algorithm": "base",
        "lr": "0",
        "seed": seed,
    }


def parse_conditional_path(cif_path: str) -> Dict[str, str]:
    """Parse metadata from steered prediction path.

    Two job-name formats are supported:
      - LR sweep (08/09):   {system}_{algo}_{lr}_k{k}             → fields: algo, lr, seed
      - N_step sweep (08b/09b): {system}_{algo}_N{nstep}_lr{lr}_k{k} → fields: algo, lr, nstep, seed

    {system} is `<PDB>_<chain>`, so config_parts[0:2] is the system, [2] is algo.
    parts[-4] is the {job_name} dir; parts[-3] is the seed dir.
    """
    parts = cif_path.split("/")
    config_name = parts[-4]
    config_parts = config_name.split("_")
    algorithm = config_parts[2]
    seed = parts[-3].split("_")[1]
    if config_parts[3].startswith("N") and len(config_parts) >= 6:
        # N_step sweep layout
        nstep = config_parts[3][1:]      # strip "N"
        lr = config_parts[4][2:]          # strip "lr"
        return {"algorithm": algorithm, "lr": lr, "seed": seed, "nstep": nstep}
    return {"algorithm": algorithm, "lr": config_parts[3], "seed": seed, "nstep": None}


def process_system(
    system_name: str,
    base_dir: Path,
    k_values: List[int],
    output_dir: Path,
    prediction_pattern: str = DEFAULT_PREDICTION_PATTERN,
    molprobity_report: str = DEFAULT_MOLPROBITY_REPORT,
    steering_dir_template: Optional[str] = None,
) -> None:
    """
    Process constraint satisfaction analysis for a single system.

    Args:
        system_name: System name (e.g., "1IXC_A")
        base_dir: Base directory containing all data
        k_values: List of K values to process
        output_dir: Output directory for results
        prediction_pattern: File pattern for predictions (default: "predictions/*.cif")
        molprobity_report: Filename for MolProbity report (default: "molprobity.out")
    """
    print(f"Processing {system_name}...")

    # Load constraint definitions
    constraint_file = base_dir / f"distance_constraint_df/{system_name}_constraint.csv"
    if not constraint_file.exists():
        print(f"  Warning: Constraint file not found: {constraint_file}")
        return

    df_constraint = pd.read_csv(constraint_file)

    # Load ground truth distance matrix
    gt_file = base_dir / f"pair_dist_gt/{system_name}.npy"
    if not gt_file.exists():
        print(f"  Warning: Ground truth file not found: {gt_file}")
        return

    gt_distance_matrix = np.load(gt_file)

    results = []

    # Process base predictions
    base_pattern = str(base_dir / f"base_output/{system_name}/{system_name}_*/seed_*/{prediction_pattern}")
    base_cifs = glob.glob(base_pattern)

    if base_cifs:
        print(f"  Found {len(base_cifs)} base predictions")
        for cif_path in base_cifs:
            try:
                metadata = parse_base_path(cif_path)
                pdb_ca = PDBParser(cif_path).selection("CA[*]")
                pair_dist = compute_pairwise_distance(pdb_ca.atom_pos)
                rmse_pd = compute_rmse_pairwise_distance(pair_dist, gt_distance_matrix)

                # Check constraints for max K (20 for base baseline)
                max_k = max(k_values)
                constraint_passed, total_distance_diff = check_constraints(pair_dist, df_constraint, max_k)

                # Extract MolProbity metrics if report exists
                seed_dir = get_seed_dir_from_prediction(cif_path)
                molprobity_file = seed_dir / molprobity_report
                
                clashscore = None
                rama_outliers = None
                molprobity_score = None
                
                if molprobity_file.exists():
                    clashscore = extract_clashscore(molprobity_file)
                    rama_outliers = extract_rama_outliers(molprobity_file)
                    molprobity_score = extract_molprobity_score(molprobity_file)

                results.append({
                    "algorithm": metadata["algorithm"],
                    "constraint_total": max_k,
                    "lr": metadata["lr"],
                    "seed": metadata["seed"],
                    "rmse_pd": rmse_pd,
                    "constraint_passed": constraint_passed,
                    "total_distance_diff": total_distance_diff,
                    "clashscore": clashscore,
                    "rama_outliers": rama_outliers,
                    "molprobity_score": molprobity_score,
                })
            except Exception as e:
                print(f"    Error processing {cif_path}: {e}")
                continue
    else:
        print("  No base predictions found")

    # Process conditional predictions for each K value
    for k in k_values:
        if steering_dir_template is not None:
            output_dir_k = base_dir / steering_dir_template.format(k=k)
        else:
            output_dir_k = base_dir / f"steering_output_k{k}"
        conditional_pattern = str(output_dir_k / f"{system_name}/{system_name}_*/seed_*/{prediction_pattern}")
        prediction_cifs = glob.glob(conditional_pattern)

        if prediction_cifs:
            print(f"  Found {len(prediction_cifs)} predictions for K={k}")
            for cif_path in prediction_cifs:
                try:
                    metadata = parse_conditional_path(cif_path)
                    pdb_ca = PDBParser(cif_path).selection("CA[*]")
                    pair_dist = compute_pairwise_distance(pdb_ca.atom_pos)
                    rmse_pd = compute_rmse_pairwise_distance(pair_dist, gt_distance_matrix)
                    constraint_passed, total_distance_diff = check_constraints(pair_dist, df_constraint, k)

                    # Extract MolProbity metrics if report exists
                    seed_dir = get_seed_dir_from_prediction(cif_path)
                    molprobity_file = seed_dir / molprobity_report

                    clashscore = None
                    rama_outliers = None
                    molprobity_score = None

                    if molprobity_file.exists():
                        clashscore = extract_clashscore(molprobity_file)
                        rama_outliers = extract_rama_outliers(molprobity_file)
                        molprobity_score = extract_molprobity_score(molprobity_file)

                    row = {
                        "algorithm": metadata["algorithm"],
                        "constraint_total": k,
                        "lr": metadata["lr"],
                        "seed": metadata["seed"],
                        "rmse_pd": rmse_pd,
                        "constraint_passed": constraint_passed,
                        "total_distance_diff": total_distance_diff,
                        "clashscore": clashscore,
                        "rama_outliers": rama_outliers,
                        "molprobity_score": molprobity_score,
                    }
                    if metadata.get("nstep") is not None:
                        row["nstep"] = int(metadata["nstep"])
                    results.append(row)
                except Exception as e:
                    print(f"    Error processing {cif_path}: {e}")
                    continue
        else:
            print(f"  No predictions found for K={k}")

    if not results:
        print(f"  Warning: No results for {system_name}")
        return

    # Create DataFrame and save
    df_results = pd.DataFrame(results)
    df_results["system"] = system_name
    df_results = df_results.sort_values(
        by=["algorithm", "constraint_total", "lr"],
        ascending=[False, True, False],
    )

    # Reorder columns
    column_order = [
        "system", "algorithm", "constraint_total", "lr", "seed",
        "rmse_pd", "constraint_passed", "total_distance_diff",
        "clashscore", "rama_outliers", "molprobity_score"
    ]
    if "nstep" in df_results.columns:
        column_order.insert(column_order.index("seed"), "nstep")
    df_results = df_results[column_order]

    output_file = output_dir / f"{system_name}_analysis.csv"
    df_results.to_csv(output_file, index=False)
    print(f"  Saved {len(df_results)} results -> {output_file}")

    # Print summary
    print("  Summary:")
    for algo in df_results["algorithm"].unique():
        algo_df = df_results[df_results["algorithm"] == algo]
        mean_rmse = algo_df["rmse_pd"].mean()
        mean_passed = algo_df["constraint_passed"].mean()
        print(f"    {algo}: RMSE={mean_rmse:.2f}Å, Constraints passed={mean_passed:.1f}")


def main():
    """Main function to process systems and check constraint satisfaction."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "system",
        nargs="?",
        help="System name (e.g., 1IXC_A). If not provided, process all systems.",
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=DEFAULT_K_VALUES,
        help=f"K values to process (default: {DEFAULT_K_VALUES})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for results (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=DEFAULT_PREDICTION_PATTERN,
        help=f"File pattern for predictions relative to seed_* directory (default: {DEFAULT_PREDICTION_PATTERN})",
    )
    parser.add_argument(
        "--molprobity",
        type=str,
        default=DEFAULT_MOLPROBITY_REPORT,
        help=f"MolProbity report filename in seed_* directory (default: {DEFAULT_MOLPROBITY_REPORT})",
    )
    parser.add_argument(
        "--steering-dir-template",
        type=str,
        default=None,
        help="Override steering output dir template (e.g. 'steering_output_nstep_k{k}'). "
             "Default: 'steering_output_k{k}'.",
    )
    args = parser.parse_args()

    # Set up paths
    base_dir = Path(__file__).parent
    output_dir = base_dir / args.output_dir

    # Create output directory
    output_dir.mkdir(exist_ok=True)

    # Determine which systems to process
    if args.system:
        systems = [args.system]
    else:
        # Find all systems from constraint files
        constraint_dir = base_dir / "distance_constraint_df"
        systems = sorted([f.stem.replace("_constraint", "") for f in constraint_dir.glob("*_constraint.csv")])

    print("=" * 80)
    print("Constraint Satisfaction Analysis with MolProbity Metrics")
    print("=" * 80)
    print(f"K values:      {args.k_values}")
    print(f"Output dir:    {args.output_dir}")
    print(f"Pattern:       {args.pattern}")
    print(f"MolProbity:    {args.molprobity}")
    print(f"Tolerance:     ±{DISTANCE_TOLERANCE} Å")
    print(f"Systems:       {len(systems)} to process")
    print("=" * 80)
    print()

    # Process each system
    for system_name in systems:
        try:
            process_system(
                system_name=system_name,
                base_dir=base_dir,
                k_values=args.k_values,
                output_dir=output_dir,
                prediction_pattern=args.pattern,
                molprobity_report=args.molprobity,
                steering_dir_template=args.steering_dir_template,
            )
        except Exception as e:
            print(f"  Error processing {system_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
        print()

    print("=" * 80)
    print(f"Done! Results saved in {args.output_dir}/")
    print("=" * 80)


if __name__ == "__main__":
    main()

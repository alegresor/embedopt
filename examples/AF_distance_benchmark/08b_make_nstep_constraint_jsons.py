#!/usr/bin/env python3
"""
Generate distance-constraint configs with paired (N_step, lr) values.

Tests robustness to scaled changes in N_step by holding the optimization budget
constant: N_step * lr ≈ 20. Mirrors mli's compass v0.6 N_step experiment.

For each system: same constraint set as 08 (top-K most-diverged residue pairs),
but each job carries an `n_step` field that 09b reads out per-job.

Usage:
    python 08b_make_nstep_constraint_jsons.py [system] [--k K] [--output-folder FOLDER]
"""

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Reuse top-K logic + contact construction from 08
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "_module_08", Path(__file__).parent / "08_make_distance_constraint_jsons.py"
)
_mod_08 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod_08)
get_top_k_diff = _mod_08.get_top_k_diff
create_constraint_contacts = _mod_08.create_constraint_contacts
create_reward_config = _mod_08.create_reward_config
DISTANCE_MARGIN = _mod_08.DISTANCE_MARGIN


# (N_step, lr) pairs holding N_step * lr = 20 — matches mli's compass v0.6 setting
NSTEP_LR_PAIRS = [
    (160, 0.125),
    (125, 0.16),
    (100, 0.20),
    (80, 0.25),
    (50, 0.40),
    (25, 0.80),
]
ALGORITHMS = ["embedopt", "dps"]
DEFAULT_K = 20


def create_job_config(
    base_config: Dict[str, Any],
    system_name: str,
    algorithm_type: str,
    algorithm_params: Dict[str, Any],
    reward_config: Dict[str, Any],
    k: int,
    n_step: int,
) -> Dict[str, Any]:
    """Single job with N_step embedded in the name and as a top-level field."""
    job = copy.deepcopy(base_config)
    if algorithm_type == "embedopt":
        lr = algorithm_params["reward_scale_s"]
    else:  # dps
        lr = algorithm_params["guided_scale"]
    job["name"] = f"{system_name}_{algorithm_type}_N{n_step}_lr{lr}_k{k}"
    job["reward"] = reward_config
    job["algorithm"] = {"type": algorithm_type, "params": algorithm_params}
    job["n_step"] = n_step
    return job


def generate_benchmark_config(
    system_name: str,
    k: int,
    base_dir: Path,
    gt_dir: Path,
    minimal_json_dir: Path,
    output_dir: Path,
) -> None:
    print(f"Processing {system_name}...")
    base_file = base_dir / f"{system_name}.npy"
    gt_file = gt_dir / f"{system_name}.npy"
    if not base_file.exists() or not gt_file.exists():
        print(f"  Warning: missing base or GT file for {system_name}")
        return

    base_matrix = np.load(base_file)
    gt_matrix = np.load(gt_file)
    diff_matrix = np.abs(gt_matrix - base_matrix)
    df = get_top_k_diff(diff_matrix, gt_matrix, base_matrix, k=k)
    if df.empty:
        print(f"  Warning: no valid constraints for {system_name}")
        return
    print(f"  Found {len(df)} constraints (requested {k})")

    contacts = create_constraint_contacts(df)
    reward_config = create_reward_config(contacts)

    minimal_json_file = minimal_json_dir / f"{system_name}_minimal-update-msa.json"
    if not minimal_json_file.exists():
        print(f"  Warning: minimal JSON not found: {minimal_json_file}")
        return

    with open(minimal_json_file) as f:
        minimal_config = json.load(f)
    if not minimal_config:
        print(f"  Warning: empty config in {minimal_json_file}")
        return
    base_config = minimal_config[0]

    jobs = []
    for n_step, lr in NSTEP_LR_PAIRS:
        jobs.append(create_job_config(
            base_config, system_name, "embedopt",
            {"reward_scale_s": lr, "reward_scale_z": lr},
            reward_config, k, n_step,
        ))
    for n_step, lr in NSTEP_LR_PAIRS:
        jobs.append(create_job_config(
            base_config, system_name, "dps",
            {"guided_scale": lr, "normalize_gradients": True},
            reward_config, k, n_step,
        ))

    output_file = output_dir / f"{system_name}_benchmark.json"
    with open(output_file, "w") as f:
        json.dump(jobs, f, indent=4)
    print(f"  Generated {len(jobs)} jobs -> {output_file}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("system", nargs="?", help="System name. If omitted, process all systems.")
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--output-folder", type=str, default=None,
                        help="Default: benchmark_cfgs_nstep_k{k}")
    args = parser.parse_args()
    if args.output_folder is None:
        args.output_folder = f"benchmark_cfgs_nstep_k{args.k}"

    script_dir = Path(__file__).parent
    base_dir = script_dir / "pair_dist_base"
    gt_dir = script_dir / "pair_dist_gt"
    minimal_json_dir = script_dir / "json_minimal"
    output_dir = script_dir / args.output_folder
    output_dir.mkdir(exist_ok=True)

    if args.system:
        systems = [args.system]
    else:
        systems = sorted([f.stem for f in base_dir.glob("*.npy")])

    print("=" * 80)
    print("N_step Robustness Benchmark Configuration Generator (N_step * lr = 20)")
    print("=" * 80)
    print(f"Output folder: {args.output_folder}")
    print(f"K:             {args.k}")
    print(f"(N_step, lr) pairs:")
    for n, lr in NSTEP_LR_PAIRS:
        print(f"  N_step={n:3d}, lr={lr:.3f}  (product={n*lr:.1f})")
    print(f"Algorithms:    {ALGORITHMS}")
    print(f"Found {len(systems)} systems to process")
    print("=" * 80)
    print()

    for system_name in systems:
        try:
            generate_benchmark_config(system_name, args.k, base_dir, gt_dir, minimal_json_dir, output_dir)
        except Exception as e:
            print(f"  Error processing {system_name}: {e}")
            import traceback; traceback.print_exc()
            continue

    n_jobs = len(NSTEP_LR_PAIRS) * len(ALGORITHMS)
    print("\n" + "=" * 80)
    print(f"Done! Generated configs in {args.output_folder}/")
    print(f"Each config contains {n_jobs} jobs ({len(NSTEP_LR_PAIRS)} per algorithm)")
    print("=" * 80)


if __name__ == "__main__":
    main()

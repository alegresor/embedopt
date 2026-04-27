#!/usr/bin/env python3
"""
Step 7: Generate benchmark configuration files from MSA-updated JSON files.

Expands each minimal JSON to include:
- 1 base (unconditional) job [included by default, disable with --no-base]
- embedopt jobs: one per (learning_rate × resolution)
- dps jobs: one per (guided_scale × resolution)

Usage:
    python 07_make_benchmark_cfgs.py
    python 07_make_benchmark_cfgs.py --algorithms embedopt dps --output-folder my_cfgs
    python 07_make_benchmark_cfgs.py --test   # only 3 systems

Input:  json_minimal/<pdbid>_<chain>_minimal-update-msa.json
Output: <output-folder>/<pdbid>_<chain>_benchmark.json
"""

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).parent.resolve()

DEFAULT_RESOLUTIONS = [5.0]
DEFAULT_LEARNING_RATES = [0.0025, 0.005, 0.01, 0.02, 0.1, 0.2, 0.5, 1.0]
DEFAULT_OUTPUT_FOLDER = "benchmark_cfgs_5A"
DEFAULT_ALGORITHMS = ["embedopt", "dps"]
DEFAULT_INCLUDE_BASE = True
DEFAULT_SAVE_EVERY_N_STEPS = 25

TEST_SYSTEMS = ["8AHU_A", "8ANE_A", "8AU1_A"]


def create_job_config(
    base_config: Dict[str, Any],
    name: str,
    pdb_id: str,
    chain_id: str,
    resolution: float,
    algorithm_type: str,
    algorithm_params: Dict[str, Any],
    save_every_n_steps: int,
) -> Dict[str, Any]:
    job = copy.deepcopy(base_config)
    job["name"] = name
    job["reward"] = {
        "type": "map",
        "params": {
            "ref_pdb_path": str(HERE / "pdb_clean" / f"{pdb_id}_{chain_id}.pdb"),
            "ref_map_path": str(HERE / "synthetic_maps" / f"{resolution}A" / f"{pdb_id}_{chain_id}.map"),
            "resolution": resolution,
            "save_pdb_every_n_steps": save_every_n_steps,
            "energy_relaxation": True,
        },
    }
    job["algorithm"] = {"type": algorithm_type, "params": algorithm_params}
    return job


def generate_benchmark_config(
    minimal_json_path: Path,
    output_dir: Path,
    resolutions: List[float],
    learning_rates: List[float],
    algorithms: List[str],
    include_base: bool,
    save_every_n_steps: int,
) -> None:
    with open(minimal_json_path) as f:
        minimal_config = json.load(f)

    if not minimal_config:
        print(f"Warning: Empty config in {minimal_json_path}")
        return

    base_config = minimal_config[0]

    # Parse PDB id and chain from filename: {PDBid}_{chain}_minimal-update-msa.json
    stem = minimal_json_path.stem.replace("_minimal-update-msa", "")
    parts = stem.split("_")
    if len(parts) < 2:
        print(f"Warning: Cannot parse filename {minimal_json_path.name}")
        return

    pdb_id = parts[0]
    chain_id = parts[1]
    print(f"Processing {pdb_id}_{chain_id}...")

    jobs = []

    if include_base:
        jobs.append(
            create_job_config(
                base_config=base_config,
                name=f"{pdb_id}_{chain_id}_base",
                pdb_id=pdb_id,
                chain_id=chain_id,
                resolution=resolutions[0],
                algorithm_type="base",
                algorithm_params={},
                save_every_n_steps=save_every_n_steps,
            )
        )

    if "embedopt" in algorithms:
        for lr in learning_rates:
            for res in resolutions:
                jobs.append(
                    create_job_config(
                        base_config=base_config,
                        name=f"{pdb_id}_{chain_id}_embedopt_{lr}_{int(res)}A",
                        pdb_id=pdb_id,
                        chain_id=chain_id,
                        resolution=res,
                        algorithm_type="embedopt",
                        algorithm_params={"reward_scale_s": lr, "reward_scale_z": lr},
                        save_every_n_steps=save_every_n_steps,
                    )
                )

    if "dps" in algorithms:
        for scale in learning_rates:
            for res in resolutions:
                jobs.append(
                    create_job_config(
                        base_config=base_config,
                        name=f"{pdb_id}_{chain_id}_dps_{scale}_{int(res)}A",
                        pdb_id=pdb_id,
                        chain_id=chain_id,
                        resolution=res,
                        algorithm_type="dps",
                        algorithm_params={"guided_scale": scale, "normalize_gradients": True},
                        save_every_n_steps=save_every_n_steps,
                    )
                )

    output_file = output_dir / f"{pdb_id}_{chain_id}_benchmark.json"
    with open(output_file, "w") as f:
        json.dump(jobs, f, indent=4)

    print(f"  Generated {len(jobs)} jobs -> {output_file.name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=["embedopt", "dps"],
        default=DEFAULT_ALGORITHMS,
        help="Algorithms to generate configs for (default: embedopt dps)",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default=DEFAULT_OUTPUT_FOLDER,
        help=f"Output folder name (default: {DEFAULT_OUTPUT_FOLDER})",
    )
    parser.add_argument(
        "--resolutions",
        nargs="+",
        type=float,
        default=DEFAULT_RESOLUTIONS,
        help=f"Map resolutions in Angstroms (default: {DEFAULT_RESOLUTIONS})",
    )
    parser.add_argument(
        "--learning-rates",
        nargs="+",
        type=float,
        default=DEFAULT_LEARNING_RATES,
        help=f"Learning rates / guided scales (default: {DEFAULT_LEARNING_RATES})",
    )
    parser.add_argument(
        "--no-base",
        action="store_false",
        dest="include_base",
        default=DEFAULT_INCLUDE_BASE,
        help="Exclude the unconditional base job (included by default)",
    )
    parser.add_argument(
        "--save-every-n-steps",
        type=int,
        default=DEFAULT_SAVE_EVERY_N_STEPS,
        help=f"Save trajectory PDB every N diffusion steps (default: {DEFAULT_SAVE_EVERY_N_STEPS})",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run on only 3 systems for quick testing",
    )
    args = parser.parse_args()

    json_minimal_dir = HERE / "json_minimal"
    output_dir = HERE / args.output_folder
    output_dir.mkdir(exist_ok=True)

    minimal_files = sorted(json_minimal_dir.glob("*_minimal-update-msa.json"))
    if args.test:
        minimal_files = [f for f in minimal_files if any(s in f.stem for s in TEST_SYSTEMS)]
        print(f"Test mode: running on {len(minimal_files)} systems")

    print("=" * 80)
    print("Benchmark Configuration Generator")
    print("=" * 80)
    print(f"Output folder:         {args.output_folder}")
    print(f"Algorithms:            {', '.join(args.algorithms)}")
    print(f"Resolutions:           {args.resolutions} Å")
    print(f"Learning rates:        {args.learning_rates}")
    print(f"Include base:          {args.include_base}")
    print(f"Save every N steps:    {args.save_every_n_steps}")
    print(f"Found {len(minimal_files)} minimal JSON files to process")
    print("=" * 80)
    print()

    for minimal_file in minimal_files:
        try:
            generate_benchmark_config(
                minimal_json_path=minimal_file,
                output_dir=output_dir,
                resolutions=args.resolutions,
                learning_rates=args.learning_rates,
                algorithms=args.algorithms,
                include_base=args.include_base,
                save_every_n_steps=args.save_every_n_steps,
            )
        except Exception as e:
            print(f"Error processing {minimal_file}: {e}")

    n_jobs_per_method = len(args.learning_rates) * len(args.resolutions)
    n_embedopt = n_jobs_per_method if "embedopt" in args.algorithms else 0
    n_dps = n_jobs_per_method if "dps" in args.algorithms else 0
    total_jobs = n_embedopt + n_dps + (1 if args.include_base else 0)

    print(f"\nDone! Configs written to {output_dir}")
    print(f"\nJob counts per protein:")
    print(f"  Total:     {total_jobs}")
    if args.include_base:
        print("  Base:      1")
    if "embedopt" in args.algorithms:
        print(f"  EmbedOpt:  {n_embedopt}")
    if "dps" in args.algorithms:
        print(f"  DPS:       {n_dps}")


if __name__ == "__main__":
    main()

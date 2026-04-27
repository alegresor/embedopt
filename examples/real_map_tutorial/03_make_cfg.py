#!/usr/bin/env python3
"""
Step 3: Generate inference JSON configs from the MSA-updated JSON produced by step 2.

Edit the configuration variables below, then run:
    pixi run python examples/real_map_tutorial/03_make_cfg.py
"""

import json
import copy
from pathlib import Path

HERE = Path(__file__).parent.resolve()

# ── Configuration ─────────────────────────────────────────────────────────────

# Algorithms to generate configs for.
# Options: "embedopt", "dps", "base"
ALGORITHMS = ["embedopt", "dps", "base"]

# Learning rates for embedopt (reward_scale_s/z) and dps (guided_scale).
# One config file is produced per (algorithm, lr) combination.
# Ignored for unconditional.
LEARNING_RATES = [0.05, 0.1]

# Reference PDB for Kabsch alignment during map reward computation.
REF_PDB = str(HERE / "9UGC.pdb")

# Cropped MRC map (output of step 0 / embedopt-crop-map).
REF_MAP = str(HERE / "emd_64136_cropped.mrc")

# Map resolution in Angstroms.
RESOLUTION = 3.52

# Output directory for generated JSON configs.
OUT_DIR = HERE / "inference_configs"

# ─────────────────────────────────────────────────────────────────────────────

MSA_JSON = HERE / "9UGC_A_minimal-update-msa.json"


def fmt_lr(lr: float) -> str:
    return str(lr).rstrip("0").rstrip(".")


def main():
    with open(MSA_JSON) as f:
        base = json.load(f)[0]

    sample_name = base["name"]
    reward = {
        "type": "map",
        "params": {
            "ref_pdb_path": REF_PDB,
            "ref_map_path": REF_MAP,
            "resolution": RESOLUTION,
            "save_pdb_every_n_steps": 25,   # save trajectory PDB every N diffusion steps
            "energy_relaxation": True,       # run OpenMM energy minimization on the final structure
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for algorithm in ALGORITHMS:
        if algorithm == "base":
            cfg = copy.deepcopy(base)
            cfg["name"] = f"{sample_name}_base"
            cfg["algorithm"] = {"type": "base", "params": {}}
            cfg["reward"] = reward
            out = OUT_DIR / f"{sample_name}_base.json"
            out.write_text(json.dumps([cfg], indent=4))
            print(f"Wrote {out}")

        elif algorithm == "dps":
            for lr in LEARNING_RATES:
                cfg = copy.deepcopy(base)
                cfg["name"] = f"{sample_name}_dps_lr{fmt_lr(lr)}"
                cfg["algorithm"] = {"type": "dps", "params": {"guided_scale": lr, "normalize_gradients": True}}
                cfg["reward"] = reward
                out = OUT_DIR / f"{sample_name}_dps_lr{fmt_lr(lr)}.json"
                out.write_text(json.dumps([cfg], indent=4))
                print(f"Wrote {out}")

        elif algorithm == "embedopt":
            for lr in LEARNING_RATES:
                cfg = copy.deepcopy(base)
                cfg["name"] = f"{sample_name}_embedopt_lr{fmt_lr(lr)}"
                cfg["algorithm"] = {
                    "type": "embedopt",
                    "params": {"reward_scale_s": lr, "reward_scale_z": lr},
                }
                cfg["reward"] = reward
                out = OUT_DIR / f"{sample_name}_embedopt_lr{fmt_lr(lr)}.json"
                out.write_text(json.dumps([cfg], indent=4))
                print(f"Wrote {out}")


if __name__ == "__main__":
    main()

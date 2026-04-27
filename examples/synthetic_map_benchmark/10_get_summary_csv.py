#!/usr/bin/env python3
"""
Step 10: Summarize validation results into a single CSV.

Reads phenix.molprobity and phenix.validation_cryoem logs from benchmark_output/
and writes one row per (system, config, seed) to a flat CSV for downstream analysis.

Input:  benchmark_output/<system>/<config>/seed_<N>/
            validation_relaxed.log         (CC_mask, from phenix.validation_cryoem)
            validation_relaxed_5A.log      (base configs use resolution-suffixed name)
            molprobity_relaxed_molprobity.log   (MolProbity score, Rama, Clashscore)
            molprobity_relaxed__molprobity.log  (same, legacy double-underscore name)
Output: benchmark_summary.csv
"""

import argparse
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent.resolve()
OUTPUT_DIR = HERE / "benchmark_output"
SEEDS = [16, 42, 101]
TEST_SYSTEMS = ["8AHU_A", "8ANE_A", "8AU1_A"]


# ── Metric extraction ──────────────────────────────────────────────────────────

def _read(path: Path) -> str:
    try:
        return path.read_text()
    except FileNotFoundError:
        return ""


def extract_cc_mask(log: str) -> float | None:
    m = re.search(r"CC_mask\s*[=:]\s*([+-]?\d+\.?\d*)", log)
    return float(m.group(1)) if m else None


def extract_molprobity_score(log: str) -> float | None:
    m = re.search(r"MolProbity score\s*=\s*(\d+\.?\d*)", log)
    return float(m.group(1)) if m else None


def extract_rama_outliers(log: str) -> float | None:
    m = re.search(r"Ramachandran outliers\s*=\s*(\d+\.?\d*)\s*%", log)
    return float(m.group(1)) if m else None


def extract_clashscore(log: str) -> float | None:
    m = re.search(r"Clashscore\s*=\s*(\d+\.?\d*)", log)
    return float(m.group(1)) if m else None


# ── Config name parsing ────────────────────────────────────────────────────────

def parse_config_name(system: str, config_name: str) -> dict:
    """Extract algorithm, lr, resolution from config name."""
    suffix = config_name[len(system) + 1:]  # strip "{system}_"

    if suffix == "base":
        return {"Algorithm": "base", "LR": None, "Resolution": 5.0}

    # e.g. "embedopt_0.5_5A" or "dps_0.0025_5A"
    m = re.match(r"(embedopt|dps)_([\d.]+)_(\d+)A$", suffix)
    if m:
        return {
            "Algorithm": m.group(1),
            "LR": float(m.group(2)),
            "Resolution": float(m.group(3)),
        }

    return {"Algorithm": suffix, "LR": None, "Resolution": None}


# ── Main collection ────────────────────────────────────────────────────────────

def collect_records(systems: list[str]) -> list[dict]:
    records = []

    for system in systems:
        system_dir = OUTPUT_DIR / system
        if not system_dir.is_dir():
            print(f"  Skipping {system}: no output directory")
            continue

        for config_dir in sorted(system_dir.iterdir()):
            if not config_dir.is_dir():
                continue
            config_name = config_dir.name
            parsed = parse_config_name(system, config_name)

            for seed in SEEDS:
                seed_dir = config_dir / f"seed_{seed}"
                if not seed_dir.is_dir():
                    continue

                # CC_mask: base uses resolution-suffixed log name
                if parsed["Algorithm"] == "base":
                    val_log = _read(seed_dir / "validation_relaxed_5A.log")
                    if not val_log:
                        val_log = _read(seed_dir / "validation_relaxed.log")
                else:
                    val_log = _read(seed_dir / "validation_relaxed.log")

                mp_log = _read(seed_dir / "molprobity_relaxed_molprobity.log") or \
                         _read(seed_dir / "molprobity_relaxed__molprobity.log")

                records.append({
                    "System": system,
                    "Config": config_name,
                    "Algorithm": parsed["Algorithm"],
                    "LR": parsed["LR"],
                    "Resolution": parsed["Resolution"],
                    "Seed": seed,
                    "CC_mask": extract_cc_mask(val_log),
                    "MolProbity_Score": extract_molprobity_score(mp_log),
                    "Rama_Outliers": extract_rama_outliers(mp_log),
                    "Clashscore": extract_clashscore(mp_log),
                })

    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=HERE / "benchmark_summary.csv",
                        help="Output CSV path (default: benchmark_summary.csv)")
    parser.add_argument("--test", action="store_true",
                        help="Run on only 3 systems for quick testing")
    args = parser.parse_args()

    if args.test:
        systems = TEST_SYSTEMS
        print(f"Test mode: {len(systems)} systems")
    else:
        systems = sorted(d.name for d in OUTPUT_DIR.iterdir() if d.is_dir()) if OUTPUT_DIR.is_dir() else []
        print(f"Found {len(systems)} systems in {OUTPUT_DIR.name}/")

    if not systems:
        print("ERROR: No systems found. Run step 08 first.")
        return

    print("Collecting records...")
    records = collect_records(systems)

    df = pd.DataFrame(records)
    df.to_csv(args.output, index=False)
    print(f"\nWrote {len(df)} records to {args.output.name}")

    # Summary
    print(f"\nSummary:")
    print(f"  Systems:    {df['System'].nunique()}")
    print(f"  Algorithms: {sorted(df['Algorithm'].unique())}")
    print(f"  Seeds:      {sorted(df['Seed'].unique())}")
    for col in ["CC_mask", "MolProbity_Score", "Rama_Outliers", "Clashscore"]:
        n = df[col].notna().sum()
        pct = 100 * n / len(df)
        print(f"  {col}: {n}/{len(df)} ({pct:.0f}%) non-null")


if __name__ == "__main__":
    main()

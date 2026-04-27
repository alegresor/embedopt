#!/usr/bin/env python3
"""
Step 5: Create minimal Protenix-format JSON inputs.

Reads each PDB from pdb_clean/ and the corresponding FASTA sequence,
then writes a minimal JSON (sequence only, no MSA paths yet).

Input:  pdb_clean/<pdbid>_<chain>.pdb, fasta/<pdbid>_<chain>.fa
Output: json_minimal/<pdbid>_<chain>_minimal.json
"""

import argparse
import json
from pathlib import Path

from tqdm import tqdm

HERE = Path(__file__).parent.resolve()

PDB_DIR = HERE / "pdb_clean"
FASTA_DIR = HERE / "fasta"
OUTPUT_DIR = HERE / "json_minimal"

TEST_SYSTEMS = ["8AHU_A", "8ANE_A", "8AU1_A"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run on only 3 systems for quick testing")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)

    pdb_files = sorted(PDB_DIR.glob("*.pdb"))
    if args.test:
        pdb_files = [p for p in pdb_files if p.stem in TEST_SYSTEMS]
        print(f"Test mode: running on {len(pdb_files)} systems")
    print(f"Found {len(pdb_files)} PDB files")

    processed = 0
    skipped = 0
    failed = []

    for pdb_path in tqdm(pdb_files, desc="Creating minimal JSONs"):
        pdb_chain = pdb_path.stem
        json_name = f"{pdb_chain}_minimal"
        output_file = OUTPUT_DIR / f"{json_name}.json"

        if output_file.exists():
            skipped += 1
            continue

        fasta_file = FASTA_DIR / f"{pdb_chain}.fa"
        if not fasta_file.exists():
            failed.append({"pdb_chain": pdb_chain, "error": f"FASTA file not found: {fasta_file}"})
            continue

        try:
            sequence_lines = []
            with open(fasta_file) as f:
                for line in f:
                    if not line.startswith(">"):
                        sequence_lines.append(line.strip())
            sequence = "".join(sequence_lines)

            if not sequence:
                raise ValueError("Empty sequence")

            json_data = [{"name": json_name, "sequences": [{"proteinChain": {"sequence": sequence, "count": 1}}]}]
            with open(output_file, "w") as f:
                json.dump(json_data, f, indent=2)

            processed += 1

        except Exception as e:
            failed.append({"pdb_chain": pdb_chain, "error": str(e)})

    print(f"\nCreated: {processed}, skipped: {skipped}, failed: {len(failed)}")

    if failed:
        import pandas as pd

        pd.DataFrame(failed).to_csv(HERE / "failed_json_generations.csv", index=False)
        print(f"Failed generations saved to failed_json_generations.csv")
        for f in failed[:10]:
            print(f"  {f['pdb_chain']}: {f['error']}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")


if __name__ == "__main__":
    main()

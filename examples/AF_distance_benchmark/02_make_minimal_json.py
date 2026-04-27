#!/usr/bin/env python3
"""
Create minimal JSON files for each PDB in pdb_clean/
Uses the corresponding sequence from fasta_extracted/
Output: json_minimal/{PDBid}_{Chainid}_minimal.json
"""

import json
from pathlib import Path
from tqdm import tqdm

# Directories
PDB_DIR = Path("pdb_extracted")
FASTA_DIR = Path("fasta_extracted")
OUTPUT_DIR = Path("json_minimal")

# Create output directory
OUTPUT_DIR.mkdir(exist_ok=True)

print("Finding PDB files...")
pdb_files = sorted(PDB_DIR.glob("*.pdb"))
print(f"Found {len(pdb_files)} PDB files")

# Track statistics
processed = 0
skipped = 0
failed = []

print("\nCreating minimal JSON files...")
for pdb_path in tqdm(pdb_files, desc="Processing PDB files"):
    # Extract PDBchain (e.g., "8AHU_A")
    pdb_chain = pdb_path.stem

    # Output JSON file path
    json_name = f"{pdb_chain}_minimal"
    output_file = OUTPUT_DIR / f"{json_name}.json"

    # Skip if already exists
    if output_file.exists():
        skipped += 1
        continue

    # Find corresponding FASTA file
    fasta_file = FASTA_DIR / f"{pdb_chain}.fasta"

    if not fasta_file.exists():
        failed.append({"pdb_chain": pdb_chain, "error": f"FASTA file not found: {fasta_file}"})
        continue

    try:
        # Read sequence from FASTA file
        sequence_lines = []
        with open(fasta_file, "r") as f:
            for line in f:
                if line.startswith(">"):
                    continue  # Skip header line
                sequence_lines.append(line.strip())

        sequence = "".join(sequence_lines)

        if not sequence:
            raise ValueError("Empty sequence")

        # Create JSON structure
        json_data = [{"name": json_name, "sequences": [{"proteinChain": {"sequence": sequence, "count": 1}}]}]

        # Write JSON file
        with open(output_file, "w") as f:
            json.dump(json_data, f, indent=2)

        processed += 1

    except Exception as e:
        failed.append({"pdb_chain": pdb_chain, "error": str(e)})

print("\n" + "=" * 50)
print("JSON Generation Summary:")
print("=" * 50)
print(f"Successfully created: {processed} JSON files")
print(f"Already existed (skipped): {skipped} JSON files")
print(f"Failed: {len(failed)} files")

if failed:
    print("\nFailed generations:")
    for f in failed[:10]:  # Show first 10
        print(f"  {f['pdb_chain']}: {f['error']}")
    if len(failed) > 10:
        print(f"  ... and {len(failed) - 10} more")

    # Save failed list to a file
    import pandas as pd

    failed_df = pd.DataFrame(failed)
    failed_df.to_csv("failed_json_generations.csv", index=False)
    print("\nFailed generations saved to: failed_json_generations.csv")

print("\nDone!")
print(f"JSON files saved to: {OUTPUT_DIR}/")

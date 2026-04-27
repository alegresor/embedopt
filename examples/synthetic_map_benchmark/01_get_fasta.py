#!/usr/bin/env python3
"""
Step 1: Extract individual FASTA files from the PISCES FASTA bundle.

Reads pisces_subsampled.csv and pulls the matching sequence for each
PDBchain from the bulk FASTA file, writing one .fa file per protein.

Input:  pisces_subsampled.csv, pisces_raw/*.fasta
Output: fasta/<pdbid>_<chain>.fa  (80 files)
"""

from pathlib import Path

import pandas as pd
from tqdm import tqdm

HERE = Path(__file__).parent.resolve()

CSV_FILE = HERE / "pisces_subsampled.csv"
FASTA_FILE = (
    HERE
    / "pisces_raw"
    / "cullpdb_pc25.0_res0.0-4.5_noBrks_noDsdr_len150-1000_R0.3_Xray+EM_d2025_11_06_chains4084.fasta"
)
OUTPUT_DIR = HERE / "fasta"


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Reading CSV file...")
    df = pd.read_csv(CSV_FILE)
    needed_chains = set(df["PDBchain"].values)
    print(f"Found {len(df)} entries to extract")

    print("Parsing FASTA file...")
    fasta_dict = {}
    current_id = None
    current_header = None
    current_seq = []

    with open(FASTA_FILE) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if current_id and current_id in needed_chains:
                    fasta_dict[current_id] = {"header": current_header, "sequence": "".join(current_seq)}
                current_header = line
                current_id = line.split()[0][1:]
                current_seq = []
            elif line:
                current_seq.append(line)

    if current_id and current_id in needed_chains:
        fasta_dict[current_id] = {"header": current_header, "sequence": "".join(current_seq)}

    print(f"Found {len(fasta_dict)} matching sequences in FASTA file")

    missing = []
    saved = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Writing FASTA files"):
        pdb_chain = row["PDBchain"]
        pdb_id = row["PDBid"]
        chain_id = row["chain_id"]

        if pdb_chain in fasta_dict:
            output_file = OUTPUT_DIR / f"{pdb_id}_{chain_id}.fa"
            seq = fasta_dict[pdb_chain]["sequence"]
            with open(output_file, "w") as f:
                f.write(fasta_dict[pdb_chain]["header"] + "\n")
                for i in range(0, len(seq), 80):
                    f.write(seq[i : i + 80] + "\n")
            saved += 1
        else:
            missing.append(pdb_chain)
            print(f"  WARNING: {pdb_chain} not found in FASTA file!")

    print(f"\nSaved {saved} FASTA files to {OUTPUT_DIR}")
    if missing:
        print(f"Missing ({len(missing)}): {', '.join(missing)}")


if __name__ == "__main__":
    main()

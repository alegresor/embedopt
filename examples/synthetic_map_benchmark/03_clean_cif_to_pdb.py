#!/usr/bin/env python3
"""
Step 3: Clean CIF files and convert to PDB.

For each entry in cif_raw/:
1. Extract only the specified polymer chain
2. Verify sequence matches the FASTA file
3. Set all B-factors to 50 (isotropic), zero out anisotropic B-factors
4. Set spacegroup to P1, unit cell = molecule bounding box + 10 Å padding per side
5. Center the molecule in the box
6. Save as PDB

Input:  cif_raw/<pdbid>_<chain>.cif, fasta/<pdbid>_<chain>.fa
Output: pdb_clean/<pdbid>_<chain>.pdb
"""

from pathlib import Path

import gemmi
import numpy as np
import pandas as pd
from SFC_Torch import PDBParser
from tqdm import tqdm

HERE = Path(__file__).parent.resolve()

CIF_DIR = HERE / "cif_raw"
FASTA_DIR = HERE / "fasta"
OUTPUT_DIR = HERE / "pdb_clean"

PADDING = 10.0  # Å added to each side of the bounding box


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    cif_files = sorted(CIF_DIR.glob("*.cif"))
    print(f"Found {len(cif_files)} CIF files")

    processed = 0
    skipped = 0
    failed = []

    for cif_path in tqdm(cif_files, desc="Converting CIF to PDB"):
        pdb_chain = cif_path.stem
        pdb_id, chain_id = pdb_chain.split("_")
        output_file = OUTPUT_DIR / f"{pdb_chain}.pdb"

        if output_file.exists():
            skipped += 1
            continue

        try:
            model = PDBParser(str(cif_path))
            chain = model.selection(f"//{chain_id};polymer")

            # Verify sequence matches FASTA
            fasta_path = FASTA_DIR / f"{pdb_chain}.fa"
            if fasta_path.exists():
                seq_from_fasta = []
                with open(fasta_path) as f:
                    for line in f:
                        if not line.startswith(">"):
                            seq_from_fasta.append(line.strip())
                if chain.sequence.strip() != "".join(seq_from_fasta):
                    raise ValueError("Sequence mismatch between CIF and FASTA")

            # Uniform B-factors
            chain.set_baniso(np.zeros_like(chain.atom_b_aniso))
            chain.set_biso(50.0 * np.ones_like(chain.atom_b_iso))

            # P1 unit cell: bounding box + 10 Å padding on each side
            chain.set_spacegroup("P1")
            x_size = chain.atom_pos[:, 0].max() - chain.atom_pos[:, 0].min()
            y_size = chain.atom_pos[:, 1].max() - chain.atom_pos[:, 1].min()
            z_size = chain.atom_pos[:, 2].max() - chain.atom_pos[:, 2].min()
            cell_size = np.array([x_size + 2 * PADDING, y_size + 2 * PADDING, z_size + 2 * PADDING])
            chain.set_unitcell(gemmi.UnitCell(*cell_size, 90, 90, 90))

            # Center molecule in box
            com = chain.atom_pos.mean(axis=0)
            chain.set_positions(chain.atom_pos + (-com + cell_size / 2))

            chain.pdb_header = []
            chain.savePDB(str(output_file))
            processed += 1

        except Exception as e:
            failed.append({"PDBchain": pdb_chain, "error": str(e)})
            if output_file.exists():
                output_file.unlink()

    print(f"\nProcessed: {processed}, skipped: {skipped}, failed: {len(failed)}")

    if failed:
        failed_path = HERE / "failed_conversions.csv"
        pd.DataFrame(failed).to_csv(failed_path, index=False)
        print(f"Failed conversions saved to {failed_path}")
        for f in failed:
            print(f"  {f['PDBchain']}: {f['error']}")


if __name__ == "__main__":
    main()

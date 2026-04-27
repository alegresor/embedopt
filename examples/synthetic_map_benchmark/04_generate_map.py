#!/usr/bin/env python3
"""
Step 4: Render PDB files to CCP4 map with SFcalculator.

Uses SFC_Torch to compute synthetic cryo-EM electron density maps from
cleaned PDB structures. Maps are normalized to zero mean and unit variance.

SFC_Torch reference:
  Li et al. "SFCalculator: connecting deep generative models and crystallography" bioRxiv (2025).
  https://www.biorxiv.org/content/10.1101/2025.01.12.632630v2

Input:  pdb_clean/<pdbid>_<chain>.pdb
Output: synthetic_maps/<res>A/<pdbid>_<chain>.map
"""

import argparse
from pathlib import Path

import SFC_Torch as sfc
import gemmi
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

HERE = Path(__file__).parent.resolve()

PDB_DIR = HERE / "pdb_clean"
OUTPUT_BASE_DIR = HERE / "synthetic_maps"
RESOLUTIONS = [5.0]  # Angstroms
DEVICE = "cuda:0"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run on only 3 systems for quick testing")
    args = parser.parse_args()

    for res in RESOLUTIONS:
        (OUTPUT_BASE_DIR / f"{res}A").mkdir(parents=True, exist_ok=True)

    pdb_files = sorted(PDB_DIR.glob("*.pdb"))
    if args.test:
        pdb_files = pdb_files[:3]
        print(f"Test mode: running on {len(pdb_files)} systems")
    print(f"Found {len(pdb_files)} PDB files")
    print(f"Resolutions: {RESOLUTIONS} Å  |  Total maps: {len(pdb_files) * len(RESOLUTIONS)}")

    processed = 0
    skipped = 0
    failed = []

    for pdb_path in tqdm(pdb_files, desc="Generating maps"):
        pdb_chain = pdb_path.stem

        for resolution in RESOLUTIONS:
            output_file = OUTPUT_BASE_DIR / f"{resolution}A" / f"{pdb_chain}.map"

            if output_file.exists():
                skipped += 1
                continue

            try:
                dcp = sfc.SFcalculator(str(pdb_path), dmin=resolution, mode="cryoem", device=DEVICE)
                dcp.inspect_data()

                Fprotein = dcp.calc_fprotein(Return=True)
                rs_grid = sfc.mask.reciprocal_grid(dcp.Hasu_array, Fprotein, dcp.gridsize)

                map_grid = torch.real(torch.fft.fftn(rs_grid, dim=(-3, -2, -1)))
                map_grid_norm = (map_grid - map_grid.mean()) / map_grid.std()

                ccp4 = gemmi.Ccp4Map()
                ccp4.grid.set_size(dcp.gridsize[0], dcp.gridsize[1], dcp.gridsize[2])
                ccp4.grid.unit_cell.set(
                    dcp.unit_cell.a, dcp.unit_cell.b, dcp.unit_cell.c,
                    dcp.unit_cell.alpha, dcp.unit_cell.beta, dcp.unit_cell.gamma,
                )
                ccp4.grid.spacegroup = dcp.space_group

                grid_data_view = np.array(ccp4.grid, copy=False)
                grid_data_view[:] = map_grid_norm.cpu().numpy()
                ccp4.update_ccp4_header()
                ccp4.write_ccp4_map(str(output_file))

                processed += 1

            except Exception as e:
                failed.append({"PDBchain": pdb_chain, "resolution": resolution, "error": str(e)})
                if output_file.exists():
                    output_file.unlink()

    print(f"\nGenerated: {processed}, skipped: {skipped}, failed: {len(failed)}")

    if failed:
        failed_path = HERE / "failed_map_generations.csv"
        pd.DataFrame(failed).to_csv(failed_path, index=False)
        print(f"Failed generations saved to {failed_path}")
        for f in failed[:10]:
            print(f"  {f['PDBchain']} @ {f['resolution']}A: {f['error']}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")


if __name__ == "__main__":
    main()

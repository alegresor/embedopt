#!/usr/bin/env python3
"""
Step 2: Download CIF structure files from RCSB PDB.

Downloads the full assembly CIF for each PDB entry in pisces_subsampled.csv.
Skips files that already exist, so safe to re-run after partial failures.

Input:  pisces_subsampled.csv
Output: cif_raw/<pdbid>_<chain>.cif  (one per row; named by chain for traceability)
"""

import time
import urllib.request
from pathlib import Path

import pandas as pd
from tqdm import tqdm

HERE = Path(__file__).parent.resolve()

CSV_FILE = HERE / "pisces_subsampled.csv"
OUTPUT_DIR = HERE / "cif_raw"


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(CSV_FILE)
    print(f"Found {len(df)} entries to download")

    downloaded = 0
    skipped = 0
    failed = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Downloading CIF files"):
        pdb_id = row["PDBid"]
        chain_id = row["chain_id"]
        output_file = OUTPUT_DIR / f"{pdb_id}_{chain_id}.cif"

        if output_file.exists():
            skipped += 1
            continue

        url = f"https://files.rcsb.org/download/{pdb_id.upper()}.cif"
        try:
            urllib.request.urlretrieve(url, str(output_file))
            downloaded += 1
            time.sleep(0.1)
        except Exception as e:
            failed.append({"PDBid": pdb_id, "chain_id": chain_id, "error": str(e)})
            if output_file.exists():
                output_file.unlink()

    print(f"\nDownloaded: {downloaded}, skipped: {skipped}, failed: {len(failed)}")

    if failed:
        failed_df = pd.DataFrame(failed)
        failed_path = HERE / "failed_downloads.csv"
        failed_df.to_csv(failed_path, index=False)
        print(f"Failed downloads saved to {failed_path}")
        for f in failed:
            print(f"  {f['PDBid']}_{f['chain_id']}: {f['error']}")


if __name__ == "__main__":
    main()

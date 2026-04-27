#!/usr/bin/env python3
"""
Step 0: Subsample ~100 proteins from a PISCES PDB culling list.

Filters to recent structures (PDB IDs starting with "8") and chains A-H,
then attempts to draw 10 proteins uniformly from each of 10 length bins
(150-1000 aa). Sparse bins (long sequences) yield fewer than 10 entries,
so the actual output is 80 proteins.
Sampling is fully deterministic (random_state=42 throughout).

Input (in pisces_raw/):
    cullpdb_pc25.0_res0.0-4.5_noBrks_noDsdr_len150-1000_R0.3_Xray+EM_d2025_11_06_chains4084

Output:
    pisces_subsampled.csv   (80 entries, committed to repo)
    pisces_len_distribution.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent.resolve()

PISCES_FILE = (
    HERE
    / "pisces_raw"
    / "cullpdb_pc25.0_res0.0-4.5_noBrks_noDsdr_len150-1000_R0.3_Xray+EM_d2025_11_06_chains4084"
)
OUTPUT_CSV = HERE / "pisces_subsampled.csv"
OUTPUT_PLOT = HERE / "pisces_len_distribution.png"

NUM_BINS = 10
TOTAL_SAMPLES = 100
RANDOM_STATE = 42


def main():
    pisces_df = pd.read_csv(PISCES_FILE, sep=r"\s+", header=0)

    # Keep only recent structures (PDB ID starts with "8") and chains A-H
    df = pisces_df[pisces_df["PDBchain"].str.startswith("8")].copy()
    df["PDBid"] = df["PDBchain"].str[:4]
    df["chain_id"] = df["PDBchain"].str[4:]
    df = df[df["chain_id"].isin(list("ABCDEFGH"))].copy()

    # Stratified sampling: 10 uniform-width bins across 150-1000 aa
    bin_edges = np.linspace(150, 1000, NUM_BINS + 1)
    df["len_bin"] = pd.cut(df["len"], bins=bin_edges, include_lowest=True, right=True)

    samples_per_bin = TOTAL_SAMPLES // NUM_BINS
    extra = TOTAL_SAMPLES - samples_per_bin * NUM_BINS  # distribute remainder to first bins

    sampled_dfs = []
    for i, b in enumerate(df["len_bin"].cat.categories):
        this_bin = df[df["len_bin"] == b]
        n = min(samples_per_bin + (1 if i < extra else 0), len(this_bin))
        if n > 0:
            sampled = this_bin.sample(n=n, random_state=RANDOM_STATE) if len(this_bin) > n else this_bin.copy()
            sampled_dfs.append(sampled)

    final_sample = pd.concat(sampled_dfs)
    n_final = min(len(final_sample), TOTAL_SAMPLES)
    final_sample = final_sample.sample(n=n_final, random_state=RANDOM_STATE)

    final_sample.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {len(final_sample)} entries to {OUTPUT_CSV}")

    plt.figure()
    plt.hist(final_sample["len"], bins=bin_edges)
    plt.xlabel("Sequence length (aa)")
    plt.ylabel("Count")
    plt.title("Sampled length distribution")
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=150)
    plt.close()
    print(f"Saved length distribution plot to {OUTPUT_PLOT}")


if __name__ == "__main__":
    main()

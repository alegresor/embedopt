"""
1. Perform a sequence alignment between the extracted ground-truth PDB (with missing residues) and
the base Protenix-generated structure.
2. For each pair of aligned residues, compute the Ca–Ca distances separately in:
   • the ground-truth PDB
   • the Protenix-generated structure
3. Assemble two pairwise distance matrices (one for the ground-truth PDB and one for the Protenix model).
4. In the ground-truth matrix, insert “nan” entries wherever residues are missing in the PDB.
5. Save each distance matrix as a NumPy file (.npy).

Reads the base prediction from `base_output/{system}/{system}_minimal/seed_${BASE_SEED}/predictions/...`.
BASE_SEED defaults to 16. Override via env var, e.g. `BASE_SEED=101 python 06_...py`.

Output:
pair_dist_gt/{PDBid}_{Chainid}.npy
pair_dist_base/{PDBid}_{Chainid}.npy
"""

import re
import os
import numpy as np
import pandas as pd
import skbio
from SFC_Torch import PDBParser


def compute_pairwise_distance(x: np.ndarray):
    """
    Compute pairwise distance between the atoms in the array
    Args:
        x: np.array of shape (N, 3)
    Returns:
        pairwise_distance: np.array of shape (N, N)
    """
    pairwise_distance = np.linalg.norm(x[None, :, :] - x[:, None, :], axis=-1)
    return pairwise_distance


def get_common_ca_ind(pdb1: PDBParser, pdb2: PDBParser):
    """
    A known bug: it can throw some residues out when the two pdbs have ideentical sequences
    for example, "DFGTT" for both, and it will only keep "GTT"
    be less strict, as there are some residue modifications, like "MET" -> "MSE"
    """  # noqa: E501
    seq1 = pdb1.sequence.strip().upper()
    seq2 = pdb2.sequence.strip().upper()
    alignment = skbio.alignment.StripedSmithWaterman(seq1)(seq2)  # Align sequence with Smith Waterman Algorithm
    subind_1 = np.arange(alignment.query_begin, alignment.query_end + 1)
    subind_2 = np.arange(alignment.target_begin, alignment.target_end_optimal + 1)
    subsubind_1, subsubind_2 = get_identical_indices(alignment.aligned_query_sequence, alignment.aligned_target_sequence)
    common_seq1 = subind_1[subsubind_1]
    common_seq2 = subind_2[subsubind_2]
    common_ca_ind_1 = [get_pattern_index(pdb1.cra_name, rf".*-{j}-.*-CA$") for j in common_seq1]
    common_ca_ind_2 = [get_pattern_index(pdb2.cra_name, rf".*-{i}-.*-CA$") for i in common_seq2]
    assert (
        np.array([i[-6] + "-" + i[-2:] for i in np.array(pdb1.cra_name)[common_ca_ind_1]])
        == np.array([i[-6] + "-" + i[-2:] for i in np.array(pdb2.cra_name)[common_ca_ind_2]])
    ).all()
    return common_ca_ind_1, common_ca_ind_2


def get_identical_indices(A, B):
    """
    Get indices of aligned string A and B to produce identical sequence

    >>> A = "EWTUY"
    >>> B = "E-RUY"
    >>> get_identical_indices(A, B)
    [0,3,4], [0,2,3]

    So A[0,3,4] = 'EUY' = B[0,2,3]
    """
    ind_A = []
    ind_B = []
    ai = 0
    bi = 0
    for a, b in zip(A, B, strict=False):
        if a == "-":
            bi += 1
            continue
        if b == "-":
            ai += 1
            continue
        if a == b:
            ind_A.append(ai)
            ind_B.append(bi)
            ai += 1
            bi += 1
        else:
            ai += 1
            bi += 1
    return np.array(ind_A), np.array(ind_B)


def get_pattern_index(str_list, pattern):
    return next((i for i, s in enumerate(str_list) if re.match(pattern, s)), None)


BASE_SEED = os.environ.get("BASE_SEED", "16")

os.makedirs("pair_dist_gt", exist_ok=True)
os.makedirs("pair_dist_base", exist_ok=True)

# Get all systems from pdb_extracted directory
pdb_files = [f for f in os.listdir("pdb_extracted") if f.endswith(".pdb")]
systems = [f.replace(".pdb", "") for f in pdb_files]

print(f"Reading base predictions from seed_{BASE_SEED}/")

results = []
for system in systems:
    print(f"Processing {system}...")
    try:
        gt_pdb = PDBParser(f"pdb_extracted/{system}.pdb")
        proteinix_pdb = PDBParser(f"base_output/{system}/{system}_minimal/seed_{BASE_SEED}/predictions/{system}_minimal_sample_0.cif")

        ind1, ind2 = get_common_ca_ind(gt_pdb, proteinix_pdb)

        pair_dist_gt = compute_pairwise_distance(gt_pdb.atom_pos[ind1])
        pair_dist_proteinix = compute_pairwise_distance(proteinix_pdb.atom_pos[ind2])
        N_diff = (np.abs(pair_dist_gt - pair_dist_proteinix) > 40).sum() / 2

        N = len(proteinix_pdb.sequence.strip())
        M = len(ind2)
        selected_indices = np.array(proteinix_pdb.from_atom_slices(ind2).res_id) - 1  # The indices of the selected residues
        gt_pair_dist_matrix_filled = np.full((N, N), np.nan)
        gt_pair_dist_matrix_filled[np.ix_(selected_indices, selected_indices)] = pair_dist_gt
        proteinix_pair_dist_matrix_filled = np.full((N, N), np.nan)
        proteinix_pair_dist_matrix_filled[np.ix_(selected_indices, selected_indices)] = pair_dist_proteinix

        assert gt_pair_dist_matrix_filled.shape == (N, N)
        assert proteinix_pair_dist_matrix_filled.shape == (N, N)

        # Save the distance matrices
        np.save(f"pair_dist_gt/{system}.npy", gt_pair_dist_matrix_filled)
        np.save(f"pair_dist_base/{system}.npy", proteinix_pair_dist_matrix_filled)

        results.append(
            {
                "system": system,
                "Total_residues": N,
                "common_residues": len(ind2),
                "N_diff > 40A": N_diff,
            }
        )
    except Exception as e:
        print(f"  Error processing {system}: {e}")
        continue

df = pd.DataFrame(results)
df.to_csv("pair_distance_overview.csv", index=False)

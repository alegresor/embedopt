import numpy as np
from SFC_Torch import PDBParser
import skbio
import re
import torch


def iterative_kabsch_alignment_common_ca(
    moving_tensor: torch.Tensor,
    ref_ca_numpy: np.ndarray,
    common_ca_ind: np.ndarray,
    cutoff: float = 2.0,
    cycles: int = 5,
) -> torch.Tensor:
    """
    Performs Kabsch alignment with iterative outlier rejection on protein structures.

    Aligns moving_tensor[common_ca_ind] to ref_ca_numpy, then applies the
    global transformation to the entire moving_tensor.

    Args:
        moving_tensor: Coordinates to move, shape [N_atoms, 3].
        ref_ca_numpy: Reference coordinates, shape [N_common_ca, 3].
        common_ca_ind: Common CA indices to select from moving_tensor for alignment.
        cutoff: Distance cutoff in Angstroms for outlier rejection.
        cycles: Max number of refinement cycles. Set to 0 for a single
            alignment without outlier rejection.

    Returns:
        aligned_tensor: Transformed moving_tensor with shape [N_atoms, 3].
    """
    # Extract common CA atoms from moving tensor for alignment
    # TODO: handle arbitrary batch size
    moving_ca_tensor = moving_tensor[common_ca_ind, :]  # [N_common_ca, 3]
    moving_ca_numpy = moving_ca_tensor.detach().cpu().numpy()

    # Initialize for iterative refinement
    inlier_indices = np.arange(len(ref_ca_numpy))
    final_R = np.identity(3)
    final_centroid_moving = None
    final_centroid_ref = None

    for cycle in range(cycles + 1):
        # Get inlier subset
        P_iter = ref_ca_numpy[inlier_indices]
        Q_iter = moving_ca_numpy[inlier_indices]

        try:
            # Use YOUR existing weighted_kabsch_svd function
            final_R = weighted_kabsch_svd(P_iter, Q_iter, weights=None)
        except ValueError:
            # Still compute centroids for the current (failed) iteration
            centroid_Q = np.mean(Q_iter, axis=0)
            centroid_P = np.mean(P_iter, axis=0)
            final_centroid_moving = centroid_Q
            final_centroid_ref = centroid_P
            break

        # Compute centroids for this iteration
        centroid_Q = np.mean(Q_iter, axis=0)
        centroid_P = np.mean(P_iter, axis=0)

        if cycle == cycles:
            # Store final centroids
            final_centroid_moving = centroid_Q
            final_centroid_ref = centroid_P
            break

        # Apply transformation and find new inliers
        t_iter = centroid_P - final_R @ centroid_Q
        transformed_ca = (final_R @ moving_ca_numpy.T).T + t_iter
        distances = np.linalg.norm(ref_ca_numpy - transformed_ca, axis=1)
        new_inlier_indices = np.where(distances < cutoff)[0]

        # Check convergence
        if len(new_inlier_indices) < 3 or np.array_equal(inlier_indices, new_inlier_indices):
            final_centroid_moving = centroid_Q
            final_centroid_ref = centroid_P
            break

        inlier_indices = new_inlier_indices

    # Convert final transformation parameters to torch tensors
    R_torch = torch.tensor(final_R, dtype=moving_tensor.dtype, device=moving_tensor.device)
    centroid_moving_torch = torch.tensor(final_centroid_moving, dtype=moving_tensor.dtype, device=moving_tensor.device)
    centroid_ref_torch = torch.tensor(final_centroid_ref, dtype=moving_tensor.dtype, device=moving_tensor.device)

    # Apply transformation to entire moving_tensor
    # aligned = R @ (moving - centroid_moving) + centroid_ref
    centered = moving_tensor - centroid_moving_torch
    rotated = torch.matmul(centered, R_torch.T)
    aligned_tensor = rotated + centroid_ref_torch

    return aligned_tensor


def weighted_kabsch_svd(P: np.ndarray, Q: np.ndarray, weights: np.ndarray = None) -> np.ndarray:
    """
    Computes the optimal rotation matrix that minimizes the weighted RMSD
    between two sets of corresponding points P and Q using SVD.

    Args:
        P: Reference points, shape (N, 3).
        Q: Points to align to P, shape (N, 3).
        weights: Weights for each point, shape (N,). If None, uniform weights.

    Returns:
        Optimal rotation matrix (3, 3).
    """
    if P.shape[0] < 3:
        raise ValueError("Need at least 3 points for stable alignment")

    # Center the point clouds
    if weights is not None:
        centroid_P = np.average(P, axis=0, weights=weights)
        centroid_Q = np.average(Q, axis=0, weights=weights)
    else:
        centroid_P = np.mean(P, axis=0)
        centroid_Q = np.mean(Q, axis=0)

    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q

    # Compute covariance matrix
    if weights is not None:
        # Weighted covariance
        W = np.diag(weights)
        H = Q_centered.T @ W @ P_centered
    else:
        # Unweighted covariance (standard Kabsch)
        H = Q_centered.T @ P_centered

    # Perform SVD
    try:
        U, _, Vt = np.linalg.svd(H)
    except np.linalg.LinAlgError:
        return np.identity(3)

    # Calculate rotation matrix R
    R = Vt.T @ U.T

    # Handle reflection case (ensures a right-handed coordinate system)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    return R


def kabsch_alignment_common_ca(
    pred_coords: torch.Tensor, ref_ca_coords: torch.Tensor, common_ca_ind: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Perform Kabsch alignment between predicted and true coordinates.

    This function aligns the predicted coordinates to the true coordinates
    using the Kabsch algorithm, which finds the optimal rotation matrix
    that minimizes the RMSD between the two sets of coordinates.

    Args:
        pred_coords: Predicted atomic coordinates (..., N_sample, N_atoms, 3).
        ref_ca_coords: Reference atomic coordinates (N_common_ca, 3).
        common_ca_ind: Common CA indices (N_common_ca,) to mask the predicted coordinates.

    Returns:
        Tuple of (aligned_coords, rotation_matrix, translation_vector):
            - aligned_coords: Aligned predicted coordinates (..., N_sample, N_atoms, 3)
            - rotation_matrix: Optimal rotation matrix (3, 3)
            - translation_vector: Translation vector (3,)
    """


def get_common_ca_ind(pdb1: PDBParser, pdb2: PDBParser):
    """
    A known bug: it can throw some residues out when the two pdbs have ideentical sequences
    for example, "DFGTT" for both, and it will only keep "GTT"
    """  # noqa: E501
    seq1 = pdb1.sequence
    seq2 = pdb2.sequence
    alignment = skbio.alignment.StripedSmithWaterman(seq1)(seq2)  # Align sequence with Smith Waterman Algorithm
    subind_1 = np.arange(alignment.query_begin, alignment.query_end + 1)
    subind_2 = np.arange(alignment.target_begin, alignment.target_end_optimal + 1)
    subsubind_1, subsubind_2 = get_identical_indices(alignment.aligned_query_sequence, alignment.aligned_target_sequence)
    common_seq1 = subind_1[subsubind_1]
    common_seq2 = subind_2[subsubind_2]
    common_ca_ind_1 = [get_pattern_index(pdb1.cra_name, rf".*-{j}-.*-CA$") for j in common_seq1]
    common_ca_ind_2 = [get_pattern_index(pdb2.cra_name, rf".*-{i}-.*-CA$") for i in common_seq2]
    assert (
        np.array([i[-6:] for i in np.array(pdb1.cra_name)[common_ca_ind_1]])
        == np.array([i[-6:] for i in np.array(pdb2.cra_name)[common_ca_ind_2]])
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

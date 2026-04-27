from typing import Any


import numpy as np
from protenix.data.tokenizer import TokenArray
from biotite.structure import AtomArray


def get_pseud_cb_indices_by_residue(token_array: TokenArray, atom_array: AtomArray):
    """
    Return one pseudo CB index per token; if a residue has no CB (e.g., GLY),
    fall back to the token's centre atom. Uses a composite residue key to
    avoid collisions across entities/copies/chains.
    """
    atom_names = atom_array.atom_name
    res_ids = atom_array.res_id

    # Build composite keys per-atom to uniquely identify residues
    label_entity_id = atom_array.get_annotation("label_entity_id")
    copy_id = atom_array.get_annotation("copy_id")
    chain_id = atom_array.get_annotation("chain_id")
    keys = list[tuple](zip(label_entity_id, copy_id, chain_id, res_ids))

    # Group atoms by composite residue key
    # res_groups[k] contains all atom indices of residue k
    # k is a tuple of (label_entity_id, copy_id, chain_id, res_id)
    res_groups: dict[tuple, list[int]] = {}
    for i, k in enumerate(keys):
        if k not in res_groups:
            res_groups[k] = []
        res_groups[k].append(i)

    pseudo_cb_indices = []

    for token in token_array.tokens:
        # Use the token's centre atom to locate its residue key
        centre_idx = token.centre_atom_index
        k = keys[centre_idx]

        res_atom_indices = res_groups[k]
        res_atom_names = atom_names[res_atom_indices]

        # Prefer CB if present; otherwise fall back to centre atom
        cb_positions = np.where(res_atom_names == "CB")[0]

        
        if len(cb_positions) >= 1:
            cb_idx = res_atom_indices[cb_positions[0]]
            pseudo_cb_indices.append(cb_idx)
        else:
            pseudo_cb_indices.append(centre_idx)

    return np.array(pseudo_cb_indices)




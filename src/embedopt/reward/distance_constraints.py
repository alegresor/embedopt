# inherit from base.py
from typing import Any, Optional
import copy

import torch
import numpy as np

from protenix.data.utils import get_atom_mask_by_name
from protenix.utils.logger import get_logger
from protenix.data.tokenizer import TokenArray
from biotite.structure import AtomArray
from biotite.structure.io.pdb import PDBFile
import uuid
import SFC_Torch as sfc
import os
from pathlib import Path


from embedopt.reward.base import Reward
from embedopt.util.energy_minimization import minimize_energy_pdb


logger = get_logger(__name__)


def reward_squared_relu(d, bmin, bmax, k=1.0):
    """
    Differentiable reward that is 0 (flat) inside [bmin, bmax] and smoothly decreases outside.
    Args:
        d: distance tensor, shape [..., N_sample, N_tokens, N_tokens]
        bmin: minimum distance tensor, shape  [N_tokens, N_tokens]
        bmax: maximum distance tensor, shape  [N_tokens, N_tokens]
        k: scaling parameter
    Returns:
        reward tensor, shape [..., N_sample, N_tokens, N_tokens], range (-inf, 0)
    """
    loss_below = torch.nn.functional.relu(k * (bmin - d))  # Range (0, inf)
    loss_above = torch.nn.functional.relu(k * (d - bmax))  # Range (0, inf)
    loss = loss_below**2 + loss_above**2

    return -loss


class DistanceConstraintsReward(Reward):
    def __init__(
        self,
        sequences: list[dict[str, Any]],
        token_array: TokenArray,
        atom_array: AtomArray,
        context: dict,
        distance_atom_type="center",
        output_dir: Optional[str] = None,
        save_pdb_every_n_steps: int = 25,  # Save PDB every N steps
        total_steps: Optional[int] = None,  # Total diffusion steps (for saving last PDB)
        energy_relaxation: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if output_dir is not None:
            self.traj_dir = Path(output_dir)
            self.traj_dir.mkdir(parents=True, exist_ok=True)  # Create the trajectory directory
        else:
            self.traj_dir = None
        self._step_counter = 0  # ← Counter for trajectory files
        self.save_pdb_every_n_steps = save_pdb_every_n_steps  # Save frequency
        self.total_steps = total_steps  # Total diffusion steps
        self.energy_relaxation = energy_relaxation

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.token_array = token_array

        self.atom_array = atom_array

        # distance_atom_type is the type of atom to use for calculating the distance between tokens
        assert distance_atom_type in [
            "center",
            "pseudo_beta",
        ], f"distance_atom_type must be either center or pseudo_beta, but got {distance_atom_type}"

        self.contact, self.contact_atom = DistanceConstraintsReward._generate_constraints(token_array, atom_array, sequences, context, distance_atom_type)

        # initialize a temp pdb file for later saving intermediate structures
        if output_dir is not None:
            self.intermediate_pdb_dir = Path(output_dir)
            self.intermediate_pdb_dir.mkdir(parents=True, exist_ok=True)
            temp_pdb_path = str(self.intermediate_pdb_dir / f"temp_pred_atom_array_{uuid.uuid4()}.pdb")
        else:
            temp_pdb_path = f"temp_pred_atom_array_{uuid.uuid4()}.pdb"
        temp_file = PDBFile()
        temp_file.set_structure(atom_array)
        temp_file.write(temp_pdb_path)
        self.pred_pdb = sfc.PDBParser(temp_pdb_path)
        # remove the temp pdb file
        os.remove(temp_pdb_path)

    def _save_pdb(self, model_coordinates: torch.Tensor, pdb_save_path: Optional[str] = None):
        self.pred_pdb.atom_pos = model_coordinates.detach().cpu().numpy()
        if pdb_save_path is not None:
            self.pred_pdb.savePDB(pdb_save_path)
        elif self.traj_dir is not None:
            self.pred_pdb.savePDB(str(self.traj_dir / f"step_{self._step_counter}.pdb"))

    def compute(self, sample: torch.Tensor, pdb_save_path: Optional[str] = None) -> torch.Tensor:
        is_last_step = self.total_steps is not None and self._step_counter == self.total_steps - 1

        # Periodic saving or last step
        if self.traj_dir is not None and (self._step_counter % self.save_pdb_every_n_steps == 0 or is_last_step):
            self._save_pdb(sample[..., 0, :, :])  # hardcode to save the first sample

        # Explicit saving if path provided
        if pdb_save_path is not None:
            self._save_pdb(sample[..., 0, :, :], pdb_save_path)

        # Energy relaxation on the last diffusion step
        if is_last_step and self.energy_relaxation and self.traj_dir is not None:
            input_pdb = str(self.traj_dir / f"step_{self._step_counter}.pdb")
            output_pdb = str(self.traj_dir / f"step_{self._step_counter}_relaxed.pdb")
            if os.path.exists(input_pdb):
                try:
                    energy, topo_ok = minimize_energy_pdb(input_pdb, output_pdb)
                    logger.info(
                        f"Energy relaxation done: {Path(output_pdb).name}, "
                        f"energy={energy:.4f} RT, topology_match={topo_ok}"
                    )
                except Exception as e:
                    logger.warning(f"Energy relaxation failed for {Path(input_pdb).name}: {e}")

        self._step_counter += 1

        coord = sample
        # coord shape [..., N_sample, N_atom, 3]

        contact_reward = self._compute_contact_reward(coord)
        contact_atom_reward = self._compute_contact_atom_reward(coord)

        # Per-sample total reward (N_sample,)
        result = contact_reward + contact_atom_reward

        # Aggregate across samples
        if self.agg == "mean":
            return result.mean()
        elif self.agg == "sum":
            return result.sum()
        return result

    def _compute_contact_reward(self, coord):
        """
        Compute the contact reward for the sample.
        Args:
            coord: torch.Tensor[..., N_sample, N_atom, 3]
        Returns:
            reward: torch.Tensor[N_sample] — sum of constraint penalties per sample
        """
        n_sample = coord.shape[-3]
        reward = torch.zeros(n_sample, device=self.device)

        for (a1, a2), (min_distance, max_distance) in self.contact.items():
            model_atom_distance = torch.cdist(coord[..., a1, :], coord[..., a2, :])
            reward_i = reward_squared_relu(model_atom_distance, min_distance.to(self.device), max_distance.to(self.device))
            # Sum over constraint dimensions (last two), keep N_sample
            reward += reward_i.sum(dim=(-1, -2))

            _satisfied = min_distance <= model_atom_distance.mean() <= max_distance
            logger.info(
                f"Contact constraint({a1}, {a2}): {min_distance:.2f} <= {model_atom_distance.mean():.2f} <= {max_distance:.2f}, reward: {reward_i.sum().item():.2f}, satisfied: {_satisfied}"  # noqa: E501
            )

        return reward

    def _compute_contact_atom_reward(self, coord):
        """
        Compute the contact atom reward for the sample.
        Args:
            coord: torch.Tensor[..., N_sample, N_atom, 3]
        Returns:
            reward: torch.Tensor[N_sample] — sum of atom constraint penalties per sample
        """
        n_sample = coord.shape[-3]
        reward = torch.zeros(n_sample, device=self.device)

        # compute atom distances
        for (a1, a2), (min_distance, max_distance) in self.contact_atom.items():
            model_atom_distance = torch.cdist(coord[..., a1, :], coord[..., a2, :])
            reward_i = reward_squared_relu(model_atom_distance, min_distance.to(self.device), max_distance.to(self.device))
            # Sum over constraint dimensions (last two), keep N_sample
            reward += reward_i.sum(dim=(-1, -2))
            _satisfied = min_distance <= model_atom_distance.mean() <= max_distance
            logger.info(
                f"Contact atom constraint({a1}, {a2}): {min_distance:.2f} <= {model_atom_distance.mean():.2f} <= {max_distance:.2f}, reward: {reward_i.sum().item():.2f}, satisfied: {_satisfied}"  # noqa: E501
            )

        return reward

    @staticmethod
    def _generate_constraints(
        token_array: TokenArray,
        atom_array: AtomArray,
        sequences: list[dict[str, Any]],
        constraint_context: dict,
        distance_atom_type: str = "center",
    ):
        """
        Generate contact constraints from the constraint parameters.
        Args:
            token_array: TokenArray
            atom_array: AtomArray
            sequences: list[dict[str, Any]]
            constraint_param: dict
            distance_atom_type: str
        Returns:
            contact: dict[(int, int), (float, float)]
                # key: (atom_idx_1, atom_idx_2)
                # value: (min_distance, max_distance)
            contact_atom: dict[(int, int), (float, float)]
                # key: (atom_idx_1, atom_idx_2)
                # value: (min_distance, max_distance)
        """

        contact_inputs = [DistanceConstraintsReward._canonicalize_contact_format(sequences, pair) for pair in constraint_context.get("contact", [])]
        atom_contact_inputs = list(filter(lambda d: d["contact_type"] == "atom_contact", contact_inputs))
        token_contact_inputs = list(filter(lambda d: d["contact_type"] == "token_contact", contact_inputs))

        # generate token-level contact constraints
        contact = {}
        if len(token_contact_inputs) > 0:
            atom_to_token_idx_dict = {}
            for idx, token in enumerate(token_array.tokens):
                for atom_idx in token.atom_indices:
                    atom_to_token_idx_dict[atom_idx] = idx
            for i, pair in enumerate(token_contact_inputs):
                atom_mask1 = get_atom_mask_by_name(
                    atom_array=atom_array,
                    entity_id=pair["id1"][0],
                    copy_id=pair["id1"][1],
                    position=pair["id1"][2],
                    atom_name=pair["id1"][3],
                )
                atom_mask2 = get_atom_mask_by_name(
                    atom_array=atom_array,
                    entity_id=pair["id2"][0],
                    copy_id=pair["id2"][1],
                    position=pair["id2"][2],
                    atom_name=pair["id2"][3],
                )

                def _get_center_atom_idx(_atom_mask):
                    _atom_mask = _atom_mask & atom_array.centre_atom_mask
                    nonzero_atom_idx = np.flatnonzero(_atom_mask)
                    if nonzero_atom_idx.size > 1:
                        logger.warning("Multiple center atoms found for residue, randomly selecting one")
                    # ranomly select one atom from the mask
                    return int(np.random.choice(nonzero_atom_idx))

                # Resolve pseudo-beta atom index (protein CB except GLY -> CA; else fallback to centre atom)
                def _get_pseudo_beta_idx(_atom_mask):
                    nonzero_atom_idx = np.flatnonzero(_atom_mask)
                    assert nonzero_atom_idx.size > 0, "Empty atom mask"
                    # ensure single residue
                    assert np.unique(atom_array.res_id[nonzero_atom_idx]).size == 1, "Mask spans multiple residues"

                    res_name = atom_array.res_name[nonzero_atom_idx][0]
                    mol_type = atom_array.mol_type[nonzero_atom_idx][0]

                    # protein heuristic: 3-letter res name and not ligand
                    is_protein = (len(res_name) == 3) and (mol_type != "ligand")
                    use_center_atom = True
                    if is_protein:
                        target = "CA" if res_name == "GLY" else "CB"
                        local_names = atom_array.atom_name[nonzero_atom_idx]
                        hit = np.where(local_names == target)[0]
                        if hit.size > 0:
                            if hit.size > 1:
                                logger.warning(f"Multiple {target} atoms found for residue {res_name}, randomly selecting one")
                            use_center_atom = False
                            return int(np.random.choice(nonzero_atom_idx[hit]))

                    if use_center_atom:
                        return _get_center_atom_idx(nonzero_atom_idx)

                if distance_atom_type == "center":
                    atom_idx_1 = _get_center_atom_idx(atom_mask1)
                    atom_idx_2 = _get_center_atom_idx(atom_mask2)

                elif distance_atom_type == "pseudo_beta":
                    atom_idx_1 = _get_pseudo_beta_idx(atom_mask1)
                    atom_idx_2 = _get_pseudo_beta_idx(atom_mask2)

                else:
                    raise ValueError(f"Invalid distance_atom_type: {distance_atom_type}")

                # Get distance bounds
                min_d = torch.tensor(pair["min_distance"], dtype=torch.float32)
                max_d = torch.tensor(pair["max_distance"], dtype=torch.float32)

                # Retrive information for logging
                token_id_1 = int(atom_to_token_idx_dict[atom_idx_1])
                token_id_2 = int(atom_to_token_idx_dict[atom_idx_2])

                chain_id_1 = atom_array.chain_id[atom_idx_1]
                chain_id_2 = atom_array.chain_id[atom_idx_2]

                res_name_1 = atom_array.res_name[atom_idx_1]
                res_name_2 = atom_array.res_name[atom_idx_2]

                atom_name_1 = atom_array.atom_name[atom_idx_1]
                atom_name_2 = atom_array.atom_name[atom_idx_2]

                contact[(atom_idx_1, atom_idx_2)] = (min_d, max_d)

                logger.info(
                    f"Contact constraint between token 1(chain_id: {chain_id_1}, token_idx: {token_id_1}, token_name: {res_name_1}, atom_idx: {atom_idx_1}, atom_name: {atom_name_1}) "  # noqa: E501
                    f"and token 2(chain_id: {chain_id_2}, token_idx: {token_id_2}, token_name: {res_name_2}, atom_idx: {atom_idx_2}, atom_name: {atom_name_2}): "  # noqa: E501
                    f"[{pair['min_distance']}, {pair['max_distance']}]"
                )

        # generate atom-level contact constraints
        contact_atom = {}
        if len(atom_contact_inputs) > 0:
            for i, pair in enumerate(atom_contact_inputs):
                # Resolve the exact atom indices for both endpoints
                atom_mask1 = get_atom_mask_by_name(
                    atom_array=atom_array,
                    entity_id=pair["id1"][0],
                    copy_id=pair["id1"][1],
                    position=pair["id1"][2],
                    atom_name=pair["id1"][3],
                )
                atom_mask2 = get_atom_mask_by_name(
                    atom_array=atom_array,
                    entity_id=pair["id2"][0],
                    copy_id=pair["id2"][1],
                    position=pair["id2"][2],
                    atom_name=pair["id2"][3],
                )

                atom_list_1 = np.nonzero(atom_mask1)[0]
                atom_list_2 = np.nonzero(atom_mask2)[0]

                if atom_list_1.size != 1 or atom_list_2.size != 1:
                    logger.info(f"Atom contact {i} not found or ambiguous; skip. sizes=({atom_list_1.size},{atom_list_2.size})")
                    continue

                a1 = int(atom_list_1.item())
                a2 = int(atom_list_2.item())

                # Fill symmetric min/max bounds
                contact_atom[(a1, a2)] = (pair["min_distance"], pair["max_distance"])

        # print the contact atom constraints
        for (a1, a2), (min_distance, max_distance) in contact_atom.items():
            atom_name1 = atom_array.atom_name[a1]
            atom_name2 = atom_array.atom_name[a2]
            logger.info(
                f"Contact atom constraint between atom {a1} {atom_name1} and atom {a2} {atom_name2}: [{min_distance}, {max_distance}]"
            )

        return contact, contact_atom

    @staticmethod
    def _canonicalize_contact_format(
        sequences: list[dict[str, Any]],
        pair: dict[str, Any],
    ) -> dict[str, Any]:
        pair = copy.deepcopy(pair)
        _pair = {}

        for id_num in ["1", "2"]:
            res_info = []
            for key in ["entity", "copy", "position", "atom"]:
                identifier_value = pair.get(f"{key}{id_num}", None)
                if identifier_value is None:
                    assert key == "atom", "contact should have at least 3 identifiers('entity', 'copy', 'position')"
                if key == "atom" and (identifier_value is not None):
                    if isinstance(identifier_value, int):
                        entity_dict = list(sequences[int(pair.get(f"entity{id_num}", None) - 1)].values())[0]
                        assert "atom_map_to_atom_name" in entity_dict
                        identifier_value = entity_dict["atom_map_to_atom_name"][identifier_value]
                res_info.append(identifier_value)
            _pair[f"id{id_num}"] = res_info

        # if hash(tuple(_pair["id1"][:2])) == hash(tuple(_pair["id2"][:2])):
        #    raise ValueError("A contact pair can not be specified on the same chain")

        _pair["min_distance"] = float(pair.get("min_distance", 0))
        _pair["max_distance"] = float(pair["max_distance"])
        print(f"min_distance: {_pair['min_distance']}, max_distance: {_pair['max_distance']}")
        if _pair["max_distance"] < _pair["min_distance"]:
            raise ValueError("max_distance must be greater than min_distance")
        if "atom1" in pair and "atom2" in pair:
            _pair["contact_type"] = "atom_contact"
        else:
            _pair["contact_type"] = "token_contact"
        return _pair

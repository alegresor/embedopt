"""Map-based reward for protein structure prediction.

This module provides the :class:`MapReward` class that computes a reward based on
the L2 loss between a predicted density map and a reference density map.
"""

from __future__ import annotations

from typing import Optional

import torch
import SFC_Torch as sfc
import uuid
from .base import Reward, StateDict
from embedopt.util.structure import get_common_ca_ind, iterative_kabsch_alignment_common_ca
from embedopt.util.energy_minimization import minimize_energy_pdb
from biotite.structure import AtomArray
from biotite.structure.io.pdb import PDBFile
import os
import gemmi
import numpy as np
from pathlib import Path
from protenix.utils.logger import get_logger

logger = get_logger(__name__)


def _set_bfactors(pdb_path: str, bfactor: float) -> None:
    with open(pdb_path) as f:
        lines = f.readlines()
    with open(pdb_path, "w") as f:
        for line in lines:
            if line.startswith(("ATOM", "HETATM")):
                f.write(f"{line[:60]}{bfactor:6.2f}{line[66:]}")
            else:
                f.write(line)


class MapReward(Reward):
    """Reward based on density map similarity to a reference structure.

    This reward computes the L2 loss between a predicted electron density map
    (computed from model coordinates) and a reference density map. The reference
    map can either be provided directly or computed from a reference PDB file.

    The reward is computed as the negative mean squared error between the
    normalized predicted and reference maps, making it suitable for maximization
    during steering.

    Args:
        ref_pdb_path: Path to the reference PDB file.
        pred_atom_array: Predicted AtomArray, from sample2feat.
        ref_pdb_path: Path to the reference PDB file.
        ref_map: Optional pre-computed reference map. If None, will be computed
            from the reference PDB file.
    """

    def __init__(
        self,
        ref_pdb_path: str,
        pred_atom_array: AtomArray,
        ref_map_path: str,  # TODO: currently only support ccp4 map
        resolution: float = 3.0,
        device: Optional[torch.device] = None,
        name: Optional[str] = None,
        agg: Optional[str] = None,
        output_dir: Optional[str] = None,
        total_steps: Optional[int] = None,
        save_pdb_every_n_steps: int = 25,
        energy_relaxation: bool = True,
    ) -> None:
        """Initialize the MapReward with a reference structure."""
        super().__init__(name=name, agg=agg)

        # Store configuration
        self.resolution = resolution
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device

        # Handle output_dir - only create if provided
        if output_dir is not None:
            self.traj_dir = Path(output_dir)
            self.traj_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.traj_dir = None

        self._step_counter = 0
        self._sample_offset = None
        self.total_steps = total_steps
        self.save_pdb_every_n_steps = save_pdb_every_n_steps
        self.energy_relaxation = energy_relaxation

        # Write the predicted atom array to a temporary PDB file, then read it as a PDBParser
        temp_pdb_path = f"temp_pred_atom_array_{uuid.uuid4()}.pdb"
        temp_file = PDBFile()
        temp_file.set_structure(pred_atom_array)
        temp_file.write(temp_pdb_path)
        pred_pdb = sfc.PDBParser(temp_pdb_path)
        # remove the temp pdb file
        os.remove(temp_pdb_path)

        # Read the reference PDB file as a PDBParser
        ref_pdb = sfc.PDBParser(ref_pdb_path)

        # Get the common CA indices between the predicted and reference PDBs
        self.common_ca_ind_1, self.common_ca_ind_2 = get_common_ca_ind(pred_pdb, ref_pdb)

        # Store reference coordinates, in numpy array
        self.ref_ca_numpy = ref_pdb.atom_pos[self.common_ca_ind_2].copy()

        # Set reference map, inherit the grid size and unit cell

        # Handle both .map and .mrc files
        if ref_map_path.endswith(".mrc"):
            import mrcfile

            with mrcfile.open(ref_map_path, mode="r") as mrc:
                # MRC files: data is in (z, y, x) order, transpose to (x, y, z) for consistency with gemmi
                map_array = np.transpose(mrc.data, (2, 1, 0)).copy()
                nx, ny, nz = map_array.shape
                map_spacing = mrc.voxel_size["x"]  # Assuming uniform spacing
                assert map_spacing < resolution / 3.0, "Map spacing must be less than third of the resolution!"
                grid_size = [nx, ny, nz]
                unit_cell = gemmi.UnitCell(
                    mrc.header.cella["x"],
                    mrc.header.cella["y"],
                    mrc.header.cella["z"],
                    90,
                    90,
                    90,  # Assuming orthogonal cell
                )
                # Store map origin: atoms must be shifted into the local [0, cella] box
                # before SFcalculator, since the FFT grid always starts at (0,0,0)
                vs = mrc.voxel_size
                map_origin = np.array(
                    [
                        float(mrc.header.origin["x"]) if float(mrc.header.origin["x"]) != 0 else float(mrc.header.nxstart) * float(vs["x"]),
                        float(mrc.header.origin["y"]) if float(mrc.header.origin["y"]) != 0 else float(mrc.header.nystart) * float(vs["y"]),
                        float(mrc.header.origin["z"]) if float(mrc.header.origin["z"]) != 0 else float(mrc.header.nzstart) * float(vs["z"]),
                    ],
                    dtype=np.float32,
                )
        else:
            # Assume .map file (CCP4 format)
            ref_map = gemmi.read_ccp4_map(ref_map_path)
            assert ref_map.full_cell(), "Reference map must be a full cell map atm!"
            map_spacing = ref_map.grid.spacing[0]
            assert map_spacing < resolution / 3.0, "Map spacing must be less than third of the resolution!"
            grid_size = [ref_map.grid.nu, ref_map.grid.nv, ref_map.grid.nw]
            unit_cell = ref_map.grid.unit_cell
            map_array = ref_map.grid.array
            # .map files with gemmi start at (0,0,0) — no coordinate shift needed
            map_origin = np.zeros(3, dtype=np.float32)

        self.valid_map_mask = (map_array != 0) & (~np.isnan(map_array))
        self.valid_ref_map = torch.from_numpy(map_array[self.valid_map_mask].astype(np.float32)).to(self.device)
        # normalize the valid reference map
        self.valid_ref_map = (self.valid_ref_map - self.valid_ref_map.mean()) / self.valid_ref_map.std()

        # Store origin offset as tensor for coordinate shifting in _model2map
        self.map_origin = torch.tensor(map_origin, dtype=torch.float32, device=self.device)

        # Initialize structure factor calculator
        pred_pdb.set_unitcell(unit_cell)
        pred_pdb.set_spacegroup(ref_pdb.spacegroup)
        pred_pdb.set_biso(50.0 * np.ones(len(pred_pdb.atom_pos)).astype(np.float32))  # Temp, set all Biso to 50
        self.dcp = sfc.SFcalculator(pred_pdb, dmin=resolution, mode="cryoem", device=self.device)
        self.dcp.gridsize = grid_size

    def _model2map(self, model_coordinates: torch.Tensor, pdb_save_path: Optional[str] = None) -> torch.Tensor:
        """Convert model coordinates to a normalized density map.

        Args:
            model_coordinates: Atomic coordinates (N_atoms, 3).
            pdb_save_path: Optional path to save the PDB file if needed

        Returns:
            Normalized density map tensor.
        """
        # Shift coordinates into the local [0, cella] box so the FFT grid (which always
        # starts at origin (0,0,0)) correctly positions the density
        shifted_coords = model_coordinates - self.map_origin

        # Calculate protein structure factors
        Fprotein = self.dcp.calc_fprotein(shifted_coords, Return=True)

        # Save out the PDB file every N steps OR on the last step (periodic)
        # Pass original (unshifted) coordinates so the saved PDB has absolute positions
        is_last_step = (self.total_steps is not None) and (self._step_counter == self.total_steps - 1)
        if self._step_counter % self.save_pdb_every_n_steps == 0 or is_last_step:
            self._save_pdb(pdb_save_path, model_coordinates)

        # Save out the PDB file to explicit path if provided
        if pdb_save_path is not None:
            self._save_pdb(pdb_save_path, model_coordinates)

        # Increment counter after check (so we track every call)
        self._step_counter += 1

        # Convert to reciprocal space grid
        rs_grid = sfc.mask.reciprocal_grid(self.dcp.Hasu_array, Fprotein, self.dcp.gridsize)

        # Fourier transform to real space
        map_grid = torch.real(torch.fft.fftn(rs_grid, dim=(-3, -2, -1)))
        map_grid_valid = map_grid[self.valid_map_mask]

        # Normalize map
        map_grid_norm = (map_grid_valid - map_grid_valid.mean()) / map_grid_valid.std()

        return map_grid_norm

    def compute(self, sample: torch.Tensor, pdb_save_path: Optional[str] = None) -> torch.Tensor:
        """Compute the map-based reward for each sample in the batch.

        Args:
            sample: Atomic coordinates (..., N_sample, N_atoms, 3) tensor
            pdb_save_path: Path to save the PDB file if needed
        Returns:
            Tensor of shape (N_sample,) with negative MSE per sample. Higher is better.
        """
        n_sample = sample.shape[-3]
        rewards = []

        # Determine whether to save PDBs at this step
        is_last_step = (self.total_steps is not None) and (self._step_counter == self.total_steps - 1)
        should_save = self._step_counter % self.save_pdb_every_n_steps == 0 or is_last_step

        for i in range(n_sample):
            # Kabsch alignment for sample i
            sample_aligned_coords = iterative_kabsch_alignment_common_ca(
                sample[..., i, :, :],
                self.ref_ca_numpy,
                self.common_ca_ind_1,
            )

            # Compute map from sample coordinates
            sample_map_normed = self._model2map_no_save(sample_aligned_coords)

            # Save PDB if needed
            if should_save:
                if n_sample == 1 and self._sample_offset is None:
                    # Single sample, no chunking — use plain filename
                    self._save_pdb(pdb_save_path, sample_aligned_coords)
                else:
                    # Multiple samples or chunked — use sample-indexed filename
                    sample_idx = (self._sample_offset or 0) + i
                    self._save_pdb_with_sample_idx(sample_aligned_coords, sample_idx)

            # Compute L2 loss (negative for reward maximization)
            mse_loss = torch.mean((sample_map_normed - self.valid_ref_map) ** 2)
            rewards.append(-mse_loss)

        # Energy relaxation on the last diffusion step
        if is_last_step and self.energy_relaxation and self.traj_dir is not None:
            for i in range(n_sample):
                if n_sample == 1 and self._sample_offset is None:
                    input_pdb = str(self.traj_dir / f"step_{self._step_counter}.pdb")
                    output_pdb = str(self.traj_dir / f"step_{self._step_counter}_relaxed.pdb")
                else:
                    sample_idx = (self._sample_offset or 0) + i
                    input_pdb = str(self.traj_dir / f"step_{self._step_counter}_sample_{sample_idx}.pdb")
                    output_pdb = str(self.traj_dir / f"step_{self._step_counter}_sample_{sample_idx}_relaxed.pdb")

                if os.path.exists(input_pdb):
                    try:
                        energy, topo_ok = minimize_energy_pdb(input_pdb, output_pdb)
                        logger.info(
                            f"Energy relaxation done: {Path(output_pdb).name}, "
                            f"energy={energy:.4f} RT, topology_match={topo_ok}"
                        )
                        # Set B-factors to 50 to match the forward model used during
                        # synthetic map generation and reward calculation.
                        if os.path.exists(output_pdb):
                            _set_bfactors(output_pdb, 50.0)
                    except Exception as e:
                        logger.warning(f"Energy relaxation failed for {Path(input_pdb).name}: {e}")

        # Increment step counter once per diffusion step
        self._step_counter += 1

        result = torch.stack(rewards)
        if self.agg == "mean":
            return result.mean()
        elif self.agg == "sum":
            return result.sum()
        return result

    def _model2map_no_save(self, model_coordinates: torch.Tensor) -> torch.Tensor:
        """Convert model coordinates to a normalized density map (no PDB saving).

        Args:
            model_coordinates: Atomic coordinates (N_atoms, 3), already aligned.

        Returns:
            Normalized density map tensor.
        """
        shifted_coords = model_coordinates - self.map_origin
        Fprotein = self.dcp.calc_fprotein(shifted_coords, Return=True)
        rs_grid = sfc.mask.reciprocal_grid(self.dcp.Hasu_array, Fprotein, self.dcp.gridsize)
        map_grid = torch.real(torch.fft.fftn(rs_grid, dim=(-3, -2, -1)))
        map_grid_valid = map_grid[self.valid_map_mask]
        map_grid_norm = (map_grid_valid - map_grid_valid.mean()) / map_grid_valid.std()
        return map_grid_norm

    def _save_pdb_with_sample_idx(self, coordinates: torch.Tensor, sample_idx: int):
        """Save PDB with sample index in filename."""
        if self.traj_dir is not None:
            original_pos = self.dcp.atom_pos_orth.clone()
            self.dcp.atom_pos_orth = coordinates.detach().to(self.device)
            self.dcp.savePDB(str(self.traj_dir / f"step_{self._step_counter}_sample_{sample_idx}.pdb"))
            self.dcp.atom_pos_orth = original_pos

    def _save_pdb(self, pdb_save_path: Optional[str] = None, coordinates: Optional[torch.Tensor] = None):
        """
        Save the current model stored in the SFCalculator.

        Args:
            pdb_save_path: Optional explicit path. If None, uses default trajectory path.
            coordinates: Optional unshifted atom coordinates to save. If provided, temporarily
                         restores them in dcp so the saved PDB has absolute (real-space) positions.
        """
        if coordinates is not None:
            # Temporarily set unshifted absolute coordinates so savePDB writes real-space positions
            original_pos = self.dcp.atom_pos_orth.clone()
            self.dcp.atom_pos_orth = coordinates.detach().to(self.device)
        if pdb_save_path is not None:
            self.dcp.savePDB(pdb_save_path)
        elif self.traj_dir is not None:
            self.dcp.savePDB(str(self.traj_dir / f"step_{self._step_counter}.pdb"))

        if coordinates is not None:
            self.dcp.atom_pos_orth = original_pos

    def state_dict(self) -> StateDict:
        """Serialize the reward's mutable state.

        Returns:
            Dictionary containing the reference map and coordinates.
        """
        return {
            "valid_ref_map": self.valid_ref_map.cpu(),
            "valid_map_mask": self.valid_map_mask.cpu() if isinstance(self.valid_map_mask, torch.Tensor) else self.valid_map_mask,
            "ref_ca_numpy": self.ref_ca_numpy,
            "resolution": self.resolution,
            "common_ca_ind_1": self.common_ca_ind_1,
            "common_ca_ind_2": self.common_ca_ind_2,
        }

    def load_state_dict(self, state_dict: StateDict) -> None:
        """Restore the reward's state from state_dict.

        Args:
            state_dict: Dictionary containing saved state.
        """
        if "ref_map" in state_dict:
            self.ref_map = state_dict["ref_map"].to(self.device)
        if "ref_ca_numpy" in state_dict:
            self.ref_ca_numpy = state_dict["ref_ca_numpy"].to(self.device)
        if "resolution" in state_dict:
            self.resolution = state_dict["resolution"]

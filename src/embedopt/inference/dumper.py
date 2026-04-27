"""
Inference results dumper
"""

from __future__ import annotations

from runner.dumper import DataDumper
from pathlib import Path
import shutil
import json
import dataclasses
from collections.abc import Mapping, Sequence

from protenix.utils.logger import get_logger

logger = get_logger(__name__)



class EmbedOptDataDumper(DataDumper):
    """EmbedOpt dumper that extends Protenix DataDumper with additional functionality."""

    def __init__(
        self,
        base_dir,
        need_atom_confidence: bool = False,
        sorted_by_ranking_score: bool = True,
    ):
        """
        Initialize the EmbedOpt dumper.

        Args:
            base_dir: Base directory for output files
            need_atom_confidence: Whether to include atom-level confidence
            sorted_by_ranking_score: Whether to sort by ranking score
            reward_func: Reward function used for steering (optional)
            reward_scale: Scale factor for reward steering (optional)
        """
        # Initialize parent class
        super().__init__(
            base_dir=base_dir,
            need_atom_confidence=need_atom_confidence,
            sorted_by_ranking_score=sorted_by_ranking_score,
        )

    def dump(
        self,
        dataset_name: str,
        pdb_id: str,
        seed: int,
        pred_dict: dict,
        atom_array,
        entity_poly_type: dict[str, str],
        json_dict: dict | None = None,
        configs: object | None = None,
        reward_history: list | None = None,
    ):
        """
        Dump predictions with EmbedOpt-specific enhancements.

        Args:
            dataset_name: Name of the dataset
            pdb_id: PDB ID of the sample
            seed: Random seed used
            pred_dict: Prediction dictionary
            atom_array: AtomArray object
            entity_poly_type: Entity poly type information
        """
        # Call parent dump method to handle standard output
        dump_dir = super().dump(
            dataset_name=dataset_name,
            pdb_id=pdb_id,
            seed=seed,
            pred_dict=pred_dict,
            atom_array=atom_array,
            entity_poly_type=entity_poly_type,
        )

        # Add EmbedOpt-specific output
        self._dump_embedopt_metadata(pdb_id, seed, pred_dict)

        try:
            # Save input JSON and resolved configs if provided

            out_dir = Path(self.base_dir) / dataset_name / pdb_id / f"seed_{seed}"
            out_dir.mkdir(parents=True, exist_ok=True)

            if json_dict is not None:
                with open(out_dir / "input.json", "w") as f:
                    json.dump(json_dict, f, indent=2)
                logger.info(f"Input JSON saved to {out_dir / 'input.json'}")

            if configs is not None:
                def to_plain(obj):
                    # Explicit API first
                    if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
                        return to_plain(obj.to_dict())
                    # Dataclass support
                    if dataclasses.is_dataclass(obj):
                        return to_plain(dataclasses.asdict(obj))
                    # Mappings/dicts
                    if isinstance(obj, Mapping):
                        return {k: to_plain(v) for k, v in obj.items()}
                    # Namespace/objects
                    if hasattr(obj, "__dict__") and not isinstance(obj, type):
                        return {k: to_plain(v) for k, v in vars(obj).items() if not k.startswith("_")}
                    # Sequences (but not strings/bytes)
                    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
                        return [to_plain(v) for v in obj]
                    # Try JSON
                    try:
                        json.dumps(obj)
                        return obj
                    except Exception:
                        return str(obj)

                with open(out_dir / "config.json", "w") as f:
                    json.dump(to_plain(configs), f, indent=2)
                logger.info(f"Config saved to {out_dir / 'config.json'}")

            # Save reward history
            if reward_history is not None:
                reward_log_path = out_dir / "reward_history.json"
                with open(reward_log_path, "w") as f:
                    json.dump(reward_history, f, indent=2)
                logger.info(f"Reward history saved to {reward_log_path}")

        except Exception:
            pass

        # Return dump directory for callers if needed
        return dump_dir

    def _dump_embedopt_metadata(self, pdb_id: str, seed: int, pred_dict: dict):
        """
        Dump EmbedOpt-specific metadata and analysis.

        Args:
            pdb_id: PDB ID of the sample
            seed: Random seed used
            pred_dict: Prediction dictionary
        """
        # Create EmbedOpt metadata
        pass

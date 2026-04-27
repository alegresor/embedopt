import logging
import time
import traceback
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
from protenix.utils.torch_utils import collate_fn_identity, dict_to_tensor


from biotite.structure import AtomArray

from protenix.data.msa.msa_featurizer import InferenceMSAFeaturizer
from protenix.data.utils import data_type_transform, make_dummy_feature
from protenix.data.inference.infer_dataloader import InferenceDataset
from protenix.data.inference.json_to_feature import SampleDictToFeatures
from protenix.utils.distributed import DIST_WRAPPER


logger = logging.getLogger(__name__)


def get_inference_with_reward_dataloader(configs: Any) -> DataLoader:
    """
    Creates and returns a DataLoader for inference using the InferenceWithRewardDataset.

    Args:
        configs: A configuration object containing the necessary parameters for the DataLoader.

    Returns:
        A DataLoader object configured for inference.
    """
    inference_dataset = InferenceWithRewardDataset(
        configs=configs,
    )
    sampler = DistributedSampler(
        dataset=inference_dataset,
        num_replicas=DIST_WRAPPER.world_size,
        rank=DIST_WRAPPER.rank,
        shuffle=False,
    )
    dataloader = DataLoader(
        dataset=inference_dataset,
        batch_size=1,
        sampler=sampler,
        collate_fn=collate_fn_identity,
        num_workers=configs.num_workers,
    )
    return dataloader


class InferenceWithRewardDataset(InferenceDataset):
    def process_one(
        self,
        single_sample_dict: Mapping[str, Any],
    ) -> tuple[dict[str, torch.Tensor], AtomArray, dict[str, float]]:
        """
        Processes a single sample from the input JSON to generate features and statistics.

        Args:
            single_sample_dict: A dictionary containing the sample data.

        Returns:
            A tuple containing:
                - A dictionary of features.
                - An AtomArray object.
                - A dictionary of time tracking statistics.
        """
        # general features
        t0 = time.time()
        sample2feat = SampleDictToFeatures(
            single_sample_dict,
        )
        features_dict, atom_array, token_array = sample2feat.get_feature_dict()
        features_dict["distogram_rep_atom_mask"] = torch.Tensor(atom_array.distogram_rep_atom_mask).long()
        entity_poly_type = sample2feat.entity_poly_type
        t1 = time.time()

        # Msa features
        msa_features = (
            InferenceMSAFeaturizer.make_msa_feature(
                bioassembly=single_sample_dict["sequences"],
                atom_array=atom_array,
            )
            if self.use_msa
            else {}
        )

        # MSA depth subsampling (for MSA depth ablation)
        msa_depth_fraction = single_sample_dict.get("msa_depth_fraction", 1.0)
        msa_depth_seed = single_sample_dict.get("msa_depth_seed", 0)
        if msa_depth_fraction < 1.0 and len(msa_features) > 0:
            n_msa = msa_features["msa"].shape[0]
            n_keep = max(1, int(n_msa * msa_depth_fraction))
            # Row 0 = query sequence (always kept). Randomly subsample from rows 1..N-1.
            rng = np.random.RandomState(msa_depth_seed)
            n_other = min(n_keep - 1, n_msa - 1)
            other_indices = rng.choice(n_msa - 1, size=n_other, replace=False) + 1
            keep_indices = np.concatenate([[0], np.sort(other_indices)])
            for key in ["msa", "has_deletion", "deletion_value"]:
                if key in msa_features:
                    msa_features[key] = msa_features[key][keep_indices]
            logger.warning(f"Warning: subsampling MSA {n_msa} -> {n_keep} rows (fraction={msa_depth_fraction}, seed={msa_depth_seed})")

        # Esm features
        if self.esm_enable:
            x_esm = self.esm_featurizer(
                token_array=token_array,
                atom_array=atom_array,
                bioassembly_dict=single_sample_dict,
                inference_mode=True,
            )
            features_dict["esm_token_embedding"] = x_esm

        # Make dummy features for not implemented features
        dummy_feats = ["template"]
        if len(msa_features) == 0:
            dummy_feats.append("msa")
        else:
            msa_features = dict_to_tensor(msa_features)
            features_dict.update(msa_features)
        features_dict = make_dummy_feature(
            features_dict=features_dict,
            dummy_feats=dummy_feats,
        )

        # Transform to right data type
        feat = data_type_transform(feat_or_label_dict=features_dict)

        t2 = time.time()

        data = {}
        data["input_feature_dict"] = feat

        # Add dimension related items
        N_token = feat["token_index"].shape[0]
        N_atom = feat["atom_to_token_idx"].shape[0]
        N_msa = feat["msa"].shape[0]

        stats = {}
        for mol_type in ["ligand", "protein", "dna", "rna"]:
            mol_type_mask = feat[f"is_{mol_type}"].bool()
            stats[f"{mol_type}/atom"] = int(mol_type_mask.sum(dim=-1).item())
            stats[f"{mol_type}/token"] = len(torch.unique(feat["atom_to_token_idx"][mol_type_mask]))

        N_asym = len(torch.unique(data["input_feature_dict"]["asym_id"]))
        data.update(
            {
                "N_asym": torch.tensor([N_asym]),
                "N_token": torch.tensor([N_token]),
                "N_atom": torch.tensor([N_atom]),
                "N_msa": torch.tensor([N_msa]),
            }
        )

        def formatted_key(key):
            type_, unit = key.split("/")
            if type_ == "protein":
                type_ = "prot"
            elif type_ == "ligand":
                type_ = "lig"
            else:
                pass
            return f"N_{type_}_{unit}"

        data.update(
            {
                formatted_key(k): torch.tensor([stats[k]])
                for k in [
                    "protein/atom",
                    "ligand/atom",
                    "dna/atom",
                    "rna/atom",
                    "protein/token",
                    "ligand/token",
                    "dna/token",
                    "rna/token",
                ]
            }
        )
        data.update({"entity_poly_type": entity_poly_type})
        t3 = time.time()
        time_tracker = {
            "crop": t1 - t0,
            "featurizer": t2 - t1,
            "added_feature": t3 - t2,
        }

        reward_configs = single_sample_dict.get("reward", {})
        reward_type = reward_configs.get("type", None)
        reward_dict = {}

        if reward_type == "constraint":
            reward_dict["sequences"] = single_sample_dict["sequences"]
            reward_dict.update(reward_configs.get("params", {}))
        elif reward_type == "map":
            reward_context = reward_configs.get("params", {})
            reward_dict.update(reward_context)  # Merge, preserving "type"
        elif reward_type is None:
            # No reward specified - allowed for unconditional algorithm
            pass
        else:
            raise ValueError(f"Unsupported reward type: {reward_type}")

        # Add token_array and atom_array to reward_dict (even if empty)
        reward_dict["token_array"] = token_array
        reward_dict["atom_array"] = atom_array

        return data, reward_dict, atom_array, time_tracker

    def __getitem__(self, index: int) -> tuple[dict[str, torch.Tensor], AtomArray, str]:
        try:
            single_sample_dict = self.inputs[index]
            sample_name = single_sample_dict["name"]
            logger.info(f"Featurizing {sample_name}...")

            data, reward_dict, atom_array, _ = self.process_one(single_sample_dict=single_sample_dict)
            error_message = ""
        except Exception as e:
            data, reward_dict, atom_array = {}, {}, None
            error_message = f"{e}:\n{traceback.format_exc()}"
        data["sample_name"] = single_sample_dict["name"]
        data["sample_index"] = index
        return data, reward_dict, atom_array, error_message

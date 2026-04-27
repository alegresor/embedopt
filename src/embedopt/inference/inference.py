"""
EmbedOpt Inference Runner
"""

import traceback
from os.path import join as opjoin
from typing import Any
import json

import torch
import logging
from pathlib import Path


from runner.inference import InferenceRunner, download_inference_cache, update_inference_configs
from protenix.utils.seed import seed_everything
from protenix.utils.distributed import DIST_WRAPPER
from protenix.utils.logger import get_logger

from embedopt.data.infer_data_reward_pipeline import get_inference_with_reward_dataloader
from embedopt.inference.dumper import EmbedOptDataDumper
from embedopt.sampler import DiffusionPosteriorSampler, ProtenixBase, EmbedOptSampler
from embedopt.util.logging import _TqdmHandler

logger = get_logger(__name__)


class EmbedOptInferenceRunner(InferenceRunner):
    """Inference runner that works with the base sampler as well as steering samplers."""

    def __init__(self, configs: Any) -> None:
        # read algorithm which is used to initialize the protenix model
        # Default to "base" if algorithm is not specified
        json_data = json.load(open(configs.input_json_path))[0]
        self.algorithm = json_data.get("algorithm", {}).get("type", "base")
        super().__init__(configs)

    def init_model(self):
        """Instantiate the sampler matching the configured algorithm."""
        valid_options = ["base", "dps", "embedopt"]
        if self.algorithm == "dps":
            self.model = DiffusionPosteriorSampler(configs=self.configs).to(self.device)
        elif self.algorithm == "base":
            self.model = ProtenixBase(configs=self.configs).to(self.device)
        elif self.algorithm == "embedopt":
            self.model = EmbedOptSampler(configs=self.configs).to(self.device)
        else:
            raise ValueError(f"Invalid algorithm: {self.algorithm}, please specify algorithm in {valid_options}")

    def init_dumper(self, need_atom_confidence: bool = False, sorted_by_ranking_score: bool = True):
        """Override to use EmbedOptDataDumper instead of DataDumper."""
        self.dumper = EmbedOptDataDumper(
            base_dir=self.dump_dir, need_atom_confidence=need_atom_confidence, sorted_by_ranking_score=sorted_by_ranking_score
        )
        logger.info("EmbedOptDataDumper initialized")


def infer_predict_with_reward(runner: EmbedOptInferenceRunner, configs: Any) -> None:
    # Route logging through tqdm once to avoid breaking progress bars
    _root = logging.getLogger()
    if not any(isinstance(h, _TqdmHandler) for h in _root.handlers):
        _root.addHandler(_TqdmHandler())
    # Data
    logger.info(f"Loading data from\n{configs.input_json_path}")
    try:
        dataloader = get_inference_with_reward_dataloader(configs=configs)
    except Exception as e:
        error_message = f"{e}:\n{traceback.format_exc()}"
        logger.info(error_message)
        with open(opjoin(runner.error_dir, "error.txt"), "a") as f:
            f.write(error_message)
        return

    _json_input = json.load(open(configs.input_json_path))

    import gc
    num_data = len(dataloader.dataset)
    for seed in configs.seeds:
        gc.collect()
        torch.cuda.empty_cache()
        seed_everything(seed=seed, deterministic=configs.deterministic)
        for batch_index, batch in enumerate(dataloader):
            try:
                data, reward_dict, atom_array, data_error_message = batch[0]
                sample_name = data["sample_name"]

                # read algorithm from config, default to "base" if not specified
                algorithm = _json_input[batch_index].get("algorithm", {}).get("type", "base")
                if algorithm != runner.algorithm:
                    runner.algorithm = algorithm
                    logger.info(f"Algorithm changed to {algorithm}, reinitialized model and checkpoint")
                    runner.init_model()
                    runner.load_checkpoint()
                algorithm_config = _json_input[batch_index].get("algorithm", {}).get("params", {})

                # Add output_dir to algorithm config for embedopt sampler
                # This allows saving optimized embeddings during sampling
                if algorithm == "embedopt":
                    sample_output_dir = Path(configs.dump_dir) / sample_name / f"seed_{seed}"
                    algorithm_config["output_dir"] = str(sample_output_dir)

                runner.model.update_algorithm_config(algorithm_config)

                if len(data_error_message) > 0:
                    logger.info(data_error_message)
                    with open(opjoin(runner.error_dir, f"{sample_name}.txt"), "a") as f:
                        f.write(data_error_message)
                    continue

                logger.info(
                    (
                        f"[Rank {DIST_WRAPPER.rank} ({data['sample_index'] + 1}/{num_data})] {sample_name}: "
                        f"N_asym {data['N_asym'].item()}, N_token {data['N_token'].item()}, "
                        f"N_atom {data['N_atom'].item()}, N_msa {data['N_msa'].item()}"
                    )
                )
                new_configs = update_inference_configs(configs, data["N_token"].item())
                runner.update_model_configs(new_configs)

                # Create reward (optional for base algorithm)
                reward_type = _json_input[batch_index].get("reward", {}).get("type", None)
                reward = None

                # Build trajectory output path
                traj_dir = Path(configs.dump_dir) / sample_name / f"seed_{seed}" / "trajectory"
                traj_dir.mkdir(parents=True, exist_ok=True)
                runner.model.set_output_dir(traj_dir)

                # Aggregation mode determined by algorithm
                if algorithm == "dps":
                    reward_agg = "sum"
                elif algorithm in ("embedopt", "base"):
                    reward_agg = "mean"
                else:
                    raise ValueError(f"Unknown algorithm '{algorithm}' — cannot determine reward aggregation mode")

                if reward_type == "constraint":
                    from embedopt.reward.distance_constraints import DistanceConstraintsReward

                    reward = DistanceConstraintsReward(
                        **reward_dict,
                        agg=reward_agg,
                        output_dir=str(traj_dir),
                        total_steps=configs.sample_diffusion.N_step,
                        save_pdb_every_n_steps=reward_dict.get("save_pdb_every_n_steps", 25),
                    )

                elif reward_type == "map":
                    from embedopt.reward.map import MapReward

                    reward = MapReward(
                        ref_pdb_path=reward_dict["ref_pdb_path"],
                        pred_atom_array=atom_array,  # From dataloader
                        ref_map_path=reward_dict["ref_map_path"],
                        resolution=reward_dict.get("resolution"),
                        agg=reward_agg,
                        output_dir=str(traj_dir),
                        total_steps=configs.sample_diffusion.N_step,
                        save_pdb_every_n_steps=reward_dict.get("save_pdb_every_n_steps", 25),
                        energy_relaxation=reward_dict.get("energy_relaxation", True),
                    )
                elif reward_type is None:
                    # No reward specified - only allowed for base algorithm
                    if algorithm != "base":
                        raise ValueError(
                            f"Reward must be specified for algorithm '{algorithm}'. "
                            "Only 'base' algorithm can run without a reward."
                        )
                    logger.info(f"Running {sample_name} with base algorithm (no reward)")
                else:
                    raise ValueError(f"Unsupported reward type: {reward_type}")

                # Update reward in model (if reward is defined)
                if reward is not None:
                    runner.model.set_reward(reward)

                prediction = runner.predict(data)

                # Save reward history
                reward_history = None

                if reward is not None:
                    if hasattr(runner.model, "last_reward_history"):
                        reward_history = runner.model.last_reward_history

                # Save profiling info if available
                if hasattr(runner.model, "last_profile"):
                    profile = runner.model.last_profile
                    logger.info(
                        f"[Profile] {sample_name} seed={seed}: "
                        f"{profile['elapsed_seconds']:.1f}s, "
                        f"peak GPU: {profile['peak_gpu_memory_gb']:.2f} GB"
                    )
                    # Save to output dir (same level as reward_history.json)
                    import json as _json

                    _profile_dir = Path(configs.dump_dir) / sample_name / f"seed_{seed}"
                    _profile_dir.mkdir(parents=True, exist_ok=True)
                    with open(_profile_dir / "profile.json", "w") as _f:
                        _json.dump(profile, _f, indent=2)

                runner.dumper.dump(
                    dataset_name="",
                    pdb_id=sample_name,
                    seed=seed,
                    pred_dict=prediction,
                    atom_array=atom_array,
                    entity_poly_type=data["entity_poly_type"],
                    json_dict=_json_input[batch_index],
                    configs=configs,
                    reward_history=reward_history,
                )

                logger.info(f"[Rank {DIST_WRAPPER.rank}] {data['sample_name']} succeeded.\n" f"Results saved to {configs.dump_dir}")
                torch.cuda.empty_cache()
            except Exception as e:
                error_message = f"[Rank {DIST_WRAPPER.rank}]{data['sample_name']} {e}:\n{traceback.format_exc()}"
                logger.info(error_message)
                # Save error info
                with open(opjoin(runner.error_dir, f"{sample_name}.txt"), "a") as f:
                    f.write(error_message)
                if hasattr(torch.cuda, "empty_cache"):
                    torch.cuda.empty_cache()


def main() -> None:
    LOG_FORMAT = "%(asctime)s,%(msecs)-3d %(levelname)-8s [%(filename)s:%(lineno)s %(funcName)s] %(message)s"
    logging.basicConfig(
        format=LOG_FORMAT,
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filemode="w",
    )

    logger.info("Starting EmbedOpt Inference")

    from collections.abc import Mapping
    from protenix.config.config import parse_configs, parse_sys_args
    from configs import configs_base, configs_data, configs_inference
    from configs.configs_model_type import model_configs

    def _deep_update(d, u):
        for k, v in u.items():
            if isinstance(v, Mapping) and k in d and isinstance(d[k], Mapping):
                _deep_update(d[k], v)
            else:
                d[k] = v
        return d

    arg_str = parse_sys_args()

    def _base_configs():
        base = {
            **configs_base.configs,
            **{"data": configs_data.data_configs},
            **configs_inference.inference_configs,
        }
        base["sample_diffusion"]["N_sample"] = 1
        return base

    # Pass 1: parse to get model_name from command-line args
    configs = parse_configs(configs=_base_configs(), arg_str=arg_str, fill_required_with_null=True)
    model_name = configs.model_name

    # Pass 2: merge model-specific arch configs (e.g. template_embedder.n_blocks=2 for v1.0.0),
    # then re-parse so command-line args still take priority
    base = _base_configs()
    if model_name in model_configs:
        _deep_update(base, model_configs[model_name])
    configs = parse_configs(configs=base, arg_str=arg_str, fill_required_with_null=True)

    logger.info(f"Model: {model_name}, N_step={configs.sample_diffusion.N_step}, N_cycle={configs.model.N_cycle}")

    # Download CCD and related cache files if missing (components.cif, cluster file, etc.)
    download_inference_cache(configs)

    # Initialize inference runner (after setting configs)
    runner = EmbedOptInferenceRunner(configs=configs)

    logger.info("EmbedOpt Inference Runner initialized")

    try:
        infer_predict_with_reward(runner, configs)
        logger.info("Inference completed successfully!")

        # Print results summary
        print("\n" + "=" * 50)
        print("EMBEDOPT INFERENCE COMPLETED")
        print("=" * 50)
        print(f"Results saved to: {configs.dump_dir}")
        print(f"Structure files: {configs.dump_dir}/*/seed_*/predictions/*.cif")
        print(f"Confidence files: {configs.dump_dir}/*/seed_*/predictions/*_summary_confidence.json")
        print("=" * 50)

    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise


if __name__ == "__main__":
    main()

from __future__ import annotations

from typing import Any, Callable, Optional
from pathlib import Path

import torch
from tqdm import tqdm

from protenix.model.protenix import Protenix
from protenix.utils.torch_utils import autocasting_disable_decorator
from protenix.model.utils import centre_random_augmentation
from protenix.utils.logger import get_logger

from embedopt.reward import Reward

Tensor = torch.Tensor
ScalarSchedule = Callable[[int, int], float]


logger = get_logger(__name__)


class ProtenixBase(Protenix):
    """Base sampler — Protenix diffusion sampler (AF3 Algorithm 18).

    Used directly, this sampler runs vanilla Protenix diffusion sampling. A reward can still
    be attached for tracking purposes (logged per step), but it does not influence
    the sampling trajectory.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.allowed_keys = {
            "override_embeddings_path": (str, type(None)),
        }
        self._reward = None  # Initialize reward as None (optional)

    def set_reward(self, reward: Reward):
        self._reward = reward

    @property
    def reward(self):
        return self._reward

    def update_algorithm_config(self, algorithm_config: dict):
        """Update algorithm configuration with validation."""
        # Only allow specific keys
        for key, value in algorithm_config.items():
            if key not in self.allowed_keys:
                raise ValueError(f"Invalid algorithm config key: {key}. Allowed keys: {self.allowed_keys}")

            # Type check
            expected_types = self.allowed_keys[key]
            if not isinstance(value, expected_types):
                raise TypeError(f"{key} must be one of {expected_types}, got {type(value)}")

            setattr(self, key, value)

    def sample_diffusion(self, **kwargs) -> torch.Tensor:
        """
        Samples diffusion process based on the provided configurations.

        Returns:
            torch.Tensor: The result of the diffusion sampling process.
        """
        _configs = {
            key: self.configs.sample_diffusion.get(key)
            for key in [
                "gamma0",
                "gamma_min",
                "noise_scale_lambda",
                "step_scale_eta",
            ]
        }
        _configs.update(
            {
                "attn_chunk_size": (self.configs.infer_setting.chunk_size if not self.training else None),
                "diffusion_chunk_size": (self.configs.infer_setting.sample_diffusion_chunk_size if not self.training else None),
            }
        )
        logger.info("Calling sample_diffusion from the base model!")
        return autocasting_disable_decorator(self.configs.skip_amp.sample_diffusion)(self._sample_diffusion)(**_configs, **kwargs)

    def _sample_diffusion(
        self,
        denoise_net: Callable,
        input_feature_dict: dict[str, Any],
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        noise_schedule: torch.Tensor,
        N_sample: int = 1,
        gamma0: float = 0.8,
        gamma_min: float = 1.0,
        noise_scale_lambda: float = 1.003,
        step_scale_eta: float = 1.5,
        diffusion_chunk_size: Optional[int] = None,
        inplace_safe: bool = False,
        attn_chunk_size: Optional[int] = None,
        **extra_kwargs,
    ) -> torch.Tensor:
        """Implements Algorithm 18 in AF3.
        It performances denoising steps from time 0 to time T.
        The time steps (=noise levels) are given by noise_schedule.

        Args:
            denoise_net (Callable): the network that performs the denoising step.
            input_feature_dict (dict[str, Any]): input meta feature dict
            s_inputs (torch.Tensor): single embedding from InputFeatureEmbedder
                [..., N_tokens, c_s_inputs]
            s_trunk (torch.Tensor): single feature embedding from PairFormer (Alg17)
                [..., N_tokens, c_s]
            z_trunk (torch.Tensor): pair feature embedding from PairFormer (Alg17)
                [..., N_tokens, N_tokens, c_z]
            noise_schedule (torch.Tensor): noise-level schedule (which is also the time steps) since sigma=t.
                [N_iterations]
            N_sample (int): number of generated samples
            gamma0 (float): params in Alg.18.
            gamma_min (float): params in Alg.18.
            noise_scale_lambda (float): params in Alg.18.
            step_scale_eta (float): params in Alg.18.
            diffusion_chunk_size (Optional[int]): Chunk size for diffusion operation. Defaults to None.
            inplace_safe (bool): Whether to use inplace operations safely. Defaults to False.
            attn_chunk_size (Optional[int]): Chunk size for attention operation. Defaults to None.

        Returns:
            torch.Tensor: the denoised coordinates of x in inference stage
                [..., N_sample, N_atom, 3]
        """
        N_atom = input_feature_dict["atom_to_token_idx"].size(-1)
        batch_shape = s_inputs.shape[:-2]
        device = s_inputs.device
        dtype = s_inputs.dtype

        # Load pre-optimized embeddings if specified
        override_embeddings_path = getattr(self, "override_embeddings_path", None)
        if override_embeddings_path is not None:
            override_path = Path(override_embeddings_path)
            if not override_path.exists():
                raise FileNotFoundError(f"Override embeddings path not found: {override_path}")

            logger.info(f"Loading pre-optimized embeddings from {override_path}")
            checkpoint = torch.load(override_path, map_location=device)
            s_trunk = checkpoint["s_trunk"].to(device, dtype)
            z_trunk = checkpoint["z_trunk"].to(device, dtype)

            if "step" in checkpoint:
                logger.info(f"Loaded embeddings from step {checkpoint['step']}/{checkpoint.get('total_steps', '?')}")
            logger.info("Pre-optimized embeddings loaded successfully")

        reward = self.reward  # Can be None for unconditional sampling

        reward_history = []

        # Profiling: start timer and reset GPU memory stats
        import time as _time

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        _profile_start = _time.time()

        def _chunk_sample_diffusion(chunk_n_sample, inplace_safe):
            # init noise
            # [..., N_sample, N_atom, 3]
            x_l = noise_schedule[0] * torch.randn(
                size=(*batch_shape, chunk_n_sample, N_atom, 3), device=device, dtype=dtype
            )  # NOTE: set seed in distributed training

            # Create progress bar
            total_steps = len(noise_schedule) - 1
            pbar = tqdm(
                enumerate(zip(noise_schedule[:-1], noise_schedule[1:])),
                total=total_steps,
                desc="Diffusion Sampling",
                ncols=170,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
            )

            for step_idx, (c_tau_last, c_tau) in pbar:
                # [..., N_sample, N_atom, 3]
                x_l = centre_random_augmentation(x_input_coords=x_l, N_sample=1).squeeze(dim=-3).to(dtype)

                # Denoise with a predictor-corrector sampler
                # 1. Add noise to move x_{c_tau_last} to x_{t_hat}
                gamma = float(gamma0) if c_tau > gamma_min else 0
                t_hat = c_tau_last * (gamma + 1)

                delta_noise_level = torch.sqrt(t_hat**2 - c_tau_last**2)
                x_noisy = x_l + noise_scale_lambda * delta_noise_level * torch.randn(size=x_l.shape, device=device, dtype=dtype)

                # 2. Denoise from x_{t_hat} to x_{c_tau}
                # Euler step only
                t_hat = t_hat.reshape((1,) * (len(batch_shape) + 1)).expand(*batch_shape, chunk_n_sample).to(dtype)

                x_denoised = denoise_net(
                    x_noisy=x_noisy,
                    t_hat_noise_level=t_hat,
                    input_feature_dict=input_feature_dict,
                    s_inputs=s_inputs,
                    s_trunk=s_trunk,
                    z_trunk=z_trunk,
                    pair_z=None,
                    p_lm=None,
                    c_l=None,
                    chunk_size=attn_chunk_size,
                    inplace_safe=inplace_safe,
                    enable_efficient_fusion=False,
                )

                # Track reward if defined (optional for unconditional sampling)
                if reward is not None:
                    _reward_values = reward(x_denoised)
                    _reward_mean = _reward_values.mean().detach().item()
                    logger.info(f"Reward value: {_reward_mean} at time {c_tau:.2f}")

                    reward_history.append(
                        {
                            "step": step_idx,
                            "noise_level": float(c_tau_last.item()),
                            "reward": _reward_mean,
                        }
                    )

                delta = (x_noisy - x_denoised) / t_hat[
                    ..., None, None
                ]  # Line 9 of AF3 uses 'x_l_hat' instead, which we believe  is a typo.
                dt = c_tau - t_hat
                x_l = x_noisy + step_scale_eta * dt[..., None, None] * delta

            return x_l

        if diffusion_chunk_size is None:
            x_l = _chunk_sample_diffusion(N_sample, inplace_safe=inplace_safe)
        else:
            x_l = []
            no_chunks = N_sample // diffusion_chunk_size + (N_sample % diffusion_chunk_size != 0)
            for i in range(no_chunks):
                # Set sample offset so MapReward saves PDBs with correct sample indices
                # Skip for N_sample=1 so PDBs use plain naming (step_N.pdb, not step_N_sample_0.pdb)
                if reward is not None and hasattr(reward, "_sample_offset") and N_sample > 1:
                    reward._sample_offset = i * diffusion_chunk_size
                    reward._step_counter = 0
                chunk_n_sample = diffusion_chunk_size if i < no_chunks - 1 else N_sample - i * diffusion_chunk_size
                chunk_x_l = _chunk_sample_diffusion(chunk_n_sample, inplace_safe=inplace_safe)
                x_l.append(chunk_x_l)
            # Reset sample offset to default (single sample mode)
            if reward is not None and hasattr(reward, "_sample_offset") and N_sample > 1:
                reward._sample_offset = None
            x_l = torch.cat(x_l, -3)  # [..., N_sample, N_atom, 3]

        self.last_reward_history = reward_history

        # Profiling: record elapsed time and peak GPU memory
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        _profile_elapsed = _time.time() - _profile_start
        _profile_gpu_peak = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
        self.last_profile = {"elapsed_seconds": _profile_elapsed, "peak_gpu_memory_gb": _profile_gpu_peak}
        logger.info(f"[Profile] Diffusion loop: {_profile_elapsed:.1f}s, peak GPU: {_profile_gpu_peak:.2f} GB")

        return x_l

    def set_output_dir(self, output_dir: Path):
        self.output_dir = output_dir

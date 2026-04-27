from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import torch
from tqdm import tqdm

from protenix.model.utils import centre_random_augmentation
from protenix.utils.torch_utils import autocasting_disable_decorator
from protenix.utils.logger import get_logger

from embedopt.sampler.base import ProtenixBase


logger = get_logger(__name__)

Tensor = torch.Tensor
ScalarSchedule = Callable[[int, int], float]


class DiffusionPosteriorSampler(ProtenixBase):
    """Diffusion Posterior Sampling (DPS) for reward-based structure steering.

    Steers the diffusion process by computing reward gradients w.r.t. the noisy
    input coordinates at each denoising step.

    Reference:
        Chung et al., "Diffusion Posterior Sampling for General Noisy Inverse Problems",
        ICLR 2023. https://arxiv.org/abs/2209.14687
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Freeze all the model parameters
        for param in self.parameters():
            param.requires_grad = False

        # set allowed keys and dtype for each key
        self.allowed_keys = {"guided_scale": (float, int), "normalize_gradients": bool, "random_augmentation": bool}

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
                "attn_chunk_size": (
                    self.configs.infer_setting.chunk_size if not self.training else None
                ),
                "diffusion_chunk_size": (
                    self.configs.infer_setting.sample_diffusion_chunk_size
                    if not self.training
                    else None
                ),
            }
        )
        return autocasting_disable_decorator(self.configs.skip_amp.sample_diffusion)(
            self._sample_diffusion_dps
        )(**_configs, **kwargs)

    def _sample_diffusion_dps(
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
        if not hasattr(self, 'random_augmentation'):
            random_augmentation = False
        else:
            random_augmentation = self.random_augmentation

        N_atom = input_feature_dict["atom_to_token_idx"].size(-1)
        batch_shape = s_inputs.shape[:-2]
        device = s_inputs.device
        dtype = s_inputs.dtype

        reward = self.reward

        # Reset GPU memory stats for this sample
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        reward_history = []

        # Profiling: start timer
        import time as _time

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        _profile_start = _time.time()

        @torch.enable_grad()
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
                desc="Diffusion Posterior Sampling",
                ncols=170,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
            )

            for step_idx, (c_tau_last, c_tau) in pbar:
                # [..., N_sample, N_atom, 3]
                if random_augmentation:
                    x_l = (
                        centre_random_augmentation(x_input_coords=x_l, N_sample=1)
                        .squeeze(dim=-3)
                        .to(dtype)
                    )

                # Denoise with a predictor-corrector sampler
                # 1. Add noise to move x_{c_tau_last} to x_{t_hat}
                gamma = float(gamma0) if c_tau > gamma_min else 0
                t_hat = c_tau_last * (gamma + 1)

                delta_noise_level = torch.sqrt(t_hat**2 - c_tau_last**2)
                x_noisy = x_l + noise_scale_lambda * delta_noise_level * torch.randn(
                    size=x_l.shape, device=device, dtype=dtype
                )

                # 2. Denoise from x_{t_hat} to x_{c_tau}
                # Euler step only
                t_hat = (
                    t_hat.reshape((1,) * (len(batch_shape) + 1))
                    .expand(*batch_shape, chunk_n_sample)
                    .to(dtype)
                )

                x_noisy = x_noisy.requires_grad_(True)

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

                # compute reward
                _reward_value = reward(x_denoised)

                # take a gradient step of reward_value w.r.t. x_noisy
                guidance = torch.autograd.grad(_reward_value, x_noisy, retain_graph=False, create_graph=False)[0]
                guidance_norm = torch.linalg.norm(guidance, dim=(-1, -2), keepdim=True)
                guidance_norm_mean = guidance_norm.detach().mean().item()

                reward_history.append(
                    {
                        "step": step_idx,
                        "noise_level": float(c_tau_last.item()),
                        "reward": _reward_value.detach().item(),
                        "guidance_norm": guidance_norm_mean,
                    }
                )

                x_noisy = x_noisy.detach()
                x_denoised = x_denoised.detach()

                delta = (x_noisy - x_denoised) / t_hat[
                    ..., None, None
                ]  # Line 9 of AF3 uses 'x_l_hat' instead, which we believe  is a typo.

                if self.normalize_gradients:
                    delta_norm = delta.norm(dim=(-1, -2), keepdim=True)
                    guidance = guidance / (guidance_norm+1e-10) * delta_norm

                delta = delta - guidance * self.guided_scale

                dt = c_tau - t_hat # this is a negative value
                x_l = x_noisy + step_scale_eta * dt[..., None, None] * delta

                # Update progress bar with reward info (OUTSIDE torch.no_grad())
                if torch.cuda.is_available():
                    gpu_mem_gb = torch.cuda.max_memory_allocated() / 1024**3
                    pbar.set_postfix(
                        {
                            "Reward": f"{_reward_value.detach().item():+.4f}",
                            "Noise": f"{c_tau_last:.3f}",
                            "Guidance Norm": f"{guidance_norm_mean:.2e}",
                            "GPU": f"{gpu_mem_gb:.2f}GB",
                        }
                    )
                
                else:
                    pbar.set_postfix(
                        {
                            "Reward": f"{_reward_value.detach().item():+.4f}",
                            "Noise": f"{c_tau_last:.3f}",
                            "Guidance Norm": f"{guidance_norm_mean:.2e}",
                        }
                    )

            return x_l

        if diffusion_chunk_size is None:
            x_l = _chunk_sample_diffusion(N_sample, inplace_safe=inplace_safe)
        else:
            x_l = []
            no_chunks = N_sample // diffusion_chunk_size + (
                N_sample % diffusion_chunk_size != 0
            )
            for i in range(no_chunks):
                chunk_n_sample = (
                    diffusion_chunk_size
                    if i < no_chunks - 1
                    else N_sample - i * diffusion_chunk_size
                )
                chunk_x_l = _chunk_sample_diffusion(
                    chunk_n_sample, inplace_safe=inplace_safe
                )
                x_l.append(chunk_x_l)
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

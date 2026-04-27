import torch
from typing import Any, Optional
from pathlib import Path
from protenix.model.utils import centre_random_augmentation
from protenix.utils.torch_utils import autocasting_disable_decorator
from typing import Callable

from embedopt.reward import Reward
from tqdm import tqdm
from protenix.utils.logger import get_logger

from embedopt.sampler.base import ProtenixBase

logger = get_logger(__name__)


class EmbedOptSampler(ProtenixBase):
    """Embedding Optimization (EmbedOpt) sampler for reward-based structure steering.

    Steers the diffusion sampling process by optimizing the trunk embeddings (`s_trunk`, `z_trunk`)
    at each denoising step using reward gradients.
    """

    def __init__(
        self,
        **kwargs,
    ):
        """
        reward_scale_s: Scale factor for the s embeddings
        reward_scale_z: Scale factor for the z embeddings
        reset_embedding_fraction: Fraction of total steps at which to reset embeddings (0-1)
                                   None = no reset (default), 0.7 = reset at 70% of steps
        """
        super().__init__(**kwargs)

        # Freeze all the model parameters
        for param in self.parameters():
            param.requires_grad = False

        self.allowed_keys = {
            "reward_scale_s": (float, int),
            "reward_scale_z": (float, int),
            "random_augmentation": bool,
            "embedding_reg_scale": (float, int),
            "reset_embedding_fraction": (float, int, type(None)),
            "save_embeddings_every_n_steps": (int, type(None)),
            "output_dir": (str, type(None)),
            "override_embeddings_path": (str, type(None)),
            "ensemble_averaging": bool,
        }

    def set_reward(self, reward: Reward):
        self._reward = reward

    def sample_diffusion(self, **kwargs):
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
        return autocasting_disable_decorator(self.configs.skip_amp.sample_diffusion)(self._sample_diffusion_embedopt)(
            **_configs, **kwargs
        )

    def _sample_diffusion_embedopt(
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
        ensemble_averaging = getattr(self, "ensemble_averaging", False)

        # Ensemble averaging requires all particles to share the same embedding update,
        # so chunking must be disabled (each chunk would optimize independently).
        if ensemble_averaging and diffusion_chunk_size is not None:
            logger.warning("Ensemble averaging requires all particles to share embeddings. Disabling diffusion chunking.")
            diffusion_chunk_size = None

        # Without ensemble averaging, N_sample must be 1 to avoid ambiguity
        # (MapReward returns per-sample rewards, but only sample 0 would drive the gradient)
        if not ensemble_averaging:
            assert N_sample == 1, f"N_sample must be 1 when ensemble_averaging is False, got {N_sample}"

        if not hasattr(self, "embedding_reg_scale"):
            embedding_reg_scale = 0
        else:
            assert self.embedding_reg_scale >= 0, f"embedding_reg_scale must be non-negative, but got {self.embedding_reg_scale}"
            embedding_reg_scale = self.embedding_reg_scale

        if not hasattr(self, "random_augmentation"):
            random_augmentation = False
        else:
            random_augmentation = self.random_augmentation

        # Get reset_embedding_fraction parameter (None = no reset, 0.7 = reset at 70% of steps)
        if not hasattr(self, "reset_embedding_fraction"):
            reset_embedding_fraction = None  # Default: no reset
        else:
            reset_embedding_fraction = self.reset_embedding_fraction
            if reset_embedding_fraction is not None:
                assert (
                    0 <= reset_embedding_fraction <= 1
                ), f"reset_embedding_fraction must be between 0 and 1 or None, but got {reset_embedding_fraction}"

        assert self.reward_scale_s >= 0, f"reward_scale_s must be positive, but got {self.reward_scale_s}"
        assert self.reward_scale_z >= 0, f"reward_scale_z must be positive, but got {self.reward_scale_z}"

        N_atom = input_feature_dict["atom_to_token_idx"].size(-1)
        batch_shape = s_inputs.shape[:-2]
        device = s_inputs.device
        dtype = s_inputs.dtype

        # Reset GPU memory stats for this sample
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # Initialize reward tracking
        reward_history = []

        # Profiling: start timer
        import time as _time

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        _profile_start = _time.time()

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

        # Setup embedding saving
        save_embeddings_every_n_steps = getattr(self, "save_embeddings_every_n_steps", None)
        output_dir = getattr(self, "output_dir", None)
        save_dir = None

        if save_embeddings_every_n_steps is not None and output_dir is not None:
            save_dir = Path(output_dir) / "opted_vectors"
            save_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Will save optimized embeddings every {save_embeddings_every_n_steps} steps to {save_dir}")

        @torch.enable_grad()
        def _chunk_sample_diffusion(chunk_n_sample, inplace_safe):
            # init noise
            x_l = noise_schedule[0] * torch.randn(size=(*batch_shape, chunk_n_sample, N_atom, 3), device=device, dtype=dtype)

            # Create mutable copies for accumulating updates
            s_trunk_current = s_trunk.clone()
            z_trunk_current = z_trunk.clone()

            # Store initial embeddings for potential reset
            s_trunk_initial = s_trunk.clone()
            z_trunk_initial = z_trunk.clone()

            # Create progress bar
            total_steps = len(noise_schedule) - 1
            # Calculate reset threshold based on reset_embedding_fraction
            # None = no reset, 0.7 = reset at 70% of steps
            if reset_embedding_fraction is None:
                reset_threshold = total_steps + 1  # Never reset (threshold beyond total steps)
            else:
                reset_threshold = int(reset_embedding_fraction * total_steps)
            pbar = tqdm(
                enumerate(zip(noise_schedule[:-1], noise_schedule[1:])),
                total=total_steps,
                desc="Diffusion Sampling",
                ncols=170,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
            )

            for step_idx, (c_tau_last, c_tau) in pbar:
                # Reset embeddings for last 30% of steps
                if step_idx >= reset_threshold:
                    s_trunk_current = s_trunk_initial.clone()
                    z_trunk_current = z_trunk_initial.clone()

                # [..., N_sample, N_atom, 3]
                if random_augmentation:
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

                if self.reward is not None:
                    # First 70%: Optimize embeddings
                    if step_idx < reset_threshold:
                        # Create trainable deltas for this step
                        delta_s_trunk = torch.zeros_like(s_trunk_current, requires_grad=True)
                        delta_z_trunk = torch.zeros_like(z_trunk_current, requires_grad=True)

                        # Forward pass with perturbed embeddings
                        x_denoised = denoise_net(
                            x_noisy=x_noisy,
                            t_hat_noise_level=t_hat,
                            input_feature_dict=input_feature_dict,
                            s_inputs=s_inputs,
                            s_trunk=s_trunk_current + delta_s_trunk,
                            z_trunk=z_trunk_current + delta_z_trunk,
                            pair_z=None,
                            p_lm=None,
                            c_l=None,
                            chunk_size=attn_chunk_size,
                            inplace_safe=inplace_safe,
                            enable_efficient_fusion=False,
                        )

                        # Compute reward and gradients w.r.t. the perturbed embeddings
                        reward_value = self.reward(x_denoised)
                        reward_val = reward_value.item()

                        # take a gradient step of reward_value w.r.t. delta_s_trunk and delta_z_trunk
                        if embedding_reg_scale > 0:
                            regualrization = torch.mean((s_trunk_current + delta_s_trunk - s_trunk) ** 2) + torch.mean(
                                (z_trunk_current + delta_z_trunk - z_trunk) ** 2
                            )
                            regularization_value = embedding_reg_scale * regualrization
                            regualrization_val = regularization_value.item()
                        else:
                            regularization_value = 0
                            regualrization_val = 0

                        regularized_reward_value = reward_value - regularization_value

                        grad_s, grad_z = torch.autograd.grad(
                            regularized_reward_value, (delta_s_trunk, delta_z_trunk), retain_graph=False, create_graph=False
                        )

                        # Update embeddings for next iteration
                        with torch.no_grad():  # ← CRITICAL: Must wrap all update operations
                            # RMS normalization for interpretable scales
                            grad_s_rms = torch.sqrt((grad_s**2).mean(dim=(-1, -2), keepdim=True) + 1e-20)
                            grad_z_rms = torch.sqrt((grad_z**2).mean(dim=(-1, -2), keepdim=True) + 1e-20)

                            grad_s_normalized = grad_s / grad_s_rms
                            grad_z_normalized = grad_z / grad_z_rms

                            # Apply scaled, normalized gradients
                            s_trunk_current = s_trunk_current + grad_s_normalized * self.reward_scale_s
                            z_trunk_current = z_trunk_current + grad_z_normalized * self.reward_scale_z
                    else:
                        # Last 30%: Track reward but don't update embeddings (use initial)
                        x_denoised = denoise_net(
                            x_noisy=x_noisy,
                            t_hat_noise_level=t_hat,
                            input_feature_dict=input_feature_dict,
                            s_inputs=s_inputs,
                            s_trunk=s_trunk_current,  # Already reset to initial
                            z_trunk=z_trunk_current,
                            pair_z=None,
                            p_lm=None,
                            c_l=None,
                            chunk_size=attn_chunk_size,
                            inplace_safe=inplace_safe,
                            enable_efficient_fusion=False,
                        )

                        # Compute reward for tracking only
                        reward_values = self.reward(x_denoised)
                        reward_val = reward_values.mean().item()

                        regualrization_val = 0.0
                        grad_s = torch.zeros_like(s_trunk_current)
                        grad_z = torch.zeros_like(z_trunk_current)
                        grad_s_normalized = torch.zeros_like(s_trunk_current)
                        grad_z_normalized = torch.zeros_like(z_trunk_current)

                    # Track reward
                    with torch.no_grad():
                        cumulative_shift_s = (s_trunk_current - s_trunk_initial).norm().item()
                        cumulative_shift_z = (z_trunk_current - z_trunk_initial).norm().item()
                    reward_history.append(
                        {
                            "step": step_idx,
                            "noise_level": float(c_tau_last.item()),
                            "reward": reward_val,
                            "embedding_reg": regualrization_val,
                            "nabla_s": grad_s.norm().item(),
                            "nabla_z": grad_z.norm().item(),
                            "delta_s": (grad_s_normalized * self.reward_scale_s).norm().item(),
                            "delta_z": (grad_z_normalized * self.reward_scale_z).norm().item(),
                            "cumulative_shift_s": cumulative_shift_s,
                            "cumulative_shift_z": cumulative_shift_z,
                        }
                    )

                    # Update progress bar with reward info (OUTSIDE torch.no_grad())
                    if torch.cuda.is_available():
                        gpu_mem_gb = torch.cuda.max_memory_allocated() / 1024**3
                        with torch.no_grad():  # ← Also needed here for norm calculations
                            pbar.set_postfix(
                                {
                                    "Reward": f"{reward_val:+.4f}",
                                    "Embedding Reg": f"{regualrization_val:.4f}",
                                    "Noise": f"{c_tau_last:.3f}",
                                    "∇s": f"{grad_s.norm():.2e}",
                                    "∇z": f"{grad_z.norm():.2e}",
                                    "Δs": f"{(grad_s_normalized * self.reward_scale_s).norm():.3f}",
                                    "Δz": f"{(grad_z_normalized * self.reward_scale_z).norm():.3f}",
                                    "GPU": f"{gpu_mem_gb:.2f}GB",
                                }
                            )
                    else:
                        with torch.no_grad():  # ← And here too
                            pbar.set_postfix(
                                {
                                    "Reward": f"{reward_val:+.4f}",
                                    "Embedding Reg": f"{regualrization_val:.4f}",
                                    "Noise": f"{c_tau_last:.3f}",
                                    "∇s": f"{grad_s.norm():.2e}",
                                    "∇z": f"{grad_z.norm():.2e}",
                                    "Δs": f"{(grad_s_normalized * self.reward_scale_s).norm():.3f}",
                                    "Δz": f"{(grad_z_normalized * self.reward_scale_z).norm():.3f}",
                                }
                            )
                else:
                    # Update progress bar without reward
                    if torch.cuda.is_available():
                        gpu_mem_gb = torch.cuda.max_memory_allocated() / 1024**3
                        pbar.set_postfix({"Noise": f"{c_tau_last:.3f}", "GPU": f"{gpu_mem_gb:.2f}GB"})
                    else:
                        pbar.set_postfix({"Noise": f"{c_tau_last:.3f}"})

                # Final denoising step with current embeddings
                x_denoised = denoise_net(
                    x_noisy=x_noisy,
                    t_hat_noise_level=t_hat,
                    input_feature_dict=input_feature_dict,
                    s_inputs=s_inputs,
                    s_trunk=s_trunk_current,
                    z_trunk=z_trunk_current,
                    pair_z=None,
                    p_lm=None,
                    c_l=None,
                    chunk_size=attn_chunk_size,
                    inplace_safe=inplace_safe,
                    enable_efficient_fusion=False,
                )

                delta = (x_noisy - x_denoised) / t_hat[..., None, None]
                dt = c_tau - t_hat
                x_l = x_noisy + step_scale_eta * dt[..., None, None] * delta

                # Save embeddings at specified intervals (-1 = last step only)
                if save_dir is not None and save_embeddings_every_n_steps is not None:
                    if save_embeddings_every_n_steps == -1:
                        should_save = step_idx == total_steps - 1
                    else:
                        should_save = (step_idx + 1) % save_embeddings_every_n_steps == 0 or step_idx == total_steps - 1
                    if should_save:
                        save_path = save_dir / f"embeddings_step_{step_idx}.pt"
                        torch.save(
                            {
                                "s_trunk": s_trunk_current.detach().cpu(),
                                "z_trunk": z_trunk_current.detach().cpu(),
                                "step": step_idx,
                                "total_steps": total_steps,
                            },
                            save_path,
                        )
                        logger.info(f"Saved optimized embeddings at step {step_idx} to {save_path}")

            # Close progress bar
            pbar.close()

            return x_l, s_trunk_current, z_trunk_current

        if diffusion_chunk_size is None:
            x_l, s_trunk_opt, z_trunk_opt = _chunk_sample_diffusion(N_sample, inplace_safe=False)
            # explicitly set to False to avoid inplace operations, as we are taking gradients
        else:
            x_l = []
            s_trunk_opt_list = []
            z_trunk_opt_list = []
            no_chunks = N_sample // diffusion_chunk_size + (N_sample % diffusion_chunk_size != 0)
            for i in range(no_chunks):
                chunk_n_sample = diffusion_chunk_size if i < no_chunks - 1 else N_sample - i * diffusion_chunk_size
                chunk_x_l, s_opt, z_opt = _chunk_sample_diffusion(chunk_n_sample, inplace_safe=False)
                x_l.append(chunk_x_l)
                s_trunk_opt_list.append(s_opt)
                z_trunk_opt_list.append(z_opt)
            x_l = torch.cat(x_l, -3)
            # Take first chunk's embeddings (they should be similar across chunks)
            s_trunk_opt = s_trunk_opt_list[0]
            z_trunk_opt = z_trunk_opt_list[0]

        # Store reward history
        self.last_reward_history = reward_history

        # Profiling: record elapsed time and peak GPU memory
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        _profile_elapsed = _time.time() - _profile_start
        _profile_gpu_peak = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
        self.last_profile = {"elapsed_seconds": _profile_elapsed, "peak_gpu_memory_gb": _profile_gpu_peak}
        logger.info(f"[Profile] Diffusion loop: {_profile_elapsed:.1f}s, peak GPU: {_profile_gpu_peak:.2f} GB")

        # Store final optimized embeddings as instance attributes
        self.last_s_trunk_optimized = s_trunk_opt.detach().cpu()
        self.last_z_trunk_optimized = z_trunk_opt.detach().cpu()

        return x_l

"""Partial diffusion (SDEdit-style) for EmbedOpt: start from a given structure at an intermediate noise level.

Enabled by environment variables (set by the Makefile when INIT is given):
    PARTIAL_INIT   path to a structure (cif/pdb) whose atoms are in Protenix's model atom order, e.g. a prediction.
                   May contain "{seed}" and glob wildcards, resolved per seed (exactly one match), e.g.
                   out/SYS/moment2_lr0.1/*/seed_{seed}/predictions/*_sample_0.cif  (sequential / two-pass runs)
    PARTIAL_SIGMA  starting noise level in A; the noise schedule is truncated to the levels <= PARTIAL_SIGMA
The sampler initializes x = sigma_0 * randn(...); we add x_init / sigma_0 to that one draw, so x = x_init + sigma_0 * eps,
then run the remaining steps unchanged (same embedding updates, rewards and noise injection).
"""

import glob
import os

import biotite.structure.io as bio
import torch
from protenix.utils.logger import get_logger

logger = get_logger(__name__)


def load_init(template, seed):
    paths = sorted(glob.glob(template.format(seed=seed)))
    if len(paths) != 1:
        raise RuntimeError(f"partial diffusion: {template!r} with seed={seed} matches {len(paths)} files: {paths[:3]}")
    x = torch.as_tensor(bio.load_structure(paths[0]).coord, dtype=torch.float32)
    return paths[0], x - x.mean(0)


def install_partial_diffusion():
    template, sigma = os.environ.get("PARTIAL_INIT"), os.environ.get("PARTIAL_SIGMA")
    if not template:
        return
    sigma = float(sigma)

    import embedopt.sampler.embedopt as E
    from moments.reward import CURRENT

    original = E.EmbedOptSampler._sample_diffusion_embedopt

    def wrapped(self, *args, **kwargs):
        path, x_init = load_init(template, CURRENT["seed"])
        ns = kwargs["noise_schedule"]
        k = int((ns > sigma).sum())  # first index with noise level <= sigma
        kwargs["noise_schedule"] = ns[k:]
        sigma0 = float(ns[k])
        logger.info(f"[PartialDiffusion] seed {CURRENT['seed']}: start from {path} ({len(x_init)} atoms), "
                    f"sigma0 = {sigma0:.2f} A, {len(ns) - k - 1} of {len(ns) - 1} steps")
        real_randn, state = torch.randn, {"done": False}

        def randn(*a, **kw):
            out = real_randn(*a, **kw)
            if not state["done"] and out.shape[-2:] == x_init.shape:
                state["done"] = True
                out = out + (x_init / sigma0).to(out.device, out.dtype)
            return out

        torch.randn = randn
        try:
            result = original(self, *args, **kwargs)
        finally:
            torch.randn = real_randn
        if not state["done"]:
            raise RuntimeError(f"partial diffusion: no initial draw matched {tuple(x_init.shape)} atoms; is {path} in model atom order?")
        return result

    E.EmbedOptSampler._sample_diffusion_embedopt = wrapped
    logger.info(f"[PartialDiffusion] enabled: init {template}, noise <= {sigma} A")

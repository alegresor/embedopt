"""Partial diffusion (SDEdit-style) for EmbedOpt: start from a given structure at an intermediate noise level.

Enabled by environment variables (set by the Makefile when INIT is given):
    PARTIAL_INIT   path to a structure (cif/pdb) whose atoms are in Protenix's model atom order, e.g. a prediction
    PARTIAL_SIGMA  starting noise level in A; the noise schedule is truncated to the levels <= PARTIAL_SIGMA
The sampler initializes x = sigma_0 * randn(...); we add x_init / sigma_0 to that one draw, so x = x_init + sigma_0 * eps,
then run the remaining steps unchanged (same embedding updates, rewards and noise injection).
"""

import os

import biotite.structure.io as bio
import torch
from protenix.utils.logger import get_logger

logger = get_logger(__name__)


def install_partial_diffusion():
    path, sigma = os.environ.get("PARTIAL_INIT"), os.environ.get("PARTIAL_SIGMA")
    if not path:
        return
    sigma = float(sigma)
    a = bio.load_structure(path)
    x_init = torch.as_tensor(a.coord, dtype=torch.float32)
    x_init = x_init - x_init.mean(0)

    import embedopt.sampler.embedopt as E

    original = E.EmbedOptSampler._sample_diffusion_embedopt

    def wrapped(self, *args, **kwargs):
        ns = kwargs["noise_schedule"]
        k = int((ns > sigma).sum())  # first index with noise level <= sigma
        kwargs["noise_schedule"] = ns[k:]
        sigma0 = float(ns[k])
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
    logger.info(f"[PartialDiffusion] start from {path} ({len(x_init)} atoms) at noise <= {sigma} A")

"""Oracle moment reward for EmbedOpt (uniform pose density, exact invariants of the ground truth x*).

    order 2:  loss = m12 = |a0 - a0*|^2 / |a0*|^2 + sum_l |C_l - C_l*|_F^2 / sum_l |C_l*|_F^2   (first two moments)
    order 3:  loss = m12 * m3^LAMBDA3,  m3 = mean over degree triples of |G_b - G_b*|_F^2 / |G_b*|_F^2

Each bispectrum block b = (l1, l2, l3) is normalized separately, so all degrees count (an unnormalized norm is dominated
by l = 0 and barely sees Kam scrambling). Under EmbedOpt's RMS gradient normalization, loss = m12 * m3^lambda gives the
same updates as log m12 + lambda log m3, i.e. each term is weighted by its relative change. Reward = -loss.

Debye-Waller weighting (MOMENT_DW=1). The reward is evaluated on the posterior mean xhat0 = E[x0 | x_t]. Under the
Gaussian model behind AF3's EDM preconditioning, the posterior per-coordinate variance is
    s_t^2 = t^2 sigma_data^2 / (t^2 + sigma_data^2)        (t = current noise level, sigma_data = 16 A),
and isotropic positional uncertainty s_t damps the density by w_t(kappa) = exp(-2 pi^2 s_t^2 kappa^2) per shell. The
numerators of m12 and m3 are weighted by w_t on every shell index (w(k1) w(k2) on C_l, w(ka) w(kb) w(kc) on G), with
the denominators fixed at their unweighted values, so the reward's resolution follows the denoiser's own uncertainty:
low shells only at high noise, the unweighted reward as t -> 0. No tunable parameters. The noise level is read from a
hook on Protenix's DiffusionModule.forward (installed by the entry point below).

Image space (MOMENT_SPACE=image). The same structure is compared to data in image space instead of through the
invariants: a, c2_q (in-plane Fourier coefficients of the second moment) and c3_q1q2 on the lowest MOMENT_NB3 shells
(moments.image_moments); m12 and m3 are then the image-space losses, combined in the same way. MOMENT_DATA is "exact"
(moments of x* through the same maps; zero floor) or an .npz written by `make imagechecks` (N rendered images).
Debye-Waller weighting is not available in image space.

Configuration (environment variables, set by the Makefile): MOMENT_ORDER (2|3), MOMENT_DW (0|1), MOMENT_LB (bispectrum
bandlimit), MOMENT_LAMBDA3, MOMENT_NKAPPA, MOMENT_SPACE (invariant|image), MOMENT_DATA (exact|<npz>), MOMENT_NB3,
MOMENT_Q3, MOMENT_C3_CHUNK. The resolution comes from the reward params of the JSON config.
Per-step traces (t, s_t, m12, m3 as optimized, CA RMSD, wall time per diffusion step, peak GPU memory) are written to <trajectory dir>/moment_trace.json.

Usage: python -m moments.reward <embedopt.inference CLI>, with a config whose reward is
{"type": "map", "params": {"ref_pdb_path": <x* pdb>, "resolution": 5.0}} (ref_map_path is ignored).
"""

import json
import os
import re
import time

import numpy as np
import torch
from biotite.structure.io.pdb import PDBFile
from protenix.utils.logger import get_logger
from scipy.spatial.transform import Rotation

from embedopt.reward.base import Reward
from moments.image_moments import ImageLoss, ImageMoments, load_data
from moments.invariants import Invariants, form_factors, match_atoms

logger = get_logger(__name__)

NOISE = {"t": None, "sigma_data": None}  # set by the DiffusionModule hook at every denoiser call
CURRENT = {"seed": None}  # seed of the trajectory being sampled (parsed from the runner's output dir)
R_SAFETY = 1.2  # r_max = R_SAFETY * radius of x* (x0 may be less compact early on)
B_FACTOR = 50.0


def env(name, default, cast):
    return cast(os.environ.get(name, default))


def install_noise_hook():
    """Record the noise level of every denoiser call (the reward itself only receives xhat0)."""
    import protenix.model.modules.diffusion as diffusion

    forward = diffusion.DiffusionModule.forward

    def hooked(self, *args, **kwargs):
        t = kwargs.get("t_hat_noise_level", args[1] if len(args) > 1 else None)
        NOISE.update(t=None if t is None else float(t.reshape(-1)[0]), sigma_data=float(self.sigma_data))
        return forward(self, *args, **kwargs)

    diffusion.DiffusionModule.forward = hooked


def kabsch_rmsd(P, Q):  # min RMSD of Q onto P over O(3); returns (rmsd_proper, rmsd_mirror)
    P, Q = P - P.mean(0), Q - Q.mean(0)
    U, _, Vt = np.linalg.svd(Q.T @ P)
    d = np.sign(np.linalg.det(U @ Vt))
    return tuple(np.sqrt(np.mean(np.sum((Q @ U @ np.diag([1.0, 1.0, s]) @ Vt - P) ** 2, -1))) for s in (d, -d))


def _suffix(dt, mem):
    parts = ([] if dt is None else [f"step {dt:.2f}s"]) + ([] if mem is None else [f"peak {mem:.1f}GB"])
    return f" ({', '.join(parts)})" if parts else ""


class OracleMomentReward(Reward):
    def __init__(self, ref_pdb_path, pred_atom_array, resolution=5.0, agg="mean", device=None, order=None, dw=None,
                 output_dir=None, space=None, data=None, **unused):
        super().__init__(name="OracleMoment", agg=agg)
        self.order = order or env("MOMENT_ORDER", 2, int)
        self.dw = bool(env("MOMENT_DW", 0, int)) if dw is None else dw
        self.space = space or env("MOMENT_SPACE", "invariant", str)
        if self.space not in ("invariant", "image"):
            raise ValueError(f"MOMENT_SPACE must be invariant or image, got {self.space}")
        if self.space == "image" and self.dw:
            raise ValueError("Debye-Waller weighting (MOMENT_DW=1) is not implemented in image space")
        # image-space scoring is set up when optimizing in image space, or when data are given explicitly (moments.check)
        data = data if data is not None else (env("MOMENT_DATA", "exact", str) if self.space == "image" else None)
        self.trace_path = os.path.join(output_dir, "moment_trace.json") if output_dir else None
        seed = re.search(r"seed_(\d+)", output_dir or "")
        CURRENT["seed"] = int(seed.group(1)) if seed else None
        self.trace = []
        self.lam3 = env("MOMENT_LAMBDA3", 1.0, float)
        self.dev = dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ref = PDBFile.read(ref_pdb_path).get_structure(model=1)

        # Same atom set on both sides (else extra/missing atoms such as OXT bias the moments and R(x*) != 0)
        keep, order_idx = match_atoms(ref, pred_atom_array)
        self.keep = torch.as_tensor(keep, device=dev)
        x_ref = torch.as_tensor(ref.coord[order_idx], dtype=torch.float32, device=dev)
        x_ref = x_ref - x_ref.mean(0)
        self.ca_pred = pred_atom_array.atom_name[keep] == "CA"  # logging only
        self.ca_ref = ref.coord[order_idx][self.ca_pred]

        r_max = R_SAFETY * x_ref.norm(dim=1).max().item()
        self.inv = Invariants(r_max, resolution=resolution,
                              n_kappa=env("MOMENT_NKAPPA", 16, int), LB=env("MOMENT_LB", 24, int), dev=dev)
        self.rho = torch.as_tensor(form_factors(pred_atom_array.element[keep], self.inv.kappa.cpu().numpy(), B_FACTOR),
                                   dtype=torch.float32, device=dev)

        with torch.no_grad():
            A_ref = self.inv.coefficients(x_ref, self.rho)
            self.a0_ref, self.C_ref = self.inv.power(A_ref)
            G = self.inv.bispectrum(A_ref)
            n = G.pow(2).sum(dim=(3, 4, 5))
            self.blocks = n > 1e-12 * n.max()  # nonzero degree triples (triangle rule, even l1+l2+l3)
            self.G_ref, self.G_ref_norm = G, n
            # Checks: invariance to a rotation and to the reflection; Kam scramble (odd degrees from a rotated copy)
            # leaves m12 unchanged but should change m3
            Q = torch.as_tensor(Rotation.random(random_state=0).as_matrix(), dtype=torch.float32, device=dev)
            A_rot = self.inv.coefficients(x_ref @ Q.T, self.rho)
            A_scr = A_ref.clone()
            for l in range(1, self.inv.L + 1, 2):
                A_scr[:, l * l:(l + 1) ** 2] = A_rot[:, l * l:(l + 1) ** 2]
            chk = {"rotation": A_rot, "reflection": self.inv.coefficients(-x_ref, self.rho), "Kam scramble": A_scr}
            chk_img = chk
            chk = ", ".join(f"{k} m12={self.m12(A):.1e} m3={self.m3(A):.1e}" for k, A in chk.items())
            self.img = None
            if data is not None:
                n_kappa = len(self.inv.kappa)
                self.img = ImageMoments(np.arange(1, n_kappa + 1) / (n_kappa * resolution), r_max, L=self.inv.L,
                                        nb3=env("MOMENT_NB3", 8, int), Q3=env("MOMENT_Q3", 16, int),
                                        chunk=env("MOMENT_C3_CHUNK", 256, int), dev=dev)
                self.img_data = data
                self.img_loss = ImageLoss(load_data(data, self.img, A_ref))
                chk_img = {"x* vs data": A_ref, **chk_img}
                chk_img = ", ".join(f"{k} m12={a:.1e} m3={b:.1e}" for k, (a, b) in
                                    ((k, self.img_losses(A, True)) for k, A in chk_img.items()))
                logger.info(f"[OracleMoment] {self.img.describe()}; data {data}; image-space checks: {chk_img}")
        logger.info(f"[OracleMoment] order {self.order}, space {self.space}, DW weighting {self.dw}, {len(order_idx)} matched atoms (dropped {(~keep).sum()} model, "
                    f"{len(ref) - len(order_idx)} ref incl. H), L={self.inv.L}, {len(self.inv.omega)} S^2 nodes, "
                    f"{len(self.inv.kappa)} shells; bispectrum LB={self.inv.LB}, {len(self.inv.shells)} shells, "
                    f"{int(self.blocks.sum())} blocks, {len(self.inv.w2)} nodes; checks: {chk} "
                    f"(global TF32 matmul flag outside the reward: {torch.backends.cuda.matmul.allow_tf32})")

    def shell_weights(self, s):
        """Debye-Waller weights w(kappa) = exp(-2 pi^2 s^2 kappa^2) for positional uncertainty s (A); s = 0 -> ones."""
        return torch.exp(-2 * np.pi**2 * s**2 * self.inv.kappa**2)

    def posterior_std(self):
        t, sd = NOISE["t"], NOISE["sigma_data"]
        return 0.0 if (not self.dw or t is None) else float(np.sqrt(t**2 * sd**2 / (t**2 + sd**2)))

    def m12(self, A, s=0.0):
        w = self.shell_weights(s)
        a0, C = self.inv.power(A)
        W = w[:, None] * w[None, :]
        return ((w * (a0 - self.a0_ref)).abs().pow(2).sum() / self.a0_ref.abs().pow(2).sum()
                + sum((W * (c - cr)).abs().pow(2).sum() for c, cr in zip(C, self.C_ref))
                / sum(cr.abs().pow(2).sum() for cr in self.C_ref))

    def m3(self, A, s=0.0):
        w = self.shell_weights(s)[self.inv.shells]
        W = w[:, None, None] * w[None, :, None] * w[None, None, :]
        d = (W * (self.inv.bispectrum(A) - self.G_ref)).pow(2).sum(dim=(3, 4, 5))
        return (d[self.blocks] / self.G_ref_norm[self.blocks]).mean()

    def img_losses(self, A, third=None):
        """Image-space (m12, m3 or None) of coefficients A against the image data."""
        third = self.order >= 3 if third is None else third
        return self.img_loss(self.img.moments(A, third))

    def losses(self, x, s=0.0, space=None, third=None):
        """x: (N_model_atoms, 3) in model order; s: DW positional uncertainty (A); space: invariant | image (default:
        the reward's own). Returns (m12, m3 or None)."""
        space, third = space or self.space, (self.order >= 3 if third is None else third)
        x = x[self.keep]
        A = self.inv.coefficients(x - x.mean(0), self.rho)
        if space == "image":
            return self.img_losses(A, third)
        return self.m12(A, s), (self.m3(A, s) if third else None)

    def compute(self, sample, pdb_save_path=None):
        s = self.posterior_std()
        now, rewards = time.time(), []
        step_dt, self._last = (None if getattr(self, "_last", None) is None else now - self._last), now
        for i in range(sample.shape[-3]):
            x = sample[..., i, :, :]
            m12, m3 = self.losses(x, s)
            loss = m12 if m3 is None else m12 * m3.pow(self.lam3)
            rewards.append(-loss)
            xc = x[self.keep].detach().cpu().numpy()
            rp, rm = kabsch_rmsd(self.ca_ref, xc[self.ca_pred])
            rec = dict(step=len(self.trace), t=NOISE["t"], s=s, m12=m12.item(), m3=None if m3 is None else m3.item(),
                       rmsd=float(rp), rmsd_mirror=float(rm), space=self.space, dt=step_dt,
                       mem=torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None)
            self.trace.append(rec)
            logger.info(f"[OracleMoment] t {rec['t'] or float('nan'):.3g}, s {s:.2f}A, m12 {rec['m12']:.4e}"
                        + ("" if m3 is None else f", m3 {rec['m3']:.4e}") + f", CA RMSD to x*: proper {rp:.2f}A, mirror {rm:.2f}A"
                        + _suffix(step_dt, rec["mem"]))
        if self.trace_path:
            with open(self.trace_path, "w") as f:
                json.dump(self.trace, f)
        r = torch.stack(rewards)
        return r.mean() if self.agg == "mean" else r.sum() if self.agg == "sum" else r


if __name__ == "__main__":
    # Use the objects of the importable module `moments.reward`, not this `__main__` copy: otherwise the module is
    # loaded twice and state shared with other modules (NOISE, CURRENT) would live in two separate dictionaries.
    import embedopt.reward.map as _map
    import moments.reward as mr
    from moments.partial import install_partial_diffusion

    _map.MapReward = mr.OracleMomentReward  # inference.py builds MapReward for reward type "map"
    mr.install_noise_hook()
    install_partial_diffusion()  # no-op unless PARTIAL_INIT is set
    from embedopt.inference.inference import main

    main()
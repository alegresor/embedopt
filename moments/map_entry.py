"""Entry point for Li et al.'s map-guided EmbedOpt with optional partial diffusion (PARTIAL_INIT / PARTIAL_SIGMA).

Identical to `python -m embedopt.inference.inference`, except that
  - MapReward records the trajectory's seed (per-seed PARTIAL_INIT templates) and the partial-diffusion patch is
    installed (no-op without PARTIAL_INIT);
  - the CA correspondence for MapReward's frame is replaced by the identity when the reference has exactly the
    predicted CA residue sequence (a docked Protenix model); logged at startup;
  - MAP_KABSCH_CYCLES=k sets the outlier-rejection cycles of MapReward's per-step Kabsch alignment (0: one global fit);
  - MAP_FRAME=fit replaces the per-step alignment to the reference structure by rigid-body refinement against the map.

MAP_FRAME=fit. MapReward places x0hat in the map by Kabsch-aligning it to ref_pdb_path at every step. That frame is
only correct when the reference is (an unbiased noisy copy of) the true structure: with a realistic reference, e.g. a
model docked into the map, the alignment follows the reference's own errors and the guidance collapses the structure.
With MAP_FRAME=fit the reference is used once, to initialize the frame at the first step; afterwards the rigid
transform (R, T) of x0hat is refined against the map at every step, warm-started from the previous step:
    aligned(w, tau) = (x - c) Exp(w)^T R^T ... = (x - c) (Exp(w) R)^T + T + tau,     c = centroid of x (constant),
minimizing the reward's own loss |normalize(map(aligned)) - normalize(V_obs)|^2 by MAP_FIT_STEPS (default 4; 30 at the
first step) normalized-gradient steps on (w, tau) with backtracking (initial step 1 degree / 0.5 A, halved on
rejection). The reward and its gradient are then evaluated at the refined, fixed transform, exactly as MapReward
does with its Kabsch transform. No structure other than x0hat enters after the first step.
"""

import functools
import os
import re

import numpy as np
import torch

import embedopt.reward.map as _map
from embedopt.util import structure as _structure
from moments.partial import install_partial_diffusion
from moments.reward import CURRENT

_MapReward = _map.MapReward
_KABSCH = _structure.iterative_kabsch_alignment_common_ca
if os.environ.get("MAP_KABSCH_CYCLES") is not None:
    _KABSCH = functools.partial(_KABSCH, cycles=int(os.environ["MAP_KABSCH_CYCLES"]))
_map.iterative_kabsch_alignment_common_ca = _KABSCH


def so3_exp(w):
    """Rotation matrix of a rotation vector (Rodrigues), differentiable, safe at w = 0."""
    th2 = (w * w).sum()
    big = th2 > 1e-12
    th = torch.sqrt(torch.where(big, th2, torch.ones_like(th2)))  # safe value in the unused branch (no NaN grads)
    K = torch.zeros(3, 3, dtype=w.dtype, device=w.device)
    K[0, 1], K[0, 2], K[1, 0], K[1, 2], K[2, 0], K[2, 1] = -w[2], w[1], w[2], -w[0], -w[1], w[0]
    a = torch.where(big, torch.sin(th) / th, 1 - th2 / 6)
    b = torch.where(big, (1 - torch.cos(th)) / th**2, 0.5 - th2 / 24)
    return torch.eye(3, dtype=w.dtype, device=w.device) + a * K + b * (K @ K)


def rigid_fit(loss_fn, x, R, T, steps, rot_step=np.deg2rad(1.0), trans_step=0.5):
    """Refine (R, T) of aligned = (x - c) R^T + T to decrease loss_fn(aligned); x detached. Returns R, T, loss."""
    c = x.mean(0)
    xc = x - c
    w = torch.zeros(3, dtype=x.dtype, device=x.device)
    tau = torch.zeros(3, dtype=x.dtype, device=x.device)

    def f(w, tau):
        return loss_fn(xc @ (so3_exp(w) @ R).T + T + tau)

    with torch.no_grad():
        best = f(w, tau).item()
    sr, st = rot_step, trans_step
    for _ in range(steps):
        wq, tq = w.clone().requires_grad_(True), tau.clone().requires_grad_(True)
        gw, gt = torch.autograd.grad(f(wq, tq), (wq, tq))
        dw = -sr * gw / (gw.norm() + 1e-30)
        dt = -st * gt / (gt.norm() + 1e-30)
        with torch.no_grad():
            val = f(w + dw, tau + dt).item()
        if val < best:
            w, tau, best = w + dw, tau + dt, val
            sr, st = sr * 1.2, st * 1.2
        else:
            sr, st = sr / 2, st / 2
    with torch.no_grad():
        return so3_exp(w) @ R, T + tau, best


class SeededMapReward(_MapReward):
    def __init__(self, *args, output_dir=None, **kwargs):
        seed = re.search(r"seed_(\d+)", output_dir or "")
        CURRENT["seed"] = int(seed.group(1)) if seed else None
        super().__init__(*args, output_dir=output_dir, **kwargs)
        names = ("ref_pdb_path", "pred_atom_array")
        ref_path, pred = (kwargs[n] if n in kwargs else args[i] for i, n in enumerate(names))
        self._check_ca_mapping(ref_path, pred)
        self.frame_mode = os.environ.get("MAP_FRAME", "kabsch")
        self.fit_steps = int(os.environ.get("MAP_FIT_STEPS", 4))
        self._frame = None  # (R, T) of the previous step (single sample)
        self._calls = 0
        if self.frame_mode == "fit":
            print(f"[MapReward frame] rigid-body refinement against the map, {self.fit_steps} steps per diffusion "
                  f"step (30 at the first); reference used only to initialize", flush=True)

    def _check_ca_mapping(self, ref_path, pred):
        import SFC_Torch as sfc

        ref = sfc.PDBParser(ref_path)
        ca_ref = [i for i, n in enumerate(ref.cra_name) if n.split("-")[-1] == "CA"]
        ca_pred = np.flatnonzero(np.asarray(pred.atom_name) == "CA")
        pos_r = {a: k for k, a in enumerate(ca_ref)}
        pos_p = {a: k for k, a in enumerate(ca_pred)}
        old = [(pos_p.get(int(i)), pos_r.get(int(j))) for i, j in zip(self.common_ca_ind_1, self.common_ca_ind_2)]
        offsets = sorted({p - r for p, r in old if p is not None and r is not None})
        same_seq = [ref.cra_name[i].split("-")[2] for i in ca_ref] == list(np.asarray(pred.res_name)[ca_pred])
        msg = (f"[MapReward CA mapping] inherited: {len(old)} pairs, residue offsets (pred - ref) {offsets[:5]}"
               f"{' ...' if len(offsets) > 5 else ''}; reference has the predicted sequence: {same_seq}")
        if same_seq:
            self.common_ca_ind_1, self.common_ca_ind_2 = [int(i) for i in ca_pred], ca_ref
            self.ref_ca_numpy = np.asarray(ref.atom_pos)[ca_ref].copy()
            msg += f" -> using the identity correspondence ({len(ca_ref)} CA)"
        print(msg, flush=True)

    def _loss(self, aligned):
        return torch.mean((self._model2map_no_save(aligned) - self.valid_ref_map) ** 2)

    def _fit_frame(self, moving, ref_ca, ind, **kw):
        x = moving.detach()
        if self._frame is None:  # initialize from the reference (Kabsch with outlier rejection), then refine longer
            a0 = _KABSCH(x, ref_ca, ind)
            c = x.mean(0)
            U, _, Vt = torch.linalg.svd((x - c).T @ (a0 - a0.mean(0)))
            d = torch.sign(torch.linalg.det(U @ Vt))
            D = torch.diag(torch.stack([torch.ones_like(d), torch.ones_like(d), d]))
            R, T, steps = (U @ D @ Vt).T, a0.mean(0), 30
        else:
            (R, T), steps = self._frame, self.fit_steps
        R, T, loss = rigid_fit(self._loss, x, R, T, steps)
        self._frame = (R, T)
        if self._calls % 25 == 0:
            print(f"[MapReward frame] step {self._calls}: map loss after rigid refinement {loss:.4f}", flush=True)
        self._calls += 1
        c = x.mean(0)
        return (moving - c) @ R.T + T  # gradient flows through moving; R, T, c constant (as in MapReward)

    def compute(self, sample, pdb_save_path=None):
        if self.frame_mode != "fit":
            return super().compute(sample, pdb_save_path)
        if sample.shape[-3] != 1:
            raise ValueError("MAP_FRAME=fit supports one sample per trajectory (N_sample = 1)")
        orig = _map.iterative_kabsch_alignment_common_ca
        _map.iterative_kabsch_alignment_common_ca = self._fit_frame
        try:
            return super().compute(sample, pdb_save_path)
        finally:
            _map.iterative_kabsch_alignment_common_ca = orig


if __name__ == "__main__":
    _map.MapReward = SeededMapReward  # inference.py builds MapReward for reward type "map"
    install_partial_diffusion()  # no-op unless PARTIAL_INIT is set
    from embedopt.inference.inference import main

    main()
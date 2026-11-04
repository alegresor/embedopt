"""Oracle moment reward for EmbedOpt (uniform pose density, exact invariants of the ground truth x*).

    order 2:  loss = m12 = |a0 - a0*|^2 / |a0*|^2 + sum_l |C_l - C_l*|_F^2 / sum_l |C_l*|_F^2   (first two moments)
    order 3:  loss = m12 * m3^LAMBDA3,  m3 = mean over degree triples of |G_b - G_b*|_F^2 / |G_b*|_F^2

Each bispectrum block b = (l1, l2, l3) is normalized separately, so all degrees count (an unnormalized norm is dominated
by l = 0 and barely sees Kam scrambling). Under EmbedOpt's RMS gradient normalization, loss = m12 * m3^lambda gives the
same updates as log m12 + lambda log m3, i.e. each term is weighted by its relative change. Reward = -loss.

Configuration (environment variables, set by the Makefile): MOMENT_ORDER (2|3), MOMENT_LB (bispectrum bandlimit),
MOMENT_LAMBDA3, MOMENT_NKAPPA. The resolution comes from the reward params of the JSON config.

Usage: python -m moments.reward <embedopt.inference CLI>, with a config whose reward is
{"type": "map", "params": {"ref_pdb_path": <x* pdb>, "resolution": 5.0}} (ref_map_path is ignored).
"""

import os

import numpy as np
import torch
from biotite.structure.io.pdb import PDBFile
from protenix.utils.logger import get_logger
from scipy.spatial.transform import Rotation

from embedopt.reward.base import Reward
from moments.invariants import Invariants, form_factors, match_atoms

logger = get_logger(__name__)

R_SAFETY = 1.2  # r_max = R_SAFETY * radius of x* (x0 may be less compact early on)
B_FACTOR = 50.0


def env(name, default, cast):
    return cast(os.environ.get(name, default))


def kabsch_rmsd(P, Q):  # min RMSD of Q onto P over O(3); returns (rmsd_proper, rmsd_mirror)
    P, Q = P - P.mean(0), Q - Q.mean(0)
    U, _, Vt = np.linalg.svd(Q.T @ P)
    d = np.sign(np.linalg.det(U @ Vt))
    return tuple(np.sqrt(np.mean(np.sum((Q @ U @ np.diag([1.0, 1.0, s]) @ Vt - P) ** 2, -1))) for s in (d, -d))


class OracleMomentReward(Reward):
    def __init__(self, ref_pdb_path, pred_atom_array, resolution=5.0, agg="mean", device=None, order=None, **unused):
        super().__init__(name="OracleMoment", agg=agg)
        self.order = order or env("MOMENT_ORDER", 2, int)
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

        self.inv = Invariants(R_SAFETY * x_ref.norm(dim=1).max().item(), resolution=resolution,
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
            chk = ", ".join(f"{k} m12={self.m12(A):.1e} m3={self.m3(A):.1e}" for k, A in chk.items())
        logger.info(f"[OracleMoment] order {self.order}, {len(order_idx)} matched atoms (dropped {(~keep).sum()} model, "
                    f"{len(ref) - len(order_idx)} ref incl. H), L={self.inv.L}, {len(self.inv.omega)} S^2 nodes, "
                    f"{len(self.inv.kappa)} shells; bispectrum LB={self.inv.LB}, {len(self.inv.shells)} shells, "
                    f"{int(self.blocks.sum())} blocks, {len(self.inv.w2)} nodes; checks: {chk} "
                    f"(global TF32 matmul flag outside the reward: {torch.backends.cuda.matmul.allow_tf32})")

    def m12(self, A):
        a0, C = self.inv.power(A)
        return ((a0 - self.a0_ref).abs().pow(2).sum() / self.a0_ref.abs().pow(2).sum()
                + sum((c - cr).abs().pow(2).sum() for c, cr in zip(C, self.C_ref)) / sum(cr.abs().pow(2).sum() for cr in self.C_ref))

    def m3(self, A):
        d = (self.inv.bispectrum(A) - self.G_ref).pow(2).sum(dim=(3, 4, 5))
        return (d[self.blocks] / self.G_ref_norm[self.blocks]).mean()

    def losses(self, x):
        """x: (N_model_atoms, 3) in model order. Returns (m12, m3 or None)."""
        x = x[self.keep]
        A = self.inv.coefficients(x - x.mean(0), self.rho)
        return self.m12(A), (self.m3(A) if self.order >= 3 else None)

    def compute(self, sample, pdb_save_path=None):
        rewards = []
        for i in range(sample.shape[-3]):
            x = sample[..., i, :, :]
            m12, m3 = self.losses(x)
            loss = m12 if m3 is None else m12 * m3.pow(self.lam3)
            rewards.append(-loss)
            msg = f"[OracleMoment] m12 {m12.item():.4e}" + ("" if m3 is None else f", m3 {m3.item():.4e}")
            xc = x[self.keep].detach().cpu().numpy()
            rp, rm = kabsch_rmsd(self.ca_ref, xc[self.ca_pred])
            logger.info(msg + f", CA RMSD to x*: proper {rp:.2f}A, mirror {rm:.2f}A")
        r = torch.stack(rewards)
        return r.mean() if self.agg == "mean" else r.sum() if self.agg == "sum" else r


if __name__ == "__main__":
    import embedopt.reward.map as _map

    _map.MapReward = OracleMomentReward  # inference.py builds MapReward for reward type "map"
    from embedopt.inference.inference import main

    main()
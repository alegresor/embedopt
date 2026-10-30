"""Stage 3a oracle: EmbedOpt steered by uniform-mu first/second moments (Kam invariants) of x*.

Under uniform mu the first two image moments depend on the structure only through
    M1(kappa)  ~ A_00(kappa)                       (radial profile)
    C_l(k1,k2) = sum_m A_lm(k1) conj(A_lm(k2))     (per-degree Gram matrices, Kam)
where A_lm(kappa) are the spherical-harmonic coefficients of Vhat[x] on the sphere of radius kappa:
    Vhat[x](kappa w) = sum_n rho_n(kappa) exp(-2 pi i kappa w . q_n),   rho_n = electron f_n * exp(-B kappa^2/4).
A_lm is computed by exact Gauss-Legendre x trapezoid quadrature on S^2 (Y_lm precomputed, so only exp() is
differentiated). Reward = -( |A_00 - A_00*|^2/|A_00*|^2 + sum_l |C_l - C_l*|^2 / sum_l |C_l*|^2 ).
It is invariant to rotations and reflections of x, so no Kabsch alignment is used: only centering.
Kabsch to x* (proper and mirrored) is computed for logging only.

Usage: same CLI as embedopt.inference.inference, with a config whose reward is
{"type": "map", "params": {"ref_pdb_path": <x* pdb>, "resolution": 5.0}} (ref_map_path ignored).
"""

import functools

import biotite.structure as bs
import gemmi
import numpy as np
import torch
from biotite.structure.io.pdb import PDBFile
from numpy.polynomial.legendre import leggauss
from protenix.utils.logger import get_logger
from scipy.spatial.transform import Rotation

from embedopt.reward.base import Reward

logger = get_logger(__name__)

N_KAPPA = 16      # radial shells kappa_i = kappa_max * i / N_KAPPA, i = 1..N_KAPPA
L_PAD = 8         # extra SH degrees beyond 2 pi kappa_max r_max
R_SAFETY = 1.2    # r_max = R_SAFETY * radius of x* (x0 may be less compact early on)
B_FACTOR = 50.0


def sph_harm(l, m, theta, phi):  # theta polar, phi azimuth; handles old/new scipy APIs
    import scipy.special as sp

    return sp.sph_harm_y(l, m, theta, phi) if hasattr(sp, "sph_harm_y") else sp.sph_harm(m, l, phi, theta)


@functools.lru_cache(maxsize=2)
def s2_quadrature(L, dev):
    """Exact quadrature on S^2 for degree <= 2L+1: Gauss-Legendre in cos(theta) x trapezoid in phi.
    Returns nodes omega (Nq, 3) and weighted conj(Y_lm) (Nq, (L+1)^2)."""
    ct, wt = leggauss(L + 1)
    ph = 2 * np.pi * np.arange(2 * L + 2) / (2 * L + 2)
    TH, PH = np.meshgrid(np.arccos(ct), ph, indexing="ij")
    TH, PH = TH.ravel(), PH.ravel()
    w = (wt[:, None] * np.full(len(ph), 2 * np.pi / len(ph))[None]).ravel()
    omega = np.stack([np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH), np.cos(TH)], 1)
    ls = np.concatenate([np.full(2 * l + 1, l) for l in range(L + 1)])
    ms = np.concatenate([np.arange(-l, l + 1) for l in range(L + 1)])
    WYc = (w[:, None] * sph_harm(ls[None], ms[None], TH[:, None], PH[:, None]).conj()).astype(np.complex64)
    return torch.as_tensor(omega, dtype=torch.float32, device=dev), torch.as_tensor(WYc, device=dev)


def atom_keys(a):  # "residue ordinal:atom name"; residue ordinal counts residues in chain order
    starts = bs.get_residue_starts(a)
    res = np.searchsorted(starts, np.arange(len(a)), side="right") - 1
    return [f"{r}:{n}" for r, n in zip(res, a.atom_name)]


def kabsch_rmsd(P, Q):  # min RMSD of Q onto P over O(3); returns (rmsd_proper, rmsd_mirror)
    P, Q = P - P.mean(0), Q - Q.mean(0)
    U, S, Vt = np.linalg.svd(Q.T @ P)
    d = np.sign(np.linalg.det(U @ Vt))
    out = []
    for sign in (d, -d):  # sign = d -> proper rotation, -d -> improper (mirror)
        D = np.diag([1.0, 1.0, sign])
        out.append(np.sqrt(np.mean(np.sum((Q @ U @ D @ Vt - P) ** 2, -1))))
    return tuple(out)


class Stage3aOracleReward(Reward):
    def __init__(self, ref_pdb_path, pred_atom_array, resolution=5.0, agg="mean", device=None, **unused):
        super().__init__(name="Stage3aOracle", agg=agg)
        self.dev = dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ref = PDBFile.read(ref_pdb_path).get_structure(model=1)
        ref = ref[ref.element != "H"]

        # Match model and reference atoms by (residue ordinal, atom name) so both sides have the SAME atom set
        # (otherwise extra/missing atoms, e.g. OXT or truncated side chains, bias the moments and R(x*) != 0)
        ref_idx = {k: i for i, k in enumerate(atom_keys(ref))}
        model_keys = atom_keys(pred_atom_array)
        keep = np.array([k in ref_idx and e != "H" for k, e in zip(model_keys, pred_atom_array.element)])
        order = np.array([ref_idx[k] for k, m in zip(model_keys, keep) if m])
        self.keep = torch.as_tensor(keep, device=dev)
        pred_el = pred_atom_array.element[keep]
        x_ref = torch.as_tensor(ref.coord[order], dtype=torch.float32, device=dev)  # model atom order
        x_ref = x_ref - x_ref.mean(0)
        n_drop_model, n_drop_ref = len(pred_atom_array) - keep.sum(), len(ref) - len(order)

        # CA atoms for logging only
        self.ca_pred = pred_atom_array.atom_name[keep] == "CA"
        self.ca_ref = ref.coord[order][self.ca_pred]

        # Radial grid and SH bandlimit
        kmax = 1.0 / resolution
        self.kappa = torch.arange(1, N_KAPPA + 1, device=dev, dtype=torch.float32) * kmax / N_KAPPA
        r_max = R_SAFETY * x_ref.norm(dim=1).max().item()
        self.L = L = int(np.ceil(2 * np.pi * kmax * r_max)) + L_PAD

        self.omega, self.WYc = s2_quadrature(L, str(dev))  # cached: depends only on L (saves ~45 s per seed)

        # Atomic transforms rho_n(kappa) = f_e(kappa) exp(-B kappa^2 / 4), f_e from gemmi (as in SFC_Torch cryoem)
        k2 = (self.kappa**2).cpu().numpy()
        table = {e: np.array([gemmi.Element(e).c4322.calculate_sf(s / 4) for s in k2]) for e in set(pred_el) | set(ref.element)}
        rho = lambda els: torch.as_tensor(np.stack([table[e] for e in els]) * np.exp(-B_FACTOR * k2 / 4), dtype=torch.float32, device=dev)
        self.rho_pred = rho_ref = rho(pred_el)  # matched atoms: same elements on both sides

        # Oracle invariants and invariance checks (should be ~0)
        with torch.no_grad():
            self.a0_ref, self.C_ref = self.invariants(x_ref, rho_ref)
            Q = torch.as_tensor(Rotation.random(random_state=0).as_matrix(), dtype=torch.float32, device=dev)
            err_rot = -self.loss_to_ref(x_ref @ Q.T, rho_ref)
            err_mir = -self.loss_to_ref(-x_ref, rho_ref)
        logger.info(
            f"[Stage3a] {len(order)} matched atoms (dropped {n_drop_model} model, {n_drop_ref} ref), L={L}, {len(self.omega)} S^2 nodes, "
            f"{N_KAPPA} shells; invariance check: R(Q x*)={err_rot:.2e}, R(-x*)={err_mir:.2e}"
        )

    def invariants(self, x, rho):
        """x: (N,3) centered, rho: (N, N_KAPPA). Returns A_00 (N_KAPPA,), [C_l (N_KAPPA, N_KAPPA)]_l."""
        proj = 2 * np.pi * (x @ self.omega.T)  # (N, Nq)
        # exp(-i k proj) via real cos/sin (complex exp is JIT-compiled by NVRTC in torch)
        V = torch.stack([torch.complex(rho[:, i] @ torch.cos(k * proj), -(rho[:, i] @ torch.sin(k * proj))) for i, k in enumerate(self.kappa)])
        A = V @ self.WYc  # (N_KAPPA, (L+1)^2)
        C = [A[:, l * l:(l + 1) ** 2] @ A[:, l * l:(l + 1) ** 2].conj().T for l in range(self.L + 1)]
        return A[:, 0], C

    def loss_to_ref(self, x, rho):
        a0, C = self.invariants(x, rho)
        m1 = (a0 - self.a0_ref).abs().pow(2).sum() / self.a0_ref.abs().pow(2).sum()
        m2 = sum((c - cr).abs().pow(2).sum() for c, cr in zip(C, self.C_ref)) / sum(cr.abs().pow(2).sum() for cr in self.C_ref)
        return -(m1 + m2)

    def compute(self, sample, pdb_save_path=None):
        rewards = []
        for i in range(sample.shape[-3]):
            x = sample[..., i, :, :][self.keep]
            x = x - x.mean(0)
            rewards.append(self.loss_to_ref(x, self.rho_pred))
            msg = f"[Stage3a] reward {rewards[-1].item():.4e}"
            if self.ca_pred.sum() == len(self.ca_ref):
                rp, rm = kabsch_rmsd(self.ca_ref, x[torch.as_tensor(self.ca_pred, device=self.dev)].detach().cpu().numpy())
                msg += f", CA RMSD to x*: proper {rp:.2f}A, mirror {rm:.2f}A"
            logger.info(msg)
        r = torch.stack(rewards)
        return r.mean() if self.agg == "mean" else r.sum() if self.agg == "sum" else r


if __name__ == "__main__":
    import embedopt.reward.map as _map

    _map.MapReward = Stage3aOracleReward  # inference.py builds MapReward for reward type "map"
    from embedopt.inference.inference import main

    main()
"""Rotation-invariant moment features of an atomic structure, computed in Fourier space.

For a structure x with atoms q_n, Vhat[x](eta) = sum_n rho_n(|eta|) exp(-2 pi i eta . q_n). On each radial shell
|eta| = kappa, expand Vhat in spherical harmonics: Vhat(kappa w) = sum_lm A_lm(kappa) Y_l^m(w).

  Second order (power spectrum):  a0 = A_00(kappa),  C_l(k1,k2) = sum_m A_lm(k1) conj(A_lm(k2)).
      Under uniform pose density these carry exactly the information of the first two image moments.
  Third order (bispectrum):  G_{l1 l2 l3}(ka,kb,kc) = int_{S^2} f_l1(ka,w) f_l2(kb,w) f_l3(kc,w) dw,
      f_l(kappa,w) = sum_m A_lm(kappa) Y_l^m(w) (degree-l band of the shell). By the Gaunt formula this is the
      3j-contracted bispectrum up to a nonzero constant for even l1+l2+l3 (odd sums vanish, consistent with the
      handedness ambiguity of projections). It couples different degrees, so it is NOT invariant under independent
      per-degree rotations (Kam's ambiguity), unlike C_l.

Both are exactly invariant to rotations and reflections of x. Quadratures on S^2 are product Gauss-Legendre x
trapezoid rules, exact for degree <= 2L+1.
"""

import contextlib
import functools

import biotite.structure as bs
import gemmi
import numpy as np
import torch
from numpy.polynomial.legendre import leggauss
from torch.utils.checkpoint import checkpoint


def sph_harm(l, m, theta, phi):  # theta polar, phi azimuth; handles old/new scipy APIs
    import scipy.special as sp

    return sp.sph_harm_y(l, m, theta, phi) if hasattr(sp, "sph_harm_y") else sp.sph_harm(m, l, phi, theta)


@contextlib.contextmanager
def full_precision():
    """Exact float32 matmuls: no TF32 (Protenix enables it globally) and no autocast. The bispectrum blocks are sums
    with heavy cancellation, and per-block normalization amplifies TF32's ~1e-3 relative error into O(1e-2) losses."""
    flags = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    try:
        with torch.autocast("cuda", enabled=False):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = flags


@functools.lru_cache(maxsize=4)
def s2_quadrature(L, dev):
    """Nodes omega (Nq, 3), weights w (Nq,), Y_lm at the nodes (Nq, (L+1)^2); exact for degree <= 2L+1."""
    ct, wt = leggauss(L + 1)
    ph = 2 * np.pi * np.arange(2 * L + 2) / (2 * L + 2)
    TH, PH = np.meshgrid(np.arccos(ct), ph, indexing="ij")
    TH, PH = TH.ravel(), PH.ravel()
    w = (wt[:, None] * np.full(len(ph), 2 * np.pi / len(ph))[None]).ravel()
    omega = np.stack([np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH), np.cos(TH)], 1)
    ls = np.concatenate([np.full(2 * l + 1, l) for l in range(L + 1)])
    ms = np.concatenate([np.arange(-l, l + 1) for l in range(L + 1)])
    Y = sph_harm(ls[None], ms[None], TH[:, None], PH[:, None]).astype(np.complex64)
    return (torch.as_tensor(omega, dtype=torch.float32, device=dev), torch.as_tensor(w, dtype=torch.float32, device=dev),
            torch.as_tensor(Y, device=dev))


def atom_keys(a):
    """'residue ordinal:atom name' for each atom; the residue ordinal counts residues in chain order."""
    starts = bs.get_residue_starts(a)
    res = np.searchsorted(starts, np.arange(len(a)), side="right") - 1
    return [f"{r}:{n}" for r, n in zip(res, a.atom_name)]


def match_atoms(ref, pred):
    """Boolean mask over pred atoms kept, and the matching indices into ref (same order). Hydrogens dropped."""
    ref_idx = {k: i for i, k in enumerate(atom_keys(ref)) if ref.element[i] != "H"}
    keys = atom_keys(pred)
    keep = np.array([k in ref_idx and e != "H" for k, e in zip(keys, pred.element)])
    return keep, np.array([ref_idx[k] for k, m in zip(keys, keep) if m])


def form_factors(elements, kappa, B):
    """rho_n(kappa) = f_e(kappa) exp(-B kappa^2 / 4), f_e the 5-Gaussian electron factor (gemmi c4322, as SFC_Torch)."""
    k2 = kappa**2
    table = {e: np.array([gemmi.Element(e).c4322.calculate_sf(s / 4) for s in k2]) for e in set(elements)}
    return np.stack([table[e] for e in elements]) * np.exp(-B * k2 / 4)


class Invariants:
    """Power-spectrum and bispectrum invariants on N_kappa shells up to kappa_max = 1 / resolution.

    L      : SH bandlimit for A_lm, from the radius r_max of the reference structure (2 pi kappa_max r_max + pad)
    LB     : bandlimit of the bispectrum (degrees l <= LB enter G)
    shells : indices of the shells used in the bispectrum (subset of the N_kappa shells)
    """

    def __init__(self, r_max, resolution=5.0, n_kappa=16, l_pad=8, LB=24, bispec_shell_stride=2, dev="cpu"):
        self.dev = dev
        self.kappa = torch.arange(1, n_kappa + 1, dtype=torch.float32, device=dev) / (n_kappa * resolution)
        self.L = int(np.ceil(2 * np.pi * self.kappa[-1].item() * r_max)) + l_pad
        self.omega, w, Y = s2_quadrature(self.L, str(dev))
        self.WYc = w[:, None] * Y.conj()  # analysis: A_lm = sum_q w_q Vhat(kappa w_q) conj(Y_lm(w_q))
        self.LB = LB = min(LB, self.L)
        self.shells = torch.arange(bispec_shell_stride - 1, n_kappa, bispec_shell_stride, device=dev)
        # Synthesis grid for the bispectrum: integrand degree <= 3 LB  =>  L_q = ceil((3 LB - 1) / 2)
        self.omega2, self.w2, Y2 = s2_quadrature(int(np.ceil((3 * LB - 1) / 2)), str(dev))
        self.Y2 = Y2[:, : (LB + 1) ** 2]
        # (-i)^l makes f_l real for a real density (f_l(-w) = conj f_l(w) and f_l(-w) = (-1)^l f_l(w))
        self.phase = torch.as_tensor(np.concatenate([np.full(2 * l + 1, (-1j) ** l) for l in range(LB + 1)]),
                                     dtype=torch.complex64, device=dev)

    def coefficients(self, x, rho):
        """x: (N, 3) centered, rho: (N, N_kappa). Returns A: (N_kappa, (L+1)^2)."""
        with full_precision():
            return self._coefficients(x.float(), rho)

    def _coefficients(self, x, rho):
        # One shell at a time, checkpointed: the (N, Nq) phase arrays are recomputed in the backward pass instead of
        # stored for all shells (memory ~ 1 shell instead of N_kappa shells, ~30% more time)
        V = torch.stack([checkpoint(_shell, x, self.omega, rho[:, i], k, use_reentrant=False)
                         for i, k in enumerate(self.kappa)])
        return V @ self.WYc

    def power(self, A):
        """a0 (N_kappa,), [C_l (N_kappa, N_kappa) for l = 0..L]."""
        with full_precision():
            return A[:, 0], [A[:, l * l:(l + 1) ** 2] @ A[:, l * l:(l + 1) ** 2].conj().T for l in range(self.L + 1)]

    def bispectrum(self, A, chunk=512):
        """G as a real tensor of shape (LB+1, LB+1, LB+1, S, S, S) over (l1, l2, l3, shell a, shell b, shell c)."""
        with full_precision():
            return self._bispectrum(A, chunk)

    def _bispectrum(self, A, chunk):
        LB, S = self.LB, len(self.shells)
        A = A[self.shells, : (LB + 1) ** 2] * self.phase  # (S, (LB+1)^2)
        F = torch.stack([(A[:, l * l:(l + 1) ** 2] @ self.Y2[:, l * l:(l + 1) ** 2].T).real for l in range(LB + 1)])
        F = F.reshape((LB + 1) * S, -1)  # (X, Nq2) with X = (l, shell)
        G = sum(checkpoint(_triple, F[:, i:i + chunk], self.w2[i:i + chunk], use_reentrant=False)
                for i in range(0, F.shape[1], chunk))
        return G.reshape(LB + 1, S, LB + 1, S, LB + 1, S).permute(0, 2, 4, 1, 3, 5)


def _shell(x, omega, r, k):
    """Vhat on the nodes of one shell: sum_n r_n exp(-2 pi i k omega . q_n), via real cos/sin (complex exp is
    JIT-compiled through NVRTC in torch)."""
    with full_precision():  # also applies when the checkpoint recomputes this in the backward pass
        phase = (2 * np.pi * k) * (x @ omega.T)
        return torch.complex(r @ torch.cos(phase), -(r @ torch.sin(phase)))


def _triple(F, w):
    with full_precision():
        return torch.einsum("xq,yq,zq->xyz", F * w, F, F)
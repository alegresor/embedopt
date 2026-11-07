"""Image-space moments for the oracle moment reward (uniform poses): differentiable torch port of image_checks.Model.

From the spherical-harmonic coefficients A_lm(kappa) of a structure (moments.invariants), the moments of its images in
in-plane Fourier coefficients I_q(kappa) are
    a(k)             = A_00(k) / sqrt(4 pi)                                     first moment (radial)
    c2_q(k1, k2)     = (1/4pi) sum_{l >= |q|} beta_lq C_l(k1, k2),   q = 0..L     beta_lq: Fourier coeffs of P_l(cos phi)
    c3_q1q2(a, b, c) = E_g[ I_q1(a) I_q2(b) I_{-q1-q2}(c) ],   |q| <= Q3, on NB3 shells, full bandlimit LB there.
The third-order shells are the lowest NB3 (S3STRIDE = 0, default) or every S3STRIDE-th shell, as in the invariant
bispectrum (S3STRIDE = 2: shells 2, 4, ..., 16 of 16, out to the full resolution); LB is set by the outermost one.
c3 is an exact quadrature over viewing directions g = Rz(alpha) Ry(beta) (Gauss-Legendre in cos beta x trapezoid in
alpha, exact for the degree-3LB integrand); the in-plane rotation is already integrated by the selection rule
q1 + q2 + q3 = 0. The node images are I_q(g) = sum_m e^{i m alpha} sum_l A_lm E^l_mq(beta), E^l_mq(beta) the in-plane
Fourier coefficients of Y_lm on the great circle Ry(beta) e(phi).

Losses. Default (identical to image_checks.losses):
    m12 (plain) = |a - a*|^2 / |a*|^2 + |c2 - c2*|^2 / |c2*|^2,
    m3 (block)  = mean over (q1, q2) blocks of |c3 - c3*|^2 / |c3*|^2 (summed over the shell triple), nonzero blocks only.
Alternatives (MOMENT_M12, MOMENT_M3):
    m12 white : the invariant m12 with the data's C_l recovered from c2 (c2_q = sum_l B_ql C_l, B = beta^T / 4pi upper
                triangular): |a - a*|^2 / |a*|^2 + sum_{l <= L - drop} |C_l - C~_l|^2 / sum |C~_l|^2, C~ = B^-1 c2*.
                The top `drop` degrees, where the inversion amplifies data error, are left out.
    m3 perq   : sum_r w_r |c3_r - c3*_r|^2 / sum_r w_r |c3*_r|^2, w_r = 1 / (K K^T)_rr (moments.c3_metric).
    m3 white  : |c3 - c3*|_W^2 / |c3*|_W^2 with W = (K K^T)^+: the error of the minimum-norm bispectrum estimate
                K^+ c3, i.e. the bispectrum restricted to its part visible in images (moments.c3_metric).
                Its norm is dominated by the large low-degree (second-order-like) triples, so it barely sees Kam's
                ambiguity (Kam scramble ~1e-6); see wblock.
    m3 wblock : as white, for the block-normalized bispectrum (each degree triple scaled to unit data norm, as in the
                invariant m3): K' = K diag(s), s_l = |B^_l| over shell triples from the minimum-norm estimate of the
                data, W' = (K' K'^T)^+. All from the data side (no oracle block norms). K' is much worse conditioned
                than K; with rcond_b = 1e-6 the kept rank is too small to see Kam's ambiguity, hence rcond_b = 1e-9.
Data: "exact" = the moments of x* through these same maps (zero floor), or an .npz written by image_checks
(a, c2, c3, kappa, L, nb3, Q3), checked against this grid.

Precision: tables are built in float64 and cast; evaluation in complex64 inside full_precision() (no TF32/autocast).
c3 is accumulated over node chunks with gradient checkpointing (memory ~ one chunk).

Self-test (no Protenix needed): python -m moments.image_moments --systems 8AHU_A 8K9Z_A 8H1I_A [--s3-stride 2 --q3 24]
"""

import functools
import os
import time

import numpy as np
import torch
from numpy.polynomial.legendre import leggauss, legval
from torch.utils.checkpoint import checkpoint

from moments.invariants import full_precision, sph_harm

L_PAD = 8


def lm_index(L):
    ls = np.concatenate([np.full(2 * l + 1, l) for l in range(L + 1)])
    ms = np.concatenate([np.arange(-l, l + 1) for l in range(L + 1)])
    return ls, ms


def third_shells(n_kappa, nb3, stride=0):
    """Shell indices of the third order: the lowest nb3 (stride 0) or every stride-th shell (as Invariants.shells)."""
    sh = np.arange(nb3) if stride == 0 else np.arange(stride - 1, n_kappa, stride)[:nb3]
    if len(sh) < nb3 or sh[-1] >= n_kappa:
        raise ValueError(f"cannot take {nb3} third-order shells with stride {stride} from {n_kappa} shells")
    return sh


def c3_tag(stride, Q3):
    """Suffix for run names and data files when the third-order grid is not the default (lowest shells, |q| <= 16)."""
    return "" if (stride == 0 and Q3 == 16) else f"_c3s{stride}q{Q3}"


def grid(kappa, r_max, shells3, l_pad=L_PAD):
    """Bandlimits as in image_checks: L for all shells, LB for the outermost third-order shell (<= L), in-plane grid."""
    L = int(np.ceil(2 * np.pi * kappa[-1] * r_max)) + l_pad
    LB = min(L, int(np.ceil(2 * np.pi * kappa[shells3[-1]] * r_max)) + l_pad)
    return L, LB, int(16 * np.ceil(2 * (L + 1) / 16))


@functools.lru_cache(maxsize=4)
def tables(L, LB, nphi, Q3):
    """float64 tables (cached per process): beta (L+1, L+1); E padded as (m, l, beta node, q); index/mask (m, l) into
    the (LB+1)^2 coefficient vector; phase e^{i m alpha} (alpha node, m); Haar weights in node order (alpha, beta)."""
    phi = 2 * np.pi * np.arange(nphi) / nphi
    e = np.stack([np.cos(phi), np.sin(phi), np.zeros(nphi)], 1)
    beta = np.stack([np.fft.fft(legval(np.cos(phi), np.eye(L + 1)[l])) / nphi for l in range(L + 1)])[:, : L + 1].real
    nb, na = (3 * LB) // 2 + 2, 3 * LB + 2
    cb, wb = leggauss(nb)
    ls, ms = lm_index(LB)
    qq = np.arange(-2 * Q3, 2 * Q3 + 1)
    M = 2 * LB + 1
    Ep = np.zeros((M, LB + 1, nb, len(qq)), np.complex64)  # complex64: ~1.4 GB at LB = 83, |q| <= 24
    for k, b in enumerate(np.arccos(cb)):
        u = e @ np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]]).T
        Y = sph_harm(ls[None], ms[None], np.arccos(np.clip(u[:, 2], -1, 1))[:, None], np.arctan2(u[:, 1], u[:, 0])[:, None])
        Ep[ms + LB, ls, k] = (np.fft.fft(Y, axis=0) / nphi)[qq % nphi].T
    idx = np.zeros((M, LB + 1), int)
    idx[ms + LB, ls] = np.arange(len(ls))
    mask = np.zeros((M, LB + 1))
    mask[ms + LB, ls] = 1.0
    phase = np.exp(1j * np.outer(2 * np.pi * np.arange(na) / na, np.arange(-LB, LB + 1)))
    w = np.broadcast_to(wb[None, :] / (2 * na), (na, nb)).ravel()  # node n = alpha index * nb + beta index
    return beta, Ep, idx, mask, phase, w


class ImageMoments:
    """Differentiable map A (N_kappa, >= (L+1)^2) -> dict(a, c2, c3) on the image_checks grid."""

    def __init__(self, kappa, r_max, L=None, nb3=8, Q3=16, stride3=0, chunk=256, dev="cpu", dtype=torch.float32):
        self.kappa = np.asarray(kappa, np.float64)
        self.shells3 = third_shells(len(self.kappa), nb3, stride3)
        Lg, self.LB, self.nphi = grid(self.kappa, r_max, self.shells3)
        self.L = Lg if L is None else L
        if self.L != Lg:
            self.nphi = int(16 * np.ceil(2 * (self.L + 1) / 16))
        self.LB = min(self.LB, self.L)
        self.nb3, self.Q3, self.stride3, self.chunk = nb3, Q3, stride3, chunk
        self.cdt = torch.complex64 if dtype == torch.float32 else torch.complex128
        beta, Ep, idx, mask, phase, w = tables(self.L, self.LB, self.nphi, Q3)
        c = lambda a: torch.as_tensor(a, dtype=self.cdt, device=dev)
        self.beta, self.Ep, self.phase = c(beta), c(Ep).reshape(Ep.shape[0], Ep.shape[1], -1), c(phase)
        self.mask, self.w = c(mask), torch.as_tensor(w.copy(), dtype=dtype, device=dev)
        self.idx = torch.as_tensor(idx, device=dev)
        self.sh3 = torch.as_tensor(self.shells3, device=dev)
        self.nb, self.na = Ep.shape[2], phase.shape[0]
        qs = np.arange(-Q3, Q3 + 1)
        self.i1 = torch.as_tensor(qs + 2 * Q3, device=dev)
        self.i3 = torch.as_tensor(-qs[:, None] - qs[None, :] + 2 * Q3, device=dev)

    def describe(self):
        which = f"the lowest {self.nb3} shells" if self.stride3 == 0 else f"{self.nb3} shells (every {self.stride3}th: {[int(v) for v in self.shells3 + 1]})"
        return (f"image space: L={self.L}, n_phi={self.nphi}, third order on {which} "
                f"(to {1 / self.kappa[self.shells3[-1]]:.1f} A) with LB={self.LB}, |q|<={self.Q3}, "
                f"{self.nb * self.na} viewing directions ({self.nb} x {self.na}), chunk {self.chunk}")

    def moments(self, A, third=True):
        with full_precision():
            A = A.to(self.cdt)
            a = A[:, 0] / np.sqrt(4 * np.pi)
            C = torch.stack([A[:, l * l:(l + 1) ** 2] @ A[:, l * l:(l + 1) ** 2].conj().T for l in range(self.L + 1)])
            c2 = (self.beta.T @ C.reshape(self.L + 1, -1)).reshape(C.shape) / (4 * np.pi)
            return dict(a=a, c2=c2, C=C, c3=self.c3(A) if third else None)

    def node_images(self, A):
        """In-plane coefficients at every viewing direction, lowest nb3 shells: (nodes, nb3, 4 Q3 + 1)."""
        with full_precision():
            Ap = A[self.sh3[:, None, None], self.idx[None]] * self.mask             # (a, m, l)
            H = Ap.permute(1, 0, 2) @ self.Ep                                       # (m, a, nb * q)
            I = (self.phase @ H.reshape(H.shape[0], -1)).reshape(self.na, self.nb3, self.nb, -1)
            return I.permute(0, 2, 1, 3).reshape(self.na * self.nb, self.nb3, -1)

    def c3(self, A):
        I = self.node_images(A)
        out = 0
        for i in range(0, I.shape[0], self.chunk):
            sl = slice(i, i + self.chunk)
            if torch.is_grad_enabled() and I.requires_grad:
                out = out + checkpoint(_c3_chunk, I[sl], self.w[sl], self.i1, self.i3, use_reentrant=False)
            else:
                out = out + _c3_chunk(I[sl], self.w[sl], self.i1, self.i3)
        return out  # (q1, q2, a, b, c)


def _c3_chunk(I, w, i1, i3):
    """sum_n w_n I_q1(a) I_q2(b) I_{-q1-q2}(c) over one chunk of nodes, as Q^2 batched (nb3^2 x n) @ (n x nb3)."""
    with full_precision():
        X, X3 = I[:, :, i1], I[:, :, i3]                                         # (n, a, Q), (n, c, Q, Q)
        Q, S = X.shape[-1], X.shape[1]
        Y = torch.einsum("naq,ndr->qradn", X * w[:, None, None], X).reshape(Q, Q, S * S, -1)
        return (Y @ X3.permute(2, 3, 0, 1)).reshape(Q, Q, S, S, S)


class ImageLoss:
    """m12 and m3 against reference moments (dict a, c2, c3). m12: plain | white; m3: block | perq | white."""

    M12, M3 = ("plain", "white"), ("block", "perq", "white", "wblock")

    def __init__(self, ref, im=None, m12="plain", m3="block", drop=2, rcond=1e-6, rcond_b=1e-9, all_variants=False):
        if m12 not in self.M12 or m3 not in self.M3:
            raise ValueError(f"m12 must be one of {self.M12}, m3 one of {self.M3}; got {m12}, {m3}")
        self.ref, self.mode12, self.mode3 = ref, m12, m3
        self.na, self.n2 = ref["a"].abs().pow(2).sum(), ref["c2"].abs().pow(2).sum()
        n = ref["c3"].abs().pow(2).sum(dim=(2, 3, 4))
        self.keep, self.n3 = n > 1e-12 * n.max(), n
        self.info = ""
        if m12 == "white" or all_variants:
            # C~_l = B^-1 c2* per shell pair (float64), degrees l <= L - drop
            L = ref["c2"].shape[0] - 1
            Bm = im.beta.real.double().T.cpu().numpy() / (4 * np.pi)
            c2 = ref["c2"].cpu().numpy().astype(np.complex128)
            Ct = np.linalg.solve(Bm, c2.reshape(L + 1, -1)).reshape(c2.shape)
            self.lw = L + 1 - drop
            self.Ct = torch.as_tensor(Ct[: self.lw], dtype=ref["c2"].dtype, device=ref["c2"].device)
            self.nC = self.Ct.abs().pow(2).sum()
            self.info += f"m12 white: C_l from c2 by B^-1 (cond {np.linalg.cond(Bm):.0f}), l <= {self.lw - 1}; "
        if m3 in ("perq", "white", "wblock") or all_variants:
            from moments.c3_metric import block_metric, metric

            M = metric(im.LB, im.Q3, rcond=rcond)
            dev = ref["c3"].device
            self.w_r = torch.as_tensor(np.where(M["diag"] > 1e-14 * M["diag"].max(), 1 / np.maximum(M["diag"], 1e-300), 0.0),
                                       dtype=torch.float64, device=dev)
            self.Wh = torch.as_tensor(M["U"] / np.sqrt(M["lam"]), dtype=torch.float64, device=dev)  # (rows, rank)
            self.nP = self._perq(ref["c3"])
            self.nW = self._white(ref["c3"])
            self.info += (f"m3 metric: {M['rows']} rows x {M['ncol']} degree triples, rank {M['rank']} "
                          f"(rcond {rcond:g}, cond {M['cond']:.1e}); ")
        if m3 == "wblock" or all_variants:
            # metric of the block-normalized bispectrum: columns of K scaled by the data's (minimum-norm) block norms
            c3r = self._rows(ref["c3"]).cpu().numpy()
            Wb, nvis = block_metric(im.LB, im.Q3, c3r, M, rcond=rcond_b)
            self.Wb = torch.as_tensor(Wb, dtype=torch.float64, device=ref["c3"].device)
            self.nB = self._wblock(ref["c3"])
            self.info += f"m3 wblock: {nvis} visible degree triples, rank {Wb.shape[1]} (rcond {rcond_b:g}); "

    def _rows(self, c3):
        Q = c3.shape[0]
        return c3.reshape(Q * Q, -1).to(torch.complex128)                            # (rows, shell triples)

    def _perq(self, c3):
        return (self.w_r[:, None] * self._rows(c3).abs().pow(2)).sum()

    def _white(self, c3):
        with full_precision():
            return (self.Wh.T.to(torch.complex128) @ self._rows(c3)).abs().pow(2).sum()

    def _wblock(self, c3):
        with full_precision():
            return (self.Wb.T.to(torch.complex128) @ self._rows(c3)).abs().pow(2).sum()

    def m12(self, m, mode=None):
        mode = mode or self.mode12
        e_a = (m["a"] - self.ref["a"]).abs().pow(2).sum() / self.na
        if mode == "white":
            return e_a + (m["C"][: self.lw] - self.Ct).abs().pow(2).sum() / self.nC
        return e_a + (m["c2"] - self.ref["c2"]).abs().pow(2).sum() / self.n2

    def m3(self, m, mode=None):
        mode = mode or self.mode3
        d = m["c3"] - self.ref["c3"]
        if mode == "perq":
            return (self._perq(d) / self.nP).to(m["c3"].real.dtype)
        if mode == "white":
            return (self._white(d) / self.nW).to(m["c3"].real.dtype)
        if mode == "wblock":
            return (self._wblock(d) / self.nB).to(m["c3"].real.dtype)
        e = d.abs().pow(2).sum(dim=(2, 3, 4))
        return (e[self.keep] / self.n3[self.keep]).mean()

    def __call__(self, m):
        return self.m12(m), (self.m3(m) if m["c3"] is not None else None)

    def variants(self, m):
        """All variants (needs all_variants=True): m12_img, m12w_img, m3_img (block), m3p_img, m3w_img, m3b_img."""
        return dict(m12_img=self.m12(m, "plain"), m12w_img=self.m12(m, "white"), m3_img=self.m3(m, "block"),
                    m3p_img=self.m3(m, "perq"), m3w_img=self.m3(m, "white"), m3b_img=self.m3(m, "wblock"))


def load_data(src, im, A_ref):
    """Reference moments: 'exact' (x* through im) or an image_checks .npz (validated against im's grid)."""
    if src == "exact":
        with torch.no_grad():
            return im.moments(A_ref, third=True)
    if not os.path.isfile(src):
        raise FileNotFoundError(f"MOMENT_DATA={src} not found (make imagechecks writes out/<SYS>/image_moments_N*.npz)")
    d = np.load(src)
    S, Q = im.nb3, 2 * im.Q3 + 1
    want = dict(a=(len(im.kappa),), c2=(im.L + 1, len(im.kappa), len(im.kappa)), c3=(Q, Q, S, S, S))
    bad = [f"{k}: {d[k].shape} vs {s}" for k, s in want.items() if d[k].shape != s]
    if not np.allclose(d["kappa"], im.kappa, rtol=1e-6):
        bad.append("kappa grid differs")
    bad += [f"{k}: {int(d[k])} vs {v}" for k, v in (("L", im.L), ("nb3", im.nb3), ("Q3", im.Q3)) if int(d[k]) != v]
    sh = d["shells3"] if "shells3" in d else np.arange(int(d["nb3"]))  # files written before strides: lowest shells
    if not np.array_equal(sh, im.shells3):
        bad.append(f"third-order shells {[int(v) for v in sh]} vs {[int(v) for v in im.shells3]}")
    if bad:
        raise ValueError(f"image data {src} does not match the reward grid: " + "; ".join(bad))
    dev = im.beta.device
    return {k: torch.as_tensor(d[k], dtype=im.cdt, device=dev) for k in ("a", "c2", "c3")}


# ----------------------------------------------------------------------------------------------- self-test
def _selftest(sys_, nb3, Q3, chunk, stride3=0, bench="examples/synthetic_map_benchmark"):
    """Port (complex64) vs image_checks.Model (float64); floors vs the .npz; rotation / Kam checks; gradient check
    (complex128 directional derivative); timing and peak memory of one forward + backward."""
    import biotite.structure.io as bio

    from moments.invariants import form_factors

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ref = bio.load_structure(f"{bench}/pdb_clean/{sys_}.pdb")
    ref = ref[ref.element != "H"]
    x = ref.coord.astype(np.float64)
    x -= x.mean(0)
    _run_selftest(sys_, x, ref.element, form_factors, nb3, Q3, chunk, dev,
                  f"out/{sys_}/image_moments_N1000000{c3_tag(stride3, Q3)}.npz", stride3=stride3)


def _run_selftest(name, x, elements, form_factors, nb3, Q3, chunk, dev, npz=None, resolution=5.0, n_kappa=16, stride3=0):
    from scipy.spatial.transform import Rotation

    from moments import image_checks as ic

    t0 = time.time()
    kappa = np.arange(1, n_kappa + 1) / (n_kappa * resolution)
    r_max = 1.2 * np.linalg.norm(x, axis=1).max()
    rho = form_factors(elements, kappa, 50.0)
    shells3 = third_shells(n_kappa, nb3, stride3)
    L, LB, nphi = grid(kappa, r_max, shells3)
    omega, WYc = ic.s2_quadrature64(L)
    coef = lambda y: ic.coefficients64(y, rho, kappa, omega, WYc)
    A = coef(x)
    ref64 = ic.Model(L, LB, nphi, nb3, Q3, shells3=shells3).moments(A)[0]
    im = ImageMoments(kappa, r_max, nb3=nb3, Q3=Q3, stride3=stride3, chunk=chunk, dev=dev)
    print(f"\n=== {name}: {len(x)} atoms; {im.describe()}; device {dev}; setup {time.time() - t0:.0f}s")
    T = lambda a: torch.as_tensor(a, dtype=torch.complex64, device=dev)
    m32 = {k: v.cpu().numpy() for k, v in im.moments(T(A)).items()}
    errs = {k: np.linalg.norm(m32[k] - ref64[k]) / np.linalg.norm(ref64[k]) for k in ("a", "c2", "c3")}
    print("  1. port (complex64) vs image_checks.Model (float64): " + "  ".join(f"{k} {v:.1e}" for k, v in errs.items()))
    loss = ImageLoss(im.moments(T(A)))
    if npz and os.path.isfile(npz):
        data = load_data(npz, im, None)
        m12, m3 = ImageLoss(data)(im.moments(T(A)))
        print(f"  2. floor vs {npz}: m12 {m12:.2e}  m3 {m3:.2e}   (image_checks, float64: see out/image_checks.log)")
    else:
        print(f"  2. floor: no data file {npz}; skipped")
    Ar = coef(x @ Rotation.random(random_state=1).as_matrix().T)
    Asc = A.copy()
    ls, _ = lm_index(L)
    Asc[:, ls % 2 == 1] = Ar[:, ls % 2 == 1]
    checks = (("rotation", Ar), ("reflection", coef(-x)), ("Kam scramble", Asc),
              ("x* + 0.5 A", coef(x + 0.5 * np.random.default_rng(0).standard_normal(x.shape))))
    for lab, Ax in checks:
        m12, m3 = loss(im.moments(T(Ax)))
        print(f"  3. {lab:<13} m12 {m12:.2e}  m3 {m3:.2e}")
    t1 = time.time()
    lossv = ImageLoss(im.moments(T(A)), im, all_variants=True)
    C0 = im.moments(T(A), third=False)["C"][: lossv.lw]
    eC = float(((C0 - lossv.Ct).abs().pow(2).sum() / C0.abs().pow(2).sum()).sqrt())
    print(f"  6. loss variants ({lossv.info}setup {time.time() - t1:.0f}s); C_l recovered from exact c2: rel err {eC:.1e}")
    print("     m12 plain / white,  m3 block / perq / white / wblock:")
    for lab, Ax in checks:
        v = {k: float(u) for k, u in lossv.variants(im.moments(T(Ax))).items()}
        print(f"     {lab:<13} m12 {v['m12_img']:.2e} / {v['m12w_img']:.2e}   "
              f"m3 {v['m3_img']:.2e} / {v['m3p_img']:.2e} / {v['m3w_img']:.2e} / {v['m3b_img']:.2e}")

    # gradient check in complex128 at a perturbed point, random complex direction
    im64 = ImageMoments(kappa, r_max, nb3=nb3, Q3=Q3, stride3=stride3, chunk=chunk, dev=dev, dtype=torch.float64)
    T64 = lambda a: torch.as_tensor(a, dtype=torch.complex128, device=dev)
    l64 = ImageLoss(im64.moments(T64(A)), im64, all_variants=True)
    rng = np.random.default_rng(1)
    scale = np.abs(A).mean(axis=1, keepdims=True)
    A1 = T64(A + 0.02 * scale * (rng.standard_normal(A.shape) + 1j * rng.standard_normal(A.shape)))
    dA = T64(scale * (rng.standard_normal(A.shape) + 1j * rng.standard_normal(A.shape)))
    for lab, key in (("m12", "m12_img"), ("m3", "m3_img"), ("m12 white", "m12w_img"), ("m3 perq", "m3p_img"), ("m3 white", "m3w_img"),
                     ("m3 wblock", "m3b_img")):
        F = lambda Ax: l64.variants(im64.moments(Ax))[key]
        A1g = A1.clone().requires_grad_(True)
        (g,) = torch.autograd.grad(F(A1g), A1g)
        ad = (g.conj() * dA).real.sum().item()
        with torch.no_grad():
            eps = 1e-5
            fd = (F(A1 + eps * dA) - F(A1 - eps * dA)).item() / (2 * eps)
        print(f"  4. gradient check {lab:<9}: autograd {ad:.6e}  finite diff {fd:.6e}  rel diff {abs(ad - fd) / abs(fd):.1e}")

    # timing / memory of the reward's image part (forward + backward), complex64
    Ag = T(A1.detach().cpu().numpy()).requires_grad_(True)
    for rep in range(2):
        if dev == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
        t1 = time.time()
        m12, m3 = loss(im.moments(Ag))
        (m12 * m3).backward()
        if dev == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t1
    mem = f", peak extra memory {(torch.cuda.max_memory_allocated() - base) / 2**30:.2f} GB" if dev == "cuda" else ""
    print(f"  5. forward + backward of m12 * m3 from A: {dt:.2f}s{mem}")

    # analytic kernel c3 = K G (on A truncated to l <= Lv) vs the quadrature, and visibility of the bispectrum
    from moments.c3_metric import bispectrum_quad, kernel, visible_fraction

    t1 = time.time()
    Lv = min(16, LB)
    c3q = ic.Model(L, Lv, nphi, nb3, Q3, shells3=shells3).moments(A)[0]["c3"]
    K, tr = kernel(Lv, Q3)
    Gv = bispectrum_quad(A, shells3, Lv)
    c3k = (K @ Gv[tr[:, 0], tr[:, 1], tr[:, 2]].reshape(len(tr), -1)).reshape(c3q.shape)
    print(f"  7. analytic kernel K G vs quadrature c3 (l <= {Lv}): rel err {np.linalg.norm(c3k - c3q) / np.linalg.norm(c3q):.1e}"
          f" ({time.time() - t1:.0f}s)")
    if LB >= 24:
        t1 = time.time()
        vf = visible_fraction(bispectrum_quad(A, shells3, 24), 24, Q3)
        print(f"  8. bispectrum l <= 24 on these shells visible in images at |q| <= {Q3}: raw {vf['raw']:.1%}, "
              f"block-normalized {vf['block-normalized']:.1%}  (rank {vf['rank']} of {vf['rows']} rows, "
              f"{vf['cols']} degree triples; {time.time() - t1:.0f}s)")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", default=["8AHU_A", "8K9Z_A", "8H1I_A"])
    ap.add_argument("--nb3", type=int, default=8)
    ap.add_argument("--q3", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--s3-stride", type=int, default=0, help="0: lowest nb3 shells; k: every k-th shell")
    a = ap.parse_args()
    for s in a.systems:
        _selftest(s, a.nb3, a.q3, a.chunk, a.s3_stride)

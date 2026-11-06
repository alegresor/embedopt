"""Unit checks for image-based moment matching (oracle particle, uniform poses, no noise / CTF yet).

Data side: N images of x* in uniformly random poses, rendered directly on a polar Fourier grid (the reward's shells
kappa_i x n_phi angles) from the atoms, I_j(kappa, phi) = Vhat[x*](kappa g_j e(phi)). With in-plane Fourier
coefficients I_jq(kappa) (FFT over phi), accumulate
    a(k)            = mean_j I_j0(k)                                         first moment (radial)
    c2_q(k1,k2)     = mean_j I_jq(k1) conj(I_jq(k2)),             q = 0..L    second moment
    c3_q1q2(a,b,c)  = mean_j I_jq1(a) I_jq2(b) I_j,-q1-q2(c),   |q| <= Q3   third moment (low shells)
Model side, from the invariants A_lm(kappa) of a structure (exact, no sampling):
    a = A_00 / sqrt(4 pi),   c2_q = (1/4pi) sum_{l>=|q|} beta_lq C_l    (beta_lq: Fourier coefficients of P_l(cos phi)),
    c3 by exact quadrature over viewing directions, using E^l_mq(beta) = in-plane Fourier coefficients of Y_lm on the
       great circle Rz(alpha) Ry(beta) e(phi), and Y_lm(Rz(alpha) v) = e^{i m alpha} Y_lm(v).
The third-order shells are the lowest NB3 shells (--s3-stride 0) or every k-th shell (--s3-stride k, as the invariant
bispectrum), with the FULL bandlimit there (data coefficients include all degrees). Non-default third-order grids get
the suffix _c3s<k>q<Q3> on the output files (image_checks*.json / .png, image_moments_N*.npz).

Checks (printed, and written to out/<SYS>/image_checks.json / .png):
  1. node machinery: image rendered from atoms at a quadrature node vs the E-route prediction (should be ~1e-7)
  2. convergence: relative error of data (a, c2, c3) vs model at N = 1e3, 1e4, 1e5, 1e6 (should fall ~ N^-1/2)
  3. triangular inversion: recover C_l from c2 by solving c2_q = (1/4pi) sum_l beta_lq C_l (per degree l)
  4. floor table: image-space losses m12, m3 of the exact x* model against the N-image data, vs the losses of
     x* + per-atom noise (sensitivity) and of a Kam scramble (odd degrees from a rotated copy)
Precision: rendering and per-batch products in float32 (TF32 disabled); invariants, model moments, accumulators,
inversion and losses in float64.

Usage: python -m moments.image_checks --systems 8AHU_A 8K9Z_A 8H1I_A [--nmax 1000000]
"""

import argparse
import json
import os
import time

import numpy as np
import scipy.special as sp
from numpy.polynomial.legendre import leggauss, legval
from scipy.spatial.transform import Rotation

from moments.invariants import form_factors

BENCH = "examples/synthetic_map_benchmark"
B_FACTOR, R_SAFETY, L_PAD = 50.0, 1.2, 8


# ----------------------------------------------------------------------------------------------- backend (torch / numpy)
class Backend:
    def __init__(self):
        try:
            import torch

            self.torch = torch if torch.cuda.is_available() else None
        except ImportError:
            self.torch = None
        if self.torch:
            self.torch.backends.cuda.matmul.allow_tf32 = self.torch.backends.cudnn.allow_tf32 = False
            self.dev = "cuda"
            t = self.torch
            self.f32, self.f64, self.c64, self.c128 = t.float32, t.float64, t.complex64, t.complex128
        else:
            self.f32, self.f64, self.c64, self.c128 = np.float32, np.float64, np.complex64, np.complex128
        self.name = "torch/cuda" if self.torch else "numpy/cpu"

    def array(self, a, dtype):
        if self.torch:
            return self.torch.as_tensor(np.asarray(a), dtype=dtype, device=self.dev)
        return np.asarray(a, dtype=dtype)

    def idx(self, a):
        return self.torch.as_tensor(np.asarray(a), dtype=self.torch.long, device=self.dev) if self.torch else np.asarray(a)

    def np(self, a):
        return a.detach().cpu().numpy() if self.torch else np.asarray(a)

    def einsum(self, eq, *ops):
        return (self.torch.einsum if self.torch else lambda e, *o: np.einsum(e, *o, optimize=True))(eq, *ops)

    def cos(self, a):
        return self.torch.cos(a) if self.torch else np.cos(a)

    def sin(self, a):
        return self.torch.sin(a) if self.torch else np.sin(a)

    def fft(self, a):
        return self.torch.fft.fft(a, dim=-1) if self.torch else np.fft.fft(a, axis=-1)

    def cast(self, a, dtype):
        return a.to(dtype) if self.torch else a.astype(dtype)

    def zeros(self, shape, dtype):
        return self.torch.zeros(shape, dtype=dtype, device=self.dev) if self.torch else np.zeros(shape, dtype=dtype)


BK = Backend()


# ----------------------------------------------------------------------------------------------- geometry and invariants
def sph_harm(l, m, theta, phi):
    return sp.sph_harm_y(l, m, theta, phi) if hasattr(sp, "sph_harm_y") else sp.sph_harm(m, l, phi, theta)


def lm_index(L):
    ls = np.concatenate([np.full(2 * l + 1, l) for l in range(L + 1)])
    ms = np.concatenate([np.arange(-l, l + 1) for l in range(L + 1)])
    return ls, ms


def s2_quadrature64(L):
    """Gauss-Legendre x trapezoid nodes on S^2, exact for degree <= 2L+1; returns omega, w * conj(Y) (float64)."""
    ct, wt = leggauss(L + 1)
    ph = 2 * np.pi * np.arange(2 * L + 2) / (2 * L + 2)
    T, P = [a.ravel() for a in np.meshgrid(np.arccos(ct), ph, indexing="ij")]
    w = (wt[:, None] * np.full(len(ph), 2 * np.pi / len(ph))).ravel()
    omega = np.stack([np.sin(T) * np.cos(P), np.sin(T) * np.sin(P), np.cos(T)], 1)
    ls, ms = lm_index(L)
    return omega, w[:, None] * sph_harm(ls[None], ms[None], T[:, None], P[:, None]).conj()


def coefficients64(x, rho, kappa, omega, WYc):
    """A_lm(kappa) in float64, one shell at a time. x (N,3) centered, rho (N, N_kappa)."""
    xb, om = BK.array(x, BK.f64), BK.array(omega, BK.f64)
    proj = 2 * np.pi * (xb @ om.T)
    V = []
    for i, k in enumerate(kappa):
        r = BK.array(rho[:, i], BK.f64)
        V.append(BK.np(r @ BK.cos(k * proj)) - 1j * BK.np(r @ BK.sin(k * proj)))
    return BK.np(BK.array(np.stack(V), BK.c128) @ BK.array(WYc, BK.c128))


class Model:
    """Exact image moments (a, c2, c3) from invariants A_lm (float64)."""

    def __init__(self, L, LB, nphi, NB3, Q3, shells3=None):
        self.L, self.LB, self.NB3, self.Q3 = L, LB, NB3, Q3
        self.shells3 = np.arange(NB3) if shells3 is None else np.asarray(shells3)
        phi = 2 * np.pi * np.arange(nphi) / nphi
        self.e = np.stack([np.cos(phi), np.sin(phi), np.zeros(nphi)], 1)
        # beta[l, q], q = 0..L: P_l(cos phi) = sum_q beta_lq e^{i q phi}
        self.beta = np.stack([np.fft.fft(legval(np.cos(phi), np.eye(L + 1)[l])) / nphi for l in range(L + 1)])[:, : L + 1].real
        # viewing-direction quadrature for the third order (integrand degree <= 3 LB)
        nb, self.na = (3 * LB) // 2 + 2, 3 * LB + 2
        cb, wb = leggauss(nb)
        self.bet, self.alp = np.arccos(cb), 2 * np.pi * np.arange(self.na) / self.na
        self.wn = np.repeat(wb, self.na) / (2 * self.na)  # normalized Haar weights (sum = 1)
        ls, ms = lm_index(LB)
        self.ms = ms
        self.qq = np.arange(-2 * Q3, 2 * Q3 + 1)
        E = np.zeros((nb, len(ls), len(self.qq)), complex)
        for k, b in enumerate(self.bet):  # E^l_mq(beta): in-plane Fourier coefficients of Y_lm on Ry(beta) e(phi)
            u = self.e @ Ry(b).T
            Yc = sph_harm(ls[None], ms[None], np.arccos(np.clip(u[:, 2], -1, 1))[:, None], np.arctan2(u[:, 1], u[:, 0])[:, None])
            E[k] = (np.fft.fft(Yc, axis=0) / nphi)[self.qq % nphi].T
        self.E = BK.array(E, BK.c128)
        mv = np.arange(-LB, LB + 1)
        self.sel_m = [BK.idx(np.flatnonzero(ms == m)) for m in mv]                # lm indices at fixed m
        self.phase = BK.array(np.exp(1j * np.outer(self.alp, mv)), BK.c128)    # e^{i m alpha}
        qs = np.arange(-Q3, Q3 + 1)
        self.i1, self.i3 = BK.idx(qs + 2 * Q3), BK.idx(-qs[:, None] - qs[None, :] + 2 * Q3)

    def node_images(self, A):
        """In-plane coefficients I_q(kappa) at every quadrature node, for the bispectrum shells: (nodes, NB3, q)."""
        Ab = BK.array(A[self.shells3, : (self.LB + 1) ** 2], BK.c128)
        # H[b, m, a, q] = sum_{l >= |m|} A_lm(a) E^l_mq(b), one m at a time (no (nb, lm, q, m) intermediate)
        H = BK.torch.stack if BK.torch else np.stack
        H = H([BK.einsum("an,bnq->baq", Ab[:, sel], self.E[:, sel]) for sel in self.sel_m], 1)  # (nb, m, NB3, q)
        I = BK.einsum("bmaq,tm->btaq", H, self.phase)                   # (nb, na, NB3, q)
        return I.reshape(-1, self.NB3, len(self.qq))

    def moments(self, A):
        a = A[:, 0] / np.sqrt(4 * np.pi)
        C = np.stack([A[:, l * l:(l + 1) ** 2] @ A[:, l * l:(l + 1) ** 2].conj().T for l in range(self.L + 1)])
        c2 = np.einsum("lq,lac->qac", self.beta, C) / (4 * np.pi)
        I = self.node_images(A)
        wn = BK.array(self.wn, BK.c128)
        c3 = 0
        for i in range(0, I.shape[0], 256):  # chunk over nodes: the (nodes, a, q, d, r) intermediate is large
            X, X3 = I[i:i + 256][:, :, self.i1], I[i:i + 256][:, :, self.i3]
            c3 = c3 + BK.einsum("n,naq,ndr,ncqr->qradc", wn[i:i + 256], X, X, X3)
        return dict(a=a, c2=c2, c3=BK.np(c3)), C


def Ry(b):
    return np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])


def Rz(a):
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


# ----------------------------------------------------------------------------------------------- data side
class Images:
    """Streams rendered images and accumulates (a, c2, c3) in float64; rendering in float32."""

    def __init__(self, x, rho, kappa, nphi, L, NB3, Q3, shells3=None):
        self.shells3 = BK.idx(np.arange(NB3) if shells3 is None else np.asarray(shells3))
        self.x, self.rho = BK.array(x, BK.f32), BK.array(rho, BK.f32)
        self.kappa, self.nphi, self.L, self.NB3, self.Q3 = kappa, nphi, L, NB3, Q3
        phi = 2 * np.pi * np.arange(nphi) / nphi
        self.e = BK.array(np.stack([np.cos(phi), np.sin(phi), np.zeros(nphi)], 1), BK.f32)
        qs = np.arange(-Q3, Q3 + 1)
        self.i1, self.i3 = BK.idx(qs % nphi), BK.idx((-qs[:, None] - qs[None, :]) % nphi)
        self.n = 0
        self.s1 = BK.zeros((len(kappa),), BK.c128)
        self.s2 = BK.zeros((L + 1, len(kappa), len(kappa)), BK.c128)
        self.s3 = BK.zeros((2 * Q3 + 1, 2 * Q3 + 1, NB3, NB3, NB3), BK.c128)

    def render(self, g):
        """g (B,3,3) float32 -> in-plane coefficients (B, N_kappa, nphi), complex64."""
        pr = BK.einsum("ni,bij,pj->bnp", self.x, g, self.e)  # e(phi) . g^T q_n
        V = [BK.einsum("n,bnp->bp", self.rho[:, i], BK.cos((2 * np.pi * k) * pr))
             - 1j * BK.einsum("n,bnp->bp", self.rho[:, i], BK.sin((2 * np.pi * k) * pr)) for i, k in enumerate(self.kappa)]
        V = BK.torch.stack(V, 1) if BK.torch else np.stack(V, 1)
        return BK.fft(V) / self.nphi

    def add(self, g):
        Iq = self.render(BK.array(g, BK.f32))
        s2 = BK.einsum("baq,bcq->qac", Iq[:, :, : self.L + 1], Iq[:, :, : self.L + 1].conj())
        Ib = Iq[:, self.shells3]
        s3 = BK.einsum("baq,bdr,bcqr->qradc", Ib[:, :, self.i1], Ib[:, :, self.i1], Ib[:, :, self.i3])
        self.s1 += BK.cast(Iq[:, :, 0].sum(0), BK.c128)
        self.s2 += BK.cast(s2, BK.c128)
        self.s3 += BK.cast(s3, BK.c128)
        self.n += len(g)

    def estimate(self):
        return dict(a=BK.np(self.s1) / self.n, c2=BK.np(self.s2) / self.n, c3=BK.np(self.s3) / self.n)


# ----------------------------------------------------------------------------------------------- losses
def losses(pred, ref):
    """Image-space losses, as the image-based reward will use them: m12 (relative, a + c2), m3 (block-normalized c3)."""
    m12 = (np.abs(pred["a"] - ref["a"]) ** 2).sum() / (np.abs(ref["a"]) ** 2).sum() \
        + (np.abs(pred["c2"] - ref["c2"]) ** 2).sum() / (np.abs(ref["c2"]) ** 2).sum()
    n = (np.abs(ref["c3"]) ** 2).sum(axis=(2, 3, 4))
    d = (np.abs(pred["c3"] - ref["c3"]) ** 2).sum(axis=(2, 3, 4))
    keep = n > 1e-12 * n.max()
    return float(m12), float((d[keep] / n[keep]).mean())


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


# ----------------------------------------------------------------------------------------------- main
def check_system(sys_, nmax, batch, resolution, n_kappa, nb3, Q3, out, stride3=0):
    import biotite.structure.io as bio

    from moments.image_moments import c3_tag, third_shells

    t0 = time.time()
    ref = bio.load_structure(f"{BENCH}/pdb_clean/{sys_}.pdb")
    ref = ref[ref.element != "H"]
    x = ref.coord.astype(np.float64)
    x -= x.mean(0)
    kappa = np.arange(1, n_kappa + 1) / (n_kappa * resolution)
    rmax = R_SAFETY * np.linalg.norm(x, axis=1).max()
    L = int(np.ceil(2 * np.pi * kappa[-1] * rmax)) + L_PAD
    shells3 = third_shells(n_kappa, nb3, stride3)
    tag = c3_tag(stride3, Q3)
    LB = min(L, int(np.ceil(2 * np.pi * kappa[shells3[-1]] * rmax)) + L_PAD)
    nphi = int(16 * np.ceil(2 * (L + 1) / 16))
    rho = form_factors(ref.element, kappa, B_FACTOR)
    print(f"\n=== {sys_}: {len(x)} atoms, L={L}, n_phi={nphi}, {n_kappa} shells to {resolution} A; "
          f"third order on shells {[int(v) for v in shells3 + 1]} (to {1 / kappa[shells3[-1]]:.1f} A), LB={LB}, |q|<={Q3}; backend {BK.name}")

    omega, WYc = s2_quadrature64(L)
    A = coefficients64(x, rho, kappa, omega, WYc)
    model = Model(L, LB, nphi, nb3, Q3, shells3=shells3)
    exact, C = model.moments(A)
    print(f"  invariants + model moments: {time.time() - t0:.0f}s")
    res = dict(system=sys_, L=L, LB=LB, nphi=nphi, Q3=Q3, nb3=nb3, shells3=shells3.tolist())

    # 1. node machinery vs rendering at one quadrature node
    imgs = Images(x, rho, kappa, nphi, L, nb3, Q3, shells3=shells3)
    kb, ka = len(model.bet) // 3, model.na // 5
    g = Rz(model.alp[ka]) @ Ry(model.bet[kb])
    Ir = BK.np(imgs.render(BK.array(g[None], BK.f32)))[0, shells3][:, model.qq % nphi]
    In = BK.np(model.node_images(A)).reshape(len(model.bet), model.na, nb3, -1)[kb, ka]
    res["node_vs_render"] = float(np.abs(In - Ir).max() / np.abs(Ir).max())
    print(f"  1. node vs render (float32 rendering): {res['node_vs_render']:.1e}")

    # 2. convergence; data snapshots at checkpoints
    checkpoints = sorted({n for n in (10**3, 10**4, 10**5, 10**6, 10**7, nmax) if n <= nmax})
    data, conv = {}, []
    rng_seed, done, t1 = 0, 0, time.time()
    for n in checkpoints:
        while done < n:
            b = min(batch, n - done)
            imgs.add(Rotation.random(b, random_state=rng_seed).as_matrix())
            rng_seed, done = rng_seed + 1, done + b
        data[n] = imgs.estimate()
        conv.append(dict(N=n, a=rel(data[n]["a"], exact["a"]), c2=rel(data[n]["c2"], exact["c2"]), c3=rel(data[n]["c3"], exact["c3"])))
        print(f"  2. N={n:>8}: rel err a {conv[-1]['a']:.2e}  c2 {conv[-1]['c2']:.2e}  c3 {conv[-1]['c3']:.2e}   "
              f"(N^-1/2 = {n**-0.5:.1e}; {time.time() - t1:.0f}s rendering so far)")
    res["convergence"] = conv
    np.savez(f"{out}/image_moments_N{checkpoints[-1]}{tag}.npz", **data[checkpoints[-1]], kappa=kappa, nb3=nb3, Q3=Q3, L=L,
             shells3=shells3)

    # 3. triangular inversion of c2 -> C_l (per degree)
    Bmat = model.beta.T / (4 * np.pi)  # c2_q = sum_l Bmat[q, l] C_l
    inv = {}
    for name, c2 in [("exact", exact["c2"])] + [(f"N={n}", data[n]["c2"]) for n in checkpoints[-2:]]:
        Ch = np.linalg.solve(Bmat, c2.reshape(L + 1, -1)).reshape(C.shape)
        inv[name] = [rel(Ch[l], C[l]) for l in range(L + 1)]
    res["inversion_cond"] = float(np.linalg.cond(Bmat))
    res["inversion_rel_err_per_l"] = inv
    ls_show = sorted({int(v) for v in np.linspace(0, L, 7)})
    print(f"  3. inversion c2 -> C_l (cond {res['inversion_cond']:.1e}); rel err at l = {ls_show}:")
    for name, e in inv.items():
        print(f"     {name:>10}: " + " ".join(f"{e[l]:.1e}" for l in ls_show))

    # 4. floor table and sensitivity
    rng = np.random.default_rng(0)
    floor = {n: losses(exact, data[n]) for n in checkpoints}
    sens = {}
    for s in (0.25, 0.5, 1.0, 2.0):
        As = coefficients64(x + s * rng.standard_normal(x.shape), rho, kappa, omega, WYc)
        sens[s] = losses(model.moments(As)[0], exact)
    Q = Rotation.random(random_state=1).as_matrix()
    Ar = coefficients64(x @ Q.T, rho, kappa, omega, WYc)
    Asc = A.copy()
    ls, _ = lm_index(L)
    Asc[:, ls % 2 == 1] = Ar[:, ls % 2 == 1]
    res["kam_scramble"] = losses(model.moments(Asc)[0], exact)
    res["rotation"] = losses(model.moments(Ar)[0], exact)
    res["floor"] = {str(n): v for n, v in floor.items()}
    res["sensitivity"] = {str(s): v for s, v in sens.items()}
    print("  4. floor (exact x* vs N-image data)       m12         m3")
    for n, (a, b) in floor.items():
        print(f"     N = {n:>8}                       {a:9.2e}  {b:9.2e}")
    print("     sensitivity (x* + noise vs exact x*)")
    for s, (a, b) in sens.items():
        print(f"     sigma = {s:4.2f} A                     {a:9.2e}  {b:9.2e}")
    print(f"     rotation of x*                      {res['rotation'][0]:9.2e}  {res['rotation'][1]:9.2e}")
    print(f"     Kam scramble                        {res['kam_scramble'][0]:9.2e}  {res['kam_scramble'][1]:9.2e}")
    json.dump(res, open(f"{out}/image_checks{tag}.json", "w"), indent=1)
    plot(res, f"{out}/image_checks{tag}.png")
    print(f"  total {time.time() - t0:.0f}s; wrote {out}/image_checks{tag}.json, .png, image_moments_N{checkpoints[-1]}{tag}.npz")


def plot(res, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    N = np.array([c["N"] for c in res["convergence"]])
    for k in ("a", "c2", "c3"):
        ax[0].loglog(N, [c[k] for c in res["convergence"]], "o-", label=k)
    ax[0].loglog(N, N**-0.5 * res["convergence"][0]["c2"] * N[0] ** 0.5, "k:", label="N^-1/2")
    ax[0].set(title=f"{res['system']}: data vs model, relative error", xlabel="images N")
    for name, e in res["inversion_rel_err_per_l"].items():
        ax[1].semilogy(e, label=name)
    ax[1].set(title=f"c2 -> C_l inversion error (cond {res['inversion_cond']:.0e})", xlabel="degree l")
    fl = res["floor"]
    for j, nm in enumerate(("m12", "m3")):
        ax[2].loglog([int(n) for n in fl], [v[j] for v in fl.values()], "o-", label=f"floor {nm}")
        for s, v in res["sensitivity"].items():
            ax[2].axhline(v[j], color=f"C{j}", ls=":", lw=0.8)
    ax[2].set(title="floor (exact x* vs data) and x*+noise levels (dotted)", xlabel="images N")
    for a in ax:
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", default=["8AHU_A", "8K9Z_A", "8H1I_A"])
    ap.add_argument("--nmax", type=int, default=10**6)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--resolution", type=float, default=5.0)
    ap.add_argument("--n-kappa", type=int, default=16)
    ap.add_argument("--nb3", type=int, default=8, help="lowest shells used for the third order")
    ap.add_argument("--q3", type=int, default=16)
    ap.add_argument("--s3-stride", type=int, default=0, help="0: lowest nb3 shells; k: every k-th shell")
    a = ap.parse_args()
    for s in a.systems:
        out = f"out/{s}"
        os.makedirs(out, exist_ok=True)
        check_system(s, a.nmax, a.batch, a.resolution, a.n_kappa, a.nb3, a.q3, out, a.s3_stride)

"""The third image moment as a linear image of the bispectrum, and the metrics on c3 derived from it.

Under uniform poses, for every shell triple (a, b, c) (same kernel for all triples):
    c3_{q1 q2}(a, b, c) = sum_{l1 l2 l3} K[(q1, q2), (l1, l2, l3)] G_{l1 l2 l3}(a, b, c),           q3 = -q1 - q2,
    K = (l1 l2 l3; q1 q2 q3) Ybar_{l1 q1} Ybar_{l2 q2} Ybar_{l3 q3} / (c_l (l1 l2 l3; 0 0 0)),
with G_{l1 l2 l3}(a,b,c) = int_{S^2} f_l1(a) f_l2(b) f_l3(c) (the bispectrum of moments.invariants, Gaunt form),
c_l = sqrt((2l1+1)(2l2+1)(2l3+1) / 4pi), Ybar_lq = Y_lq(pi/2, 0) (zero unless l + q even). Derivation: the Haar integral
of three Wigner-D matrices is |3j><3j|, and Y_ln on the equator is Ybar_ln e^{i n phi}. Only even l1 + l2 + l3 appear.
K is computed through Gaunt integrals: (l; q) Ybar.. / (c_l (l; 000)) = Gaunt(l; q) Ybar.. / (c_l (l; 000))^2, with
Gaunt(l; q) = 2 pi int_{-1}^{1} P_l1q1 P_l2q2 P_l3q3 dx (Gauss-Legendre; P_lq(cos theta) = Y_lq(theta, 0)).

K has far more columns (degree triples) than rows (in-plane pairs), so only the part of G in the row space of K is
visible in images. The metrics on c3 (rows flattened as q1 * (2Q3+1) + q2, both from -Q3):
    white: |c|_W^2 = c^H (K K^T)^+ c = |K^+ c|^2, the norm of the minimum-norm bispectrum estimate (rcond cutoff);
    perq : the diagonal of K K^T, w_r = 1 / (K K^T)_rr.
The metric depends only on (LB, Q3) and is cached in $MOMENT_CACHE (default out/cache).
"""

import os
import time
from math import lgamma, pi

import numpy as np
from numpy.polynomial.legendre import leggauss

from moments.invariants import sph_harm


def valid_triples(LB):
    """All (l1, l2, l3) <= LB with the triangle rule and even sum; float64 log of 1 / (c_l (l; 000))^2."""
    l = np.arange(LB + 1)
    L1, L2, L3 = np.meshgrid(l, l, l, indexing="ij")
    ok = (L3 >= abs(L1 - L2)) & (L3 <= L1 + L2) & ((L1 + L2 + L3) % 2 == 0)
    t = np.stack([L1[ok], L2[ok], L3[ok]], 1)
    lg = np.vectorize(lgamma)
    g = t.sum(1) // 2
    # (l; 000)^2 = (2g-2l1)!(2g-2l2)!(2g-2l3)!/(2g+1)! * [g!/((g-l1)!(g-l2)!(g-l3)!)]^2
    log3j2 = (lg(2 * g - 2 * t[:, 0] + 1) + lg(2 * g - 2 * t[:, 1] + 1) + lg(2 * g - 2 * t[:, 2] + 1) - lg(2 * g + 2)
              + 2 * (lg(g + 1) - lg(g - t[:, 0] + 1) - lg(g - t[:, 1] + 1) - lg(g - t[:, 2] + 1)))
    logc2 = np.log((2 * t + 1).prod(1) / (4 * pi))
    return t, -(logc2 + log3j2)


def kernel(LB, Q3, triples=None, chunk=64, verbose=False):
    """K (rows (2Q3+1)^2, columns = triples), float64. Rows: (q1, q2) in [-Q3, Q3]^2, q3 = -q1 - q2."""
    t, loginvD = valid_triples(LB) if triples is None else triples
    invD = np.exp(loginvD)
    n = (3 * LB) // 2 + 2
    x, w = leggauss(n)
    qq = np.arange(-2 * Q3, 2 * Q3 + 1)
    l = np.arange(LB + 1)
    # P[q, l, j] = Y_lq(theta_j, 0); Ybar[q, l] = Y_lq(pi/2, 0)
    P = sph_harm(l[None, :, None], qq[:, None, None], np.arccos(x)[None, None, :], 0.0).real
    Ybar = sph_harm(l[None, :], qq[:, None], np.pi / 2, 0.0).real
    P = np.where(l[None, :, None] >= np.abs(qq)[:, None, None], P, 0.0)
    Ybar = np.where(l[None, :] >= np.abs(qq)[:, None], Ybar, 0.0)
    qs = np.arange(-Q3, Q3 + 1)
    rows = [(a, b) for a in qs for b in qs]
    K = np.zeros((len(rows), len(t)))
    flat = (t[:, 0] * (LB + 1) + t[:, 1]) * (LB + 1) + t[:, 2]
    t0 = time.time()
    for r, (q1, q2) in enumerate(rows):
        i1, i2, i3 = q1 + 2 * Q3, q2 + 2 * Q3, -q1 - q2 + 2 * Q3
        X = (P[i1][:, None, :] * P[i2][None, :, :]).reshape(-1, n)               # (l1 l2, j)
        gaunt = (2 * pi) * (X @ (P[i3] * w).T).reshape(-1)                       # (l1 l2 l3)
        yb = (Ybar[i1][:, None, None] * Ybar[i2][None, :, None] * Ybar[i3][None, None, :]).reshape(-1)
        K[r] = gaunt[flat] * yb[flat] * invD
        if verbose and r % 500 == 0:
            print(f"    kernel row {r}/{len(rows)} ({time.time() - t0:.0f}s)", flush=True)
    return K, t


def metric(LB, Q3, rcond=1e-6, cache=None, verbose=True):
    """Eigen-decomposition of K K^T (cached): dict(lam, U, diag, rank) with lam / U the kept part (lam > rcond max)."""
    cache = cache or os.environ.get("MOMENT_CACHE", "out/cache")
    path = os.path.join(cache, f"c3_metric_LB{LB}_Q{Q3}.npz")
    if os.path.isfile(path):
        d = np.load(path)
        lam_all, U_all, diag, ncol = d["lam"], d["U"], d["diag"], int(d["ncol"])
    else:
        t0 = time.time()
        K, t = kernel(LB, Q3, verbose=verbose)
        Gm = K @ K.T
        lam_all, U_all = np.linalg.eigh(Gm)
        diag, ncol = np.diag(Gm).copy(), K.shape[1]
        os.makedirs(cache, exist_ok=True)
        np.savez(path, lam=lam_all, U=U_all, diag=diag, ncol=ncol)
        if verbose:
            print(f"    c3 metric LB={LB} Q3={Q3}: {K.shape[0]} rows x {ncol} degree triples, {time.time() - t0:.0f}s -> {path}")
    keep = lam_all > rcond * lam_all.max()
    return dict(lam=lam_all[keep], U=U_all[:, keep], diag=diag, rank=int(keep.sum()), rows=len(diag), ncol=ncol,
                cond=float(lam_all[keep].max() / lam_all[keep].min()))


def block_metric(LB, Q3, c3_rows, M, rcond=1e-9, cache=None):
    """Whitening factor for the block-normalized bispectrum. c3_rows: data c3 as (rows, shell triples), complex128;
    M: metric(LB, Q3). Returns (Wb (rows, rank), number of visible degree triples) with |Wb^T c|^2 = |K'^+ c|^2,
    K' = K diag(s), s_l = |B^_l| (minimum-norm estimate B^ = K^+ c3 of the data, norm over shell triples).
    Cached in $MOMENT_CACHE by (LB, Q3, rcond, hash of the data)."""
    import hashlib

    cache = cache or os.environ.get("MOMENT_CACHE", "out/cache")
    h = hashlib.sha1(np.ascontiguousarray(c3_rows).tobytes()).hexdigest()[:16]
    path = os.path.join(cache, f"c3_wblock_LB{LB}_Q{Q3}_rc{rcond:g}_{h}.npz")
    if os.path.isfile(path):
        d = np.load(path)
        return d["Wb"], int(d["nvis"])
    K, _ = kernel(LB, Q3)
    s = np.linalg.norm(K.T @ (M["U"] @ ((M["U"].T @ c3_rows) / M["lam"][:, None])), axis=1)  # |B^_l| over shells
    s = np.where(s > 1e-8 * s.max(), s, 0.0)
    K *= s[None, :]                                                                # K' in place (K can be ~3 GB)
    lam, U = np.linalg.eigh(K @ K.T)
    del K
    keep = lam > rcond * lam.max()
    Wb, nvis = U[:, keep] / np.sqrt(lam[keep]), int((s > 0).sum())
    os.makedirs(cache, exist_ok=True)
    np.savez(path, Wb=Wb, nvis=nvis)
    return Wb, nvis


def bispectrum_quad(A, shells, Lv):
    """G_{l1 l2 l3}(a, b, c) for l <= Lv on the given shells, by S^2 quadrature exact for degree 3 Lv (complex128)."""
    nt, nph = (3 * Lv) // 2 + 2, 3 * Lv + 2
    x, wx = leggauss(nt)
    th, ph = np.meshgrid(np.arccos(x), 2 * np.pi * np.arange(nph) / nph, indexing="ij")
    w = (np.outer(wx, np.ones(nph)) * 2 * np.pi / nph).ravel()
    ls = np.concatenate([np.full(2 * l + 1, l) for l in range(Lv + 1)])
    ms = np.concatenate([np.arange(-l, l + 1) for l in range(Lv + 1)])
    Y = sph_harm(ls[None], ms[None], th.ravel()[:, None], ph.ravel()[:, None])     # (nodes, lm)
    A = np.asarray(A)[shells, : (Lv + 1) ** 2]
    f = np.zeros((len(shells), Lv + 1, len(w)), complex)
    for l in range(Lv + 1):
        f[:, l] = A[:, l * l:(l + 1) ** 2] @ Y[:, l * l:(l + 1) ** 2].T
    F = f.reshape(-1, len(w))                                                     # (shell l, node)
    Fw = (F * w).T
    G = np.concatenate([((F[i:i + 16, None, :] * F[None, :, :]).reshape(-1, len(w)) @ Fw) for i in range(0, len(F), 16)])
    G = G.reshape(len(shells), Lv + 1, len(shells), Lv + 1, len(shells), Lv + 1)
    return G.transpose(1, 3, 5, 0, 2, 4)                                          # (l1, l2, l3, a, b, c)


def visible_fraction(G, LB, Q3, rcond=1e-10):
    """Fraction of |G|^2 (l <= LB part) in the row space of K at |q| <= Q3: raw, and with each degree triple
    normalized to unit norm over shell triples (the weighting of the invariant m3)."""
    K, t = kernel(LB, Q3)
    Gv = G[t[:, 0], t[:, 1], t[:, 2]].reshape(len(t), -1)                         # (triples, shell triples)
    lam, U = np.linalg.eigh(K @ K.T)
    keep = lam > rcond * lam.max()
    U, lam = U[:, keep], lam[keep]
    out = {}
    for name, g in (("raw", Gv), ("block-normalized", Gv / np.maximum(np.linalg.norm(Gv, axis=1, keepdims=True), 1e-300))):
        nz = np.linalg.norm(g, axis=1) > 0
        z = (U.T @ (K @ g)) / np.sqrt(lam)[:, None]
        out[name] = float((np.abs(z) ** 2).sum() / (np.abs(g[nz]) ** 2).sum())
    out["rank"], out["rows"], out["cols"] = int(keep.sum()), K.shape[0], K.shape[1]
    return out

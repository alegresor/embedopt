"""Image moments under a non-uniform viewing-direction density, and its fit (writeup, "Unknown pose density").

Density. Poses g = R_z(alpha) R_y(beta) R_z(gamma), gamma uniform, viewing direction n(alpha, beta) with density nu
(mean 1 over S^2). Parameterization (even degrees, real, b_00 = 1):
    nu_b(alpha, beta) = sum_{p even <= P} sum_{u=-p..p} b_pu Yt_pu(beta, alpha),   Yt_pu = sqrt(4 pi) Y_p^u,
    b_{p,-u} = (-1)^u conj(b_pu),  real parameters theta: b_p0 (p >= 2) and Re/Im b_pu (u > 0): 14 at P = 4, 27 at P = 6.
nu is affine in theta, nu = 1 + sum_k theta_k phi_k.

Moments, exactly, by quadrature, with the spherical-harmonic coefficients A_lm(kappa) of the structure:
  F_m(kappa, beta, q) = sum_l A_lm(kappa) E^l_mq(beta)    (E: in-plane Fourier coefficients of Y_lm on the tilted circle)
  a(kappa)            = sum_beta w_beta sum_m nu_{-m}(beta) F_m(kappa, beta, 0)
  c2_q(k1, k2)        = sum_beta w_beta sum_{m, u} nu_u(beta) F_m(k1, beta, q) conj F_{m+u}(k2, beta, q)
  c3_{q1 q2}(a, b, c) = sum_{nodes (alpha, beta)} w nu(n) I_q1(a) I_q2(b) I_q3(c),  I = sum_m e^{i m alpha} F_m
with nu_u(beta) the alpha-Fourier coefficients of nu (|u| <= P for the parameterized density). w_beta: Gauss-Legendre
weights / 2 in cos(beta); for c3, w = w_beta / n_alpha on a uniform alpha grid. Quadrature sizes include the degree P
of nu, so the formulas are exact for a band-limited nu.

All moments are linear in nu, hence affine in theta: m(theta) = m_0 + sum_k theta_k m_k. For a loss that is a quadratic
form in the moments (m12 and m3 are), the loss is a convex quadratic in theta: fit_theta evaluates it on the basis to
get the exact Hessian and gradient and solves the QP with nonnegativity of nu at collocation points (SLSQP).

Backend: every function works with numpy arrays or torch tensors (the reward uses torch on the GPU, the tests numpy).
"""

import numpy as np
from numpy.polynomial.legendre import leggauss


# ------------------------------------------------------------------------------------------------ helpers
def _sph(l, m, polar, azimuth):
    try:
        from scipy.special import sph_harm_y

        return sph_harm_y(l, m, polar, azimuth)
    except ImportError:  # older scipy
        from scipy.special import sph_harm

        return sph_harm(m, l, azimuth, polar)


def lm_index(L):
    ls = np.concatenate([np.full(2 * l + 1, l) for l in range(L + 1)])
    ms = np.concatenate([np.arange(-l, l + 1) for l in range(L + 1)])
    return ls, ms


def is_torch(x):
    return type(x).__module__.startswith("torch")


def _xp(x):
    if is_torch(x):
        import torch

        return torch
    return np


def _conj(x):
    return x.conj()


def _as(x, like):
    """numpy array x (complex) -> the backend, complex dtype and device of `like`."""
    if is_torch(like):
        import torch

        return torch.as_tensor(np.asarray(x), dtype=like.dtype, device=like.device)
    return np.asarray(x, dtype=like.dtype)


# ------------------------------------------------------------------------------------------------ the density
class SHDensity:
    """nu_theta = 1 + sum_k theta_k phi_k with real even spherical harmonics phi_k up to degree P (b_00 = 1)."""

    def __init__(self, P=4):
        self.P = P
        self.terms = []                                        # (p, u, kind): kind 0 = b_p0, 1 = Re b_pu, 2 = Im b_pu
        for p in range(2, P + 1, 2):
            self.terms.append((p, 0, 0))
            for u in range(1, p + 1):
                self.terms += [(p, u, 1), (p, u, 2)]
        self.K = len(self.terms)

    def basis_values(self, n):
        """phi_k at unit vectors n (..., 3): (K, ...)."""
        polar = np.arccos(np.clip(n[..., 2], -1, 1))
        az = np.arctan2(n[..., 1], n[..., 0])
        out = []
        for p, u, kind in self.terms:
            Y = np.sqrt(4 * np.pi) * _sph(p, u, polar, az)
            out.append(Y.real if kind == 0 else (2 * Y.real if kind == 1 else -2 * Y.imag))
        return np.stack(out)

    def basis_fourier(self, betas, U):
        """alpha-Fourier coefficients of phi_k at polar angles betas: (K, len(betas), 2U+1), u = -U..U."""
        out = np.zeros((self.K, len(betas), 2 * U + 1), complex)
        for k, (p, u, kind) in enumerate(self.terms):
            y = np.sqrt(4 * np.pi) * _sph(p, u, betas, np.zeros_like(betas))     # Yt_pu(beta, 0), real
            if kind == 0:
                out[k, :, U] = y
            elif kind == 1:                                                     # Yt_pu + conj Yt_pu
                out[k, :, U + u] += y
                out[k, :, U - u] += np.conj(y)
            else:                                                               # i (Yt_pu - conj Yt_pu)
                out[k, :, U + u] += 1j * y
                out[k, :, U - u] += -1j * np.conj(y)
        return out

    def values(self, theta, n):
        return 1 + np.tensordot(np.asarray(theta), self.basis_values(n), 1)

    def project(self, fn, nb=40, na=80):
        """Least-squares coefficients of a density fn(n) (mean 1) on the basis (orthogonal projection)."""
        cb, wb = leggauss(nb)
        al = 2 * np.pi * np.arange(na) / na
        n = dirs(np.arccos(cb)[:, None], al[None, :])
        W = (wb[:, None] / 2 / na)
        Phi = self.basis_values(n)                                              # (K, nb, na)
        G = np.einsum("kab,lab,ab->kl", Phi, Phi, np.broadcast_to(W, Phi.shape[1:]))
        r = np.einsum("kab,ab,ab->k", Phi, fn(n) - 1, np.broadcast_to(W, Phi.shape[1:]))
        return np.linalg.solve(G, r)


def dirs(beta, alpha):
    beta, alpha = np.broadcast_arrays(beta, alpha)
    return np.stack([np.sin(beta) * np.cos(alpha), np.sin(beta) * np.sin(alpha), np.cos(beta)], -1)


def fourier_of(fn, betas, U, na=256):
    """alpha-Fourier coefficients (len(betas), 2U+1) of an arbitrary density fn(n) at polar angles betas."""
    al = 2 * np.pi * np.arange(na) / na
    v = fn(dirs(betas[:, None], al[None, :]))
    F = np.fft.fft(v, axis=1) / na
    return F[:, np.arange(-U, U + 1) % na]


# ------------------------------------------------------------------------------------------------ tables
def e_tables(Lmax, nb, nphi, qs):
    """E^l_mq(beta_j) for l <= Lmax, all m, q in qs: (2 Lmax + 1, Lmax + 1, nb, len(qs)) complex, and the GL nodes."""
    cb, wb = leggauss(nb)
    phi = 2 * np.pi * np.arange(nphi) / nphi
    e = np.stack([np.cos(phi), np.sin(phi), np.zeros(nphi)], 1)
    ls, ms = lm_index(Lmax)
    E = np.zeros((2 * Lmax + 1, Lmax + 1, nb, len(qs)), complex)
    for k, b in enumerate(np.arccos(cb)):
        u = e @ np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]]).T
        Y = _sph(ls[None], ms[None], np.arccos(np.clip(u[:, 2], -1, 1))[:, None], np.arctan2(u[:, 1], u[:, 0])[:, None])
        E[ms + Lmax, ls, k] = (np.fft.fft(Y, axis=0) / nphi)[np.asarray(qs) % nphi].T
    return E, np.arccos(cb), wb / 2


class PoseMoments:
    """Moments (a, c2, c3) of a structure's A_lm under a viewing density, on the reward's image grid.

    L, nphi, kappa as the uniform-pose grid; shells3, Q3, LB as its third-order part; P = max degree of nu handled
    exactly (for an arbitrary density, its alpha-Fourier coefficients are truncated at U = P)."""

    def __init__(self, L, LB, nphi, NK, shells3, Q3, P=4, dtype=np.complex128, backend="numpy", device="cpu"):
        self.L, self.LB, self.nphi, self.NK, self.shells3, self.Q3, self.P = L, LB, nphi, NK, np.asarray(shells3), Q3, P
        self.U = P
        nb2 = L + P // 2 + 2                                     # c2 integrand: degree <= 2L + P in cos(beta)
        E2, self.b2, self.w2 = e_tables(L, nb2, nphi, np.arange(L + 1))
        nb3, na3 = (3 * LB + P) // 2 + 2, 3 * LB + P + 2
        qq = np.arange(-2 * Q3, 2 * Q3 + 1)
        E3, self.b3, w3b = e_tables(LB, nb3, nphi, qq)
        self.al3 = 2 * np.pi * np.arange(na3) / na3
        self.w3 = np.broadcast_to(w3b[None, :] / na3, (na3, nb3)).ravel()           # node = alpha index * nb3 + beta
        self.n3 = dirs(np.broadcast_to(self.b3[None, :], (na3, nb3)), np.broadcast_to(self.al3[:, None], (na3, nb3)))
        self.n3 = self.n3.reshape(-1, 3)
        ls, ms = lm_index(L)
        self.idx2 = np.zeros((2 * L + 1, L + 1), int); self.idx2[ms + L, ls] = np.arange(len(ls))
        self.mask2 = np.zeros((2 * L + 1, L + 1)); self.mask2[ms + L, ls] = 1
        ls3, ms3 = lm_index(LB)
        self.idx3 = np.zeros((2 * LB + 1, LB + 1), int); self.idx3[ms3 + LB, ls3] = np.arange(len(ls3))
        self.mask3 = np.zeros((2 * LB + 1, LB + 1)); self.mask3[ms3 + LB, ls3] = 1
        self.phase3 = np.exp(1j * np.outer(self.al3, np.arange(-LB, LB + 1)))
        qs = np.arange(-Q3, Q3 + 1)
        self.i1, self.i3 = qs + 2 * Q3, -qs[:, None] - qs[None, :] + 2 * Q3
        self.nb3, self.na3 = nb3, na3
        if backend == "torch":
            import torch

            cdt = torch.complex64 if dtype in (np.complex64, "complex64") else torch.complex128
            rdt = torch.float32 if cdt == torch.complex64 else torch.float64
            t = lambda x, d=cdt: torch.as_tensor(np.asarray(x), dtype=d, device=device)
            self.E2 = t(E2).reshape(E2.shape[0], E2.shape[1], -1)
            self.E3 = t(E3).reshape(E3.shape[0], E3.shape[1], -1)
            self.phase3_t = t(self.phase3)
            self.m2_t, self.m3_t = t(self.mask2), t(self.mask3)
            self.i2_t = torch.as_tensor(self.idx2, device=device); self.i3_idx = torch.as_tensor(self.idx3, device=device)
            self.sh3_t = torch.as_tensor(self.shells3, device=device)
            self.i1_t, self.i3_t = torch.as_tensor(self.i1, device=device), torch.as_tensor(self.i3, device=device)
            self.w2_t, self.w3_t = t(self.w2, rdt), t(self.w3, rdt)
            self.torch = True
        else:
            self.E2 = E2.astype(dtype).reshape(E2.shape[0], E2.shape[1], -1)
            self.E3 = E3.astype(dtype).reshape(E3.shape[0], E3.shape[1], -1)
            self.torch = False

    # -------------------------------------------------------------- pieces shared by all densities
    def F2(self, A):
        """F_m(k, beta, q), q = 0..L: (2L+1, NK, nb2, L+1)."""
        if self.torch:
            Ap = A[:, self.i2_t] * self.m2_t                                     # (k, m, l)
        else:
            Ap = A[:, self.idx2] * self.mask2
        H = Ap.transpose(0, 1) if self.torch else Ap.transpose(1, 0, 2)
        H = H @ self.E2                                                           # (m, k, nb2 * (L+1))
        return H.reshape(2 * self.L + 1, self.NK, len(self.b2), self.L + 1)

    def node_images3(self, A):
        """I_q(a) at the c3 nodes: (nodes, S, 4 Q3 + 1)."""
        if self.torch:
            Ap = A[self.sh3_t[:, None, None], self.i3_idx[None]] * self.m3_t
            H = Ap.transpose(0, 1) @ self.E3
            I = (self.phase3_t @ H.reshape(H.shape[0], -1))
        else:
            Ap = A[self.shells3[:, None, None], self.idx3[None]] * self.mask3
            H = Ap.transpose(1, 0, 2) @ self.E3
            I = self.phase3 @ H.reshape(H.shape[0], -1)
        S = len(self.shells3)
        I = I.reshape(self.na3, S, self.nb3, -1)
        I = I.permute(0, 2, 1, 3) if self.torch else I.transpose(0, 2, 1, 3)
        return I.reshape(self.na3 * self.nb3, S, -1)

    def pair_products(self, F):
        """Sigma_u(beta, q, k1, k2) = sum_m F_m(k1) conj F_{m+u}(k2), u = -U..U: (2U+1, nb2, L+1, NK, NK)."""
        xp = _xp(F)
        M = F.shape[0]
        out = []
        for u in range(-self.U, self.U + 1):
            lo, hi = max(0, -u), min(M, M - u)
            out.append(xp.einsum("mkbq,mjbq->bqkj", F[lo:hi], _conj(F[lo + u:hi + u])))
        return xp.stack(out)

    # -------------------------------------------------------------- moments for given density coefficients
    def moments(self, A, nu_fourier2, nu_nodes3, third=True, cache=None):
        """nu_fourier2: alpha-Fourier coefficients at the c2 beta nodes (nb2, 2U+1); nu_nodes3: values at the c3 nodes.
        cache (optional dict) keeps F, the pair products and the c3 node images for reuse with other densities."""
        xp = _xp(A)
        c = {} if cache is None else cache
        if "F" not in c:
            c["F"] = self.F2(A)
            c["S"] = self.pair_products(c["F"])
            if third:
                c["I3"] = self.node_images3(A)
        F, S = c["F"], c["S"]
        cast = lambda x: _as(np.asarray(x), F)                                     # to F's backend / complex dtype
        nf = cast(np.asarray(nu_fourier2, complex))                               # (nb2, 2U+1)
        w2 = cast(self.w2.astype(complex))
        U, L = self.U, self.L
        ms = np.arange(-U, U + 1)
        Fm0 = F[ms + L][:, :, :, 0]                                               # (2U+1, NK, nb2): m = -U..U, q = 0
        a = xp.einsum("ukb,bu,b->k", Fm0, nf[:, (-ms) + U], w2)                   # sum_m nu_{-m} F_m(., 0)
        c2 = xp.einsum("ubqkj,bu,b->qkj", S, nf, w2)
        out = dict(a=a, c2=c2, c3=None)
        if third:
            out["c3"] = self.c3_from(c["I3"], cast((self.w3 * np.asarray(nu_nodes3, float)).astype(complex)))
        return out

    def c3_from(self, I, w, chunk=256):
        xp = _xp(I)
        out = 0
        i1 = self.i1_t if self.torch else self.i1
        i3 = self.i3_t if self.torch else self.i3
        f = lambda X, ww: xp.einsum("n,naq,nbr,ncqr->qrabc", ww, X[:, :, i1], X[:, :, i1], X[:, :, i3])
        for s in range(0, I.shape[0], chunk):
            X, ww = I[s:s + chunk], w[s:s + chunk]
            if self.torch and X.requires_grad:
                from torch.utils.checkpoint import checkpoint

                out = out + checkpoint(f, X, ww, use_reentrant=False)          # memory: recompute in backward
            else:
                out = out + f(X, ww)
        return out

    # -------------------------------------------------------------- densities
    def uniform(self):
        nf = np.zeros((len(self.b2), 2 * self.U + 1)); nf[:, self.U] = 1
        return nf, np.ones(len(self.w3))

    def of_function(self, fn):
        """An arbitrary density (callable on unit vectors), alpha-Fourier truncated at U = P."""
        return fourier_of(fn, self.b2, self.U), fn(self.n3)

    def of_theta(self, dens, theta):
        nf0, nn0 = self.uniform()
        Bf = dens.basis_fourier(self.b2, self.U)                                  # (K, nb2, 2U+1)
        Bn = dens.basis_values(self.n3)                                           # (K, nodes)
        th = np.asarray(theta, float)
        return nf0 + np.tensordot(th, Bf, 1), nn0 + th @ Bn

    def basis_moments(self, A, dens, third=True, cache=None):
        """[m_0 (uniform), m_1, ..., m_K] with m(theta) = m_0 + sum_k theta_k m_k; no gradients needed."""
        c = {} if cache is None else cache
        Bf = dens.basis_fourier(self.b2, self.U)
        Bn = dens.basis_values(self.n3)
        out = [self.moments(A, *self.uniform(), third=third, cache=c)]
        for k in range(dens.K):
            out.append(self.moments(A, Bf[k], Bn[k], third=third, cache=c))
        return out


# ------------------------------------------------------------------------------------------------ the QP
def combine(basis, theta):
    th = [1.0] + [float(t) for t in theta]
    return {key: (sum(t * b[key] for t, b in zip(th, basis)) if basis[0][key] is not None else None)
            for key in basis[0]}


def fit_theta(basis, loss, dens, colloc, theta0=None, ridge=0.0):
    """Minimize loss(combine(basis, theta)) over theta, a convex quadratic, subject to nu_theta >= 0 at the
    collocation directions colloc (n_s, 3). loss: moments dict -> float (a quadratic form in the moments, e.g.
    m12 / m12_0 + m3 / m3_0). The Hessian and gradient are obtained exactly from (K+1)(K+2)/2 loss evaluations."""
    from scipy.optimize import minimize

    K = len(basis) - 1
    ev = lambda th: float(loss(combine(basis, th)))
    J0 = ev(np.zeros(K))
    e = np.eye(K)
    Jk = np.array([ev(e[k]) for k in range(K)])
    Jm = np.array([ev(-e[k]) for k in range(K)])
    g = (Jk - Jm) / 2                                                             # exact for a quadratic
    H = np.zeros((K, K))
    for i in range(K):
        H[i, i] = Jk[i] + Jm[i] - 2 * J0
        for j in range(i + 1, K):
            H[i, j] = H[j, i] = (ev(e[i] + e[j]) - Jk[i] - Jk[j] + J0)
    H = (H + H.T) / 2 + ridge * np.eye(K)
    Phi = dens.basis_values(colloc).T                                             # (n_s, K)
    obj = lambda th: J0 + g @ th + 0.5 * th @ H @ th
    jac = lambda th: g + H @ th
    cons = dict(type="ineq", fun=lambda th: 1 + Phi @ th, jac=lambda th: Phi)
    x0 = np.zeros(K) if theta0 is None else np.asarray(theta0, float)
    r = minimize(obj, x0, jac=jac, constraints=[cons], method="SLSQP", options=dict(maxiter=200, ftol=1e-12))
    return r.x, dict(loss0=J0, loss=float(r.fun), success=bool(r.success), min_nu=float((1 + Phi @ r.x).min()))


def collocation(n=400, seed=0):
    """Quasi-uniform directions on S^2 (Fibonacci sphere) for the nonnegativity constraints."""
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n
    a = np.pi * (1 + 5**0.5) * i
    r = np.sqrt(1 - z**2)
    return np.stack([r * np.cos(a), r * np.sin(a), z], 1)


# ------------------------------------------------------------------------------------------------ backend self-test
def _selftest():
    """numpy vs torch backends on a random structure-like A (small grid): moments, a backward pass, the QP."""
    import torch

    rng = np.random.default_rng(0)
    L, LB, nphi, NK, sh3, Q3, P = 20, 10, 48, 6, np.arange(4), 5, 4
    ls, ms = lm_index(L)
    A = (rng.standard_normal((NK, len(ls))) + 1j * rng.standard_normal((NK, len(ls)))) / (1 + ls)[None] ** 1.5
    dens = SHDensity(P)
    th = 0.1 * rng.standard_normal(dens.K)
    pn = PoseMoments(L, LB, nphi, NK, sh3, Q3, P=P)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pt = PoseMoments(L, LB, nphi, NK, sh3, Q3, P=P, dtype=np.complex64, backend="torch", device=dev)
    mn = pn.moments(A, *pn.of_theta(dens, th))
    At = torch.as_tensor(A, dtype=torch.complex64, device=dev).requires_grad_(True)
    mt = pt.moments(At, *pt.of_theta(dens, th))
    for k in ("a", "c2", "c3"):
        d = np.linalg.norm(mt[k].detach().cpu().numpy() - mn[k]) / np.linalg.norm(mn[k])
        print(f"  {k}: torch (complex64, {dev}) vs numpy (complex128) rel diff {d:.1e}")
    loss = sum((mt[k].abs() ** 2).sum() for k in ("a", "c2", "c3"))
    loss.backward()
    print(f"  backward: grad finite {bool(torch.isfinite(At.grad.abs()).all())}, |grad| {float(At.grad.abs().sum()):.3e}")
    with torch.no_grad():
        basis = pt.basis_moments(At.detach(), dens)
    ref = mt
    J = lambda m: float(sum(((m[k] - ref[k].detach()).abs() ** 2).sum() / (ref[k].detach().abs() ** 2).sum()
                            for k in ("a", "c2", "c3")))
    fit, info = fit_theta(basis, J, dens, collocation(400))
    print(f"  QP (torch moments): recovers theta to {np.linalg.norm(fit - th) / np.linalg.norm(th):.1e}, "
          f"loss {info['loss0']:.1e} -> {info['loss']:.1e}")


if __name__ == "__main__":
    _selftest()

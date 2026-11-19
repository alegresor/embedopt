"""Moment estimation with astigmatic CTFs and colored (stationary) noise.

Colored noise. For stationary noise with power spectrum S_k on the image's DFT grid (fft order, mean S = pixel
variance), the DFT coefficients are independent, E|n~_k|^2 = S_k L^2, and a polar sample is
    N(xi) = (s / L^2) sum_k n~_k psi_k(xi),     psi_k(xi) = prod_{d=x,y} sum_n e^{2 pi i k_d n / L} e^{-2 pi i xi_d (n - L/2) p}.
Hence, with Phi_q(kappa, k) = (1/n_phi) sum_phi psi_k(kappa e(phi)) e^{-i q phi},
    Sigma_q(k1, k2) = E[N_q(k1) conj N_q(k2)] = (s^2 / L^2) sum_k S_k Phi_q(k1, k) conj Phi_q(k2, k),
    P_q(k1, k2)     = E[N_q(k1) N_{-q}(k2)]   = (s^2 / L^2) sum_k S_k Phi_q(k1, k) Phi_{-q}(k2, -k).
For constant S this is exactly the white-noise result of moments.image_data.noise_polar_cov (checked).
The spectrum is either known (simulation) or estimated from the particle-free corners of the images (radially
averaged periodogram of the masked corners).

Astigmatism. With an angle-dependent CTF H_g(kappa, phi) per defocus group, an image's in-plane coefficients are
convolved with the CTF's angular harmonics h_{g,m}(kappa) = (1/n_phi) sum_phi H_g e^{-i m phi} (even m):
    I_q = sum_m h_m S_{q-m}.
For poses uniform in-plane, E[S_q] = a delta_{q0}, E[S_q(k1) conj S_q'(k2)] = c2_q delta_{qq'}, and E[S_q1 S_q2 S_q3] =
c3_{q1 q2} delta_{q1+q2+q3, 0}, so the observed group moments are linear in (a, c2, c3):
    E[I_m]                          = n_g h_m a
    E[sum_j I_q(k1) conj I_{q+d}(k2)] = n_g sum_m h_m(k1) conj h_{m+d}(k2) c2_{q-m}                     (d = m' - m)
    E[sum_j I_q1 I_q2 I_{e-q1-q2}]    = n_g sum_{m1+m2+m3=e} h_m1(a) h_m2(b) h_m3(c) c3_{q1-m1, q2-m2}  (e = sum of m)
and (a, c2, c3) are estimated by least squares over groups and these index combinations (weights 1/n_g): per shell
for a, a Toeplitz system per shell pair for c2, and batched conjugate gradients over all shell triples for c3. The noise
bias is subtracted as before (c2: d = 0 only; c3: the terms E[I] P with one in-plane index paired). With no
astigmatism (h_m = H delta_{m0}) every estimator reduces exactly to the radial one of moments.image_data.
"""

import numpy as np
import torch


# --------------------------------------------------------------------------------------------- CTF on the polar grid
def make_filters(defoci_u, defoci_v, angles, voltage=300, Cs=2.7, alpha=0.1):
    from aspire.operators import CTFFilter

    return [CTFFilter(voltage=voltage, defocus_u=float(u), defocus_v=float(v), defocus_ang=float(t), Cs=Cs, alpha=alpha)
            for u, v, t in zip(defoci_u, defoci_v, angles)]


def polar_ctf(filters, kappa, nphi, p):
    """H_g(kappa_i, phi_j): (G, NK, nphi), ASPIRE's evaluation at omega = 2 pi p kappa (cos phi, sin phi)."""
    phi = 2 * np.pi * np.arange(nphi) / nphi
    om = np.vstack([(2 * np.pi * p * kappa[:, None] * np.cos(phi)[None]).ravel(),
                    (2 * np.pi * p * kappa[:, None] * np.sin(phi)[None]).ravel()])
    return np.stack([np.asarray(f.evaluate(om, pixel_size=p)).reshape(len(kappa), nphi) for f in filters])


def harmonics(H, M):
    """h_{g,m}(kappa) for m = -M..M: (G, NK, 2M+1) complex."""
    nphi = H.shape[-1]
    F = np.fft.fft(H, axis=-1) / nphi
    return F[..., np.arange(-M, M + 1) % nphi]


# --------------------------------------------------------------------------------------------- colored noise
def radial_psd(kind, L, p, sigma2):
    """Noise power spectrum on the DFT grid (fft order), mean = sigma2. kind: 'white' or 'colored' (more power at low
    frequency, as from ice and background: S ~ 1 + 3 exp(-(k / 0.08 A^-1)^2))."""
    k = np.fft.fftfreq(L, d=p)
    r = np.sqrt(k[:, None] ** 2 + k[None, :] ** 2)
    f = np.ones_like(r) if kind == "white" else 1 + 3 * np.exp(-(r / 0.08) ** 2)
    return sigma2 * f / f.mean()


def add_noise(imgs, S, rng):
    """imgs + stationary noise with spectrum S (pixel variance mean(S))."""
    w = rng.standard_normal(imgs.shape)
    return imgs + np.fft.ifft2(np.sqrt(S) * np.fft.fft2(w)).real


class CornerPSD:
    """Radially averaged spectrum of the image corners (outside the inscribed circle), accumulated over images."""

    def __init__(self, L, p):
        n = np.arange(L) - L / 2
        self.w = ((n[:, None] ** 2 + n[None, :] ** 2) > (L / 2) ** 2).astype(float)
        k = np.fft.fftfreq(L, d=p)
        self.r = np.sqrt(k[:, None] ** 2 + k[None, :] ** 2)
        self.acc, self.n, self.L, self.p = np.zeros((L, L)), 0, L, p

    def add(self, imgs):
        x = imgs * self.w                                       # corners hold only noise (mean 0)
        self.acc += (np.abs(np.fft.fft2(x)) ** 2).sum(0)
        self.n += len(imgs)

    def spectrum(self, nbins=48):
        P = self.acc / self.n / (self.w ** 2).sum()              # periodogram of the masked noise (mean = variance)
        edges = np.linspace(0, self.r.max() + 1e-9, nbins + 1)
        idx = np.digitize(self.r.ravel(), edges) - 1
        prof = np.bincount(idx, P.ravel(), nbins) / np.maximum(np.bincount(idx, None, nbins), 1)
        mid = (edges[1:] + edges[:-1]) / 2
        return np.interp(self.r, mid, prof)


def noise_cov_psd(G, S, qmax2, Q3m, dev="cpu"):
    """Sigma_q (q = -qmax2..qmax2; (2 qmax2 + 1, NK, NK)) and P_q (q = -Q3m..Q3m on the third-order shells) for noise
    with spectrum S (absolute, not unit variance) on G's polar grid."""
    L, p, s, nphi = G.L_img, G.p, G.s, G.nphi
    phi = torch.as_tensor(2 * np.pi * np.arange(nphi) / nphi, dtype=torch.float64, device=dev)
    kk = torch.arange(L, dtype=torch.float64, device=dev)
    n = torch.arange(L, dtype=torch.float64, device=dev)

    def g(u):  # (nphi, L): sum_n e^{2 pi i k n / L} e^{-2 pi i u (n - L/2) p}, exact sum
        th = 2 * np.pi * (kk[None, :, None] * n[None, None, :] / L - u[:, None, None] * (n[None, None, :] - L / 2) * p)
        return torch.exp(1j * th).sum(-1)

    qs = np.arange(-qmax2, qmax2 + 1)
    St = torch.as_tensor(S, dtype=torch.float64, device=dev).reshape(-1)              # [ky, kx] flattened
    gxy = [(g(kap * torch.cos(phi)), g(kap * torch.sin(phi))) for kap in G.kappa]     # (nphi, L) each
    neg = ((-torch.arange(L, device=dev)) % L)
    negflat = (neg[:, None] * L + neg[None, :]).reshape(-1)                            # index of -k
    c = s**2 / L**2
    NK = len(G.kappa)
    Sig = torch.zeros(len(qs), NK, NK, dtype=torch.complex128, device=dev)
    q3 = np.arange(-Q3m, Q3m + 1)
    P = torch.zeros(len(q3), len(G.shells3), len(G.shells3), dtype=torch.complex128, device=dev)
    shells3 = list(G.shells3)

    def phis(qv):  # (NK, len(qv), L*L): Phi_q(kappa_i, k)
        E = torch.exp(-1j * torch.as_tensor(qv, dtype=torch.float64, device=dev)[:, None] * phi[None]) / nphi
        return torch.stack([torch.einsum("qp,py,px->qyx", E, gy, gx).reshape(len(qv), -1) for gx, gy in gxy])

    for i0 in range(0, len(qs), 16):                                                  # q chunks (memory)
        qv = qs[i0:i0 + 16]
        F = phis(qv)
        Sig[i0:i0 + len(qv)] = c * torch.einsum("iqk,jqk->qij", F * St, F.conj())
    for i0 in range(0, len(q3), 16):
        qv = q3[i0:i0 + 16]
        Fs = phis(qv)[shells3]
        Fm = phis(-qv)[shells3][:, :, negflat]                                         # Phi_{-q}(., -k)
        P[i0:i0 + len(qv)] = c * torch.einsum("iqk,jqk->qij", Fs * St, Fm)
    return Sig.cpu().numpy(), P.cpu().numpy()


# --------------------------------------------------------------------------------------------- astigmatic estimator
class Accumulator2D:
    """Group sums for the astigmatic estimator: T (q = -E1..E1), A2 (q = -(L+M2)..L+M2, d = -2M2..2M2), A3 (q1, q2 =
    -(Q3+M3)..Q3+M3, e = -3M3..3M3 even). The data polar grid must resolve |q| <= L + 3 M2 (G.nphi large enough)."""

    def __init__(self, G, n_groups, dev, M2=6, M3=2):
        self.G, self.dev, self.M2, self.M3 = G, dev, M2, M3
        NK, S = len(G.kappa), G.nb3
        self.E1 = max(M2, 3 * M3)
        self.Q2 = G.L + M2
        self.Qw = G.Q3 + M3
        self.es = np.arange(-3 * M3, 3 * M3 + 1, 2)
        if G.nphi // 2 - 1 < self.Q2 + 2 * M2:
            raise ValueError(f"n_phi {G.nphi} too small for |q| <= {self.Q2 + 2 * M2}")
        z = lambda *sh: torch.zeros(sh, dtype=torch.complex128, device=dev)
        Qo = 2 * self.Qw + 1
        self.n = np.zeros(n_groups)
        self.T = z(n_groups, NK, 2 * self.E1 + 1)
        self.A2 = z(n_groups, 2 * self.Q2 + 1, 4 * M2 + 1, NK, NK)
        self.A3 = z(n_groups, Qo, Qo, len(self.es), S, S, S)
        self.sh = torch.as_tensor(G.shells3, device=dev)
        nphi = G.nphi
        self.iT = torch.as_tensor(np.arange(-self.E1, self.E1 + 1) % nphi, device=dev)
        self.q2 = np.arange(-self.Q2, self.Q2 + 1)
        self.i2 = [(torch.as_tensor(self.q2 % nphi, device=dev), torch.as_tensor((self.q2 + d) % nphi, device=dev))
                   for d in range(-2 * M2, 2 * M2 + 1)]
        qw = np.arange(-self.Qw, self.Qw + 1)
        self.i1 = torch.as_tensor(qw % nphi, device=dev)
        self.i3 = [torch.as_tensor((e - qw[:, None] - qw[None, :]) % nphi, device=dev) for e in self.es]

    def add(self, Iq, groups):
        Iq = torch.as_tensor(Iq, device=self.dev)
        groups = np.asarray(groups)
        for g in np.unique(groups):
            X = Iq[torch.as_tensor(np.flatnonzero(groups == g), device=self.dev)]   # (b, NK, nphi)
            self.n[g] += len(X)
            self.T[g] += X[:, :, self.iT].sum(0)
            for j, (ia, ib) in enumerate(self.i2):
                self.A2[g, :, j] += torch.einsum("baq,bcq->qac", X[:, :, ia], X[:, :, ib].conj())
            Z = X[:, self.sh]                                                         # (b, S, nphi)
            A = Z[:, :, self.i1]
            Q, S = A.shape[-1], A.shape[1]
            Yo = torch.einsum("baq,bdr->qradb", A, A).reshape(Q, Q, S * S, -1)
            for j, i3 in enumerate(self.i3):
                self.A3[g, :, :, j] += (Yo @ Z[:, :, i3].permute(2, 3, 0, 1)).reshape(Q, Q, S, S, S)

    # ---- estimation
    def estimate(self, H, Sig, P, qmax2, Q3m, cg_iters=60, M2=None, M3=None):
        """H: (G, NK, nphi_h) polar CTF per group; Sig, P from noise_cov_psd (absolute); M2, M3 <= the accumulated
        ones (M = 0: ignore the astigmatism, using only the angular mean of the CTF)."""
        G, dev = self.G, self.dev
        M2 = self.M2 if M2 is None else M2
        M3 = self.M3 if M3 is None else M3
        keep = self.n > 0
        n = torch.as_tensor(self.n[keep], dtype=torch.float64, device=dev)
        h = torch.as_tensor(harmonics(H[keep], self.E1), dtype=torch.complex128, device=dev)   # (g, NK, 2E1+1)
        hm = lambda m: h[:, :, m + self.E1]
        # first moment
        T = self.T[keep]
        num = sum(hm(m).conj() * T[:, :, m + self.E1] for m in range(-M2, M2 + 1)).sum(0)
        den = sum((n[:, None] * hm(m).abs() ** 2) for m in range(-M2, M2 + 1)).sum(0)
        a = num / den
        # second moment: Toeplitz normal equations per shell pair, unknown c_p, p = -L..L
        L = G.L
        Pn = 2 * L + 1
        A2 = self.A2[keep].clone()
        d0 = 2 * self.M2
        Sq = torch.as_tensor(Sig, device=dev)                                         # q = -qmax2..qmax2
        A2[:, :, d0] -= n[:, None, None, None] * Sq[None, qmax2 - self.Q2:qmax2 + self.Q2 + 1]
        NK = len(G.kappa)
        Nmat = torch.zeros(NK, NK, Pn, Pn, dtype=torch.complex128, device=dev)
        rhs = torch.zeros(NK, NK, Pn, dtype=torch.complex128, device=dev)
        ps = torch.arange(-L, L + 1, device=dev)
        for d in range(-2 * M2, 2 * M2 + 1):
            for m in range(-M2, M2 + 1):
                if abs(m + d) > self.E1:
                    continue
                w = hm(m)[:, :, None] * hm(m + d).conj()[:, None, :]                  # (g, k1, k2)
                # rhs[p] += sum_g conj(w) O_g(q = p + m, d)
                O = A2[:, (ps + m + self.Q2), d + d0]                                   # (g, P, k1, k2)
                rhs += torch.einsum("gab,gpab->abp", w.conj(), O)
                for m2 in range(-M2, M2 + 1):
                    if abs(m2 + d) > self.E1:
                        continue
                    w2 = hm(m2)[:, :, None] * hm(m2 + d).conj()[:, None, :]
                    # N[p', p] += sum_g n_g conj(w_{m'}) w_{m2} for p' + m' = p + m2  (p' row, p col; m' = m)
                    coef = (n[:, None, None] * w.conj() * w2).sum(0)                     # (k1, k2)
                    shift = m2 - m                                                     # p' = p + m2 - m
                    lo, hi = max(0, -shift), min(Pn, Pn - shift)
                    idx = torch.arange(lo, hi, device=dev)
                    Nmat[:, :, idx + shift, idx] += coef[:, :, None]
        c2p = torch.linalg.solve(Nmat, rhs[..., None])[..., 0]                         # (k1, k2, P)
        # c2_{-p}(k1, k2) = c2_p(k2, k1) for real images: average the two estimates
        c2 = ((c2p[:, :, L:] + c2p[:, :, :L + 1].flip(-1).transpose(0, 1)) / 2).permute(2, 0, 1)  # q = 0..L
        # third moment: batched CG over shell triples on the window |p| <= Qw
        sh = self.sh
        h3 = h[:, sh]                                                                  # (g, S, 2E1+1)
        Qw, Qo = self.Qw, 2 * self.Qw + 1
        es = [int(e) for e in self.es if abs(e) <= 3 * M3]
        terms = []
        for e in es:
            je = int(np.flatnonzero(self.es == e)[0])
            for m1 in range(-M3, M3 + 1, 2):
                for m2 in range(-M3, M3 + 1, 2):
                    m3 = e - m1 - m2
                    if abs(m3) > M3:
                        continue
                    coef = (h3[:, :, m1 + self.E1][:, :, None, None] * h3[:, :, m2 + self.E1][:, None, :, None]
                            * h3[:, :, m3 + self.E1][:, None, None, :])                    # (g, a, b, c)
                    terms.append((je, m1, m2, coef))

        def shift2(X, m1, m2):  # out(q1, q2) = X(q1 - m1, q2 - m2), zero outside
            out = torch.zeros_like(X)
            s1 = slice(max(0, m1), Qo + min(0, m1)); t1 = slice(max(0, -m1), Qo + min(0, -m1))
            s2 = slice(max(0, m2), Qo + min(0, m2)); t2 = slice(max(0, -m2), Qo + min(0, -m2))
            out[s1, s2] = X[t1, t2]
            return out

        def Aop(X):  # X (Qo, Qo, S, S, S) -> Y (g, Qo, Qo, Ne, S, S, S), Y = sum coef * shift(X)
            Y = torch.zeros((len(n), Qo, Qo, len(self.es)) + X.shape[2:], dtype=X.dtype, device=dev)
            for je, m1, m2, coef in terms:
                Y[:, :, :, je] += coef[:, None, None] * shift2(X, m1, m2)[None]
            return Y

        def AHop(Y, wts):  # sum_g wts_g A_g^H Y_g
            X = 0
            for je, m1, m2, coef in terms:
                Z = (wts[:, None, None, None, None, None] * coef.conj()[:, None, None] * Y[:, :, :, je]).sum(0)
                X = X + shift2(Z, -m1, -m2)
            return X

        A3 = self.A3[keep].clone()
        Pq = torch.as_tensor(P, device=dev)                                            # q = -Q3m..Q3m
        Tg = T[:, sh]                                                                  # (g, S, 2E1+1)
        qw = np.arange(-Qw, Qw + 1)
        Pw = Pq[torch.as_tensor(qw + Q3m, device=dev)]                                  # (Qo, S, S)
        for je, e in enumerate(self.es):
            if abs(e) > self.E1:
                continue
            te = Tg[:, :, int(e) + self.E1]                                             # (g, S)
            k = int(np.flatnonzero(qw == e)[0]) if abs(e) <= Qw else None
            if k is not None:
                A3[:, k, :, je] -= torch.einsum("ga,qbc->gqabc", te, Pw)                # q1 = e: T(a) P_q2(b, c)
                A3[:, :, k, je] -= torch.einsum("gb,qac->gqabc", te, Pw)                # q2 = e: T(b) P_q1(a, c)
            for i, q1 in enumerate(qw):                                                # q1 + q2 = 0: T(c) P_q1(a, b)
                j = int(np.flatnonzero(qw == -q1)[0])
                A3[:, i, j, je] -= torch.einsum("gc,ab->gabc", te, Pw[i])
        b = AHop(A3, torch.ones_like(n))
        normal = lambda X: AHop(Aop(X), n)
        diag = 0
        for je, m1, m2, coef in terms:
            if m1 == 0 and m2 == 0 and self.es[je] == 0:
                diag = (n[:, None, None, None] * coef.abs() ** 2).sum(0)               # (S, S, S)
        Minv = 1 / diag[None, None]
        X = b * Minv
        R = b - normal(X)
        Z = R * Minv
        Pd = Z.clone()
        rz = (R.conj() * Z).real.sum()
        r0 = R.abs().pow(2).sum().sqrt()
        for it in range(cg_iters):
            Ap = normal(Pd)
            alpha = rz / (Pd.conj() * Ap).real.sum()
            X = X + alpha * Pd
            R = R - alpha * Ap
            if R.abs().pow(2).sum().sqrt() < 1e-10 * r0:
                break
            Z = R * Minv
            rz_new = (R.conj() * Z).real.sum()
            Pd = Z + (rz_new / rz) * Pd
            rz = rz_new
        self.cg = dict(iters=it + 1, rel_residual=float(R.abs().pow(2).sum().sqrt() / r0))
        Q3 = G.Q3
        c3 = X[Qw - Q3:Qw + Q3 + 1, Qw - Q3:Qw + Q3 + 1]
        return {k: v.cpu().numpy() for k, v in dict(a=a, c2=c2, c3=c3).items()}

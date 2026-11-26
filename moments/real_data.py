"""Image moments (a, c2, c3) from a real RELION particle stack, and their comparison with a structure's moments.

Data model. I_j = s E(kappa) H_j(kappa) Vhat_j + N_j: unknown intensity scale and contrast sign s, resolution envelope
E(kappa) = exp(-dB kappa^2 / 4) (beam damage, detector, model B-factor mismatch), radial CTF H_j of each particle (its
own defocus; RELION convention up to a global sign, which s absorbs), stationary colored noise N with spectrum S.

Estimators. As moments.image_data, with every particle its own defocus group. With X~_j = H_j I_j,
    a(k)       = sum_j X~_j0(k) / sum_j H_j(k)^2
    c2_q(k1,k2) = [sum_j X~_jq(k1) conj X~_jq(k2) - Sigma_q(k1,k2) sum_j H_j(k1) H_j(k2)] / sum_j H_j(k1)^2 H_j(k2)^2
    c3_q1q2     = [sum_j X~_q1(a) X~_q2(b) X~_q3(c) - bias] / sum_j H_a^2 H_b^2 H_c^2,
    bias        = d_{q1,0} T(a;b,c) P_q2(b,c) + d_{q2,0} T(b;a,c) P_q1(a,c) + d_{q3,0} T(c;a,b) P_q1(a,b),
    T(a;b,c)    = sum_j X~_j0(a) H_j(b) H_j(c),
with Sigma, P the noise covariances of the polar coefficients for the corner-estimated spectrum (moments.ctf2d).
No binning of defocus. Exact for discrete groups too. Particles are split into two halves (even / odd rows):
the half-vs-half difference measures the sampling error.

Centering: none (picked coordinates as extracted; pose-free) or origins (rlnOriginX/YAngst from 3D refinement, which
carry pose information; for comparison only). The sign convention of the origins is determined from the data.

    python -m moments.real_data stack   --star S.star --root DIR --system 7N6N --center none    -> out/real/<tag>.npz
    python -m moments.real_data selftest [--system 7N6N]                                    (no data needed)
    python -m moments.real_data compare --data out/real/<tag>.npz --system 7N6N [--model pred.cif ...]
    python -m moments.real_data export  --data out/real/<tag>.npz --system 7N6N     -> <tag>_corr.npz (DATA= for M12/M3=corr)
"""
import argparse, os, re, time
import numpy as np
import torch

from moments.image_data import Accumulator, PolarGrid, model_moments, rel
from moments.image_moments import ImageLoss
from moments.ctf2d import CornerPSD, noise_cov_psd
from moments.simulate import load_xstar


# ------------------------------------------------------------------------------------------------ CTF (RELION)
def wavelength(kv):
    V = kv * 1e3
    return 12.2643247 / np.sqrt(V * (1 + 0.978466e-6 * V))          # Angstrom


def bessel_j0(x, n=64):
    """J0(x) = mean over u of cos(x cos u) (trapezoid on the circle: exact to machine precision for |x| < ~20)."""
    u = 2 * np.pi * np.arange(n) / n
    return np.cos(np.asarray(x)[..., None] * np.cos(u)).mean(-1)


def radial_ctf(kappa, dfU, dfV, kv, cs_mm, Q, phase_deg=0.0, bfac=0.0, scale=1.0, astig_avg=True):
    """H_j(kappa), (B, NK): the in-plane average h_0 of the 2D CTF sin(chi_0 + A cos 2psi),
        h_0 = sin(chi_0) J0(A),  chi_0 = pi lam df k^2 - pi/2 Cs lam^3 k^4 + asin Q + phase,  A = pi lam (U - V)/2 k^2,
    (independent of the astigmatism angle), times scale exp(-bfac k^2 / 4). astig_avg=False: sin(chi_0) (mean defocus).
    RELION's CTF is -sin(...) of its argument; the global sign is absorbed in s."""
    col = lambda v: np.broadcast_to(np.asarray(v, float), np.shape(dfU))[:, None]          # per particle -> (B, 1)
    lam, k2 = wavelength(col(kv)), kappa[None] ** 2
    df = col((np.asarray(dfU) + np.asarray(dfV)) / 2)
    chi = np.pi * lam * df * k2 - np.pi / 2 * col(cs_mm) * 1e7 * lam**3 * k2**2 + np.arcsin(col(Q)) + np.deg2rad(col(phase_deg))
    H = np.sin(chi)
    if astig_avg:
        H = H * bessel_j0(np.pi * lam * col(np.asarray(dfU) - np.asarray(dfV)) / 2 * k2)
    return H * col(scale) * np.exp(-col(bfac) * k2 / 4)


# ------------------------------------------------------------------------------------------------ STAR / images
def read_particles(star):
    import starfile

    d = starfile.read(star)
    p, o = d["particles"], d["optics"]
    p = p.merge(o, on="rlnOpticsGroup", how="left", suffixes=("", "_optics"))
    col = lambda c, v: p[c].to_numpy(float) if c in p else np.full(len(p), v)
    meta = dict(dfU=col("rlnDefocusU", 0), dfV=col("rlnDefocusV", 0), ang=col("rlnDefocusAngle", 0),
                kv=col("rlnVoltage", 300), cs=col("rlnSphericalAberration", 2.7), Q=col("rlnAmplitudeContrast", 0.1),
                phase=col("rlnPhaseShift", 0), bfac=col("rlnCtfBfactor", 0), scale=col("rlnCtfScalefactor", 1),
                ox=col("rlnOriginXAngst", 0), oy=col("rlnOriginYAngst", 0),
                p=float(o["rlnImagePixelSize"].iloc[0]), L=int(o["rlnImageSize"].iloc[0]))
    idx, fn = zip(*(s.split("@") for s in p["rlnImageName"]))
    return meta, np.array(idx, int) - 1, np.array(fn)


def resolve(root, fn):
    """Strip leading path components of the STAR file names until they exist under root."""
    parts = fn.split("/")
    for k in range(len(parts)):
        if os.path.isfile(os.path.join(root, *parts[k:])):
            return k
    raise FileNotFoundError(f"{fn} not found under {root}")


def batches(root, idx, fn, rows, batch):
    """Yields (row indices, images (B, L, L) float64) in file order."""
    import mrcfile

    k = resolve(root, fn[rows[0]])
    order = rows[np.argsort(fn[rows], kind="stable")]
    uniq, start = np.unique(fn[order], return_index=True)
    buf_r, buf_x = [], []
    for f, r in zip(uniq, np.split(order, start[1:])):
        with mrcfile.mmap(os.path.join(root, *f.split("/")[k:]), mode="r", permissive=True) as m:
            data = m.data if m.data.ndim == 3 else m.data[None]
            buf_r.append(r); buf_x.append(np.asarray(data[idx[r]], np.float64))
        if sum(len(x) for x in buf_r) >= batch:
            yield np.concatenate(buf_r), np.concatenate(buf_x); buf_r, buf_x = [], []
    if buf_r:
        yield np.concatenate(buf_r), np.concatenate(buf_x)


# ------------------------------------------------------------------------------------------------ accumulation
class Extra:
    """CTF sums per half: sum H^2, sum H^2 H^2, sum H H (noise bias of c2), sum H^2 H^2 H^2 and T on third shells."""

    def __init__(self, G, dev):
        NK, S, z = len(G.kappa), G.nb3, lambda *s: torch.zeros(s, dtype=torch.float64, device=dev)
        self.d1, self.d2, self.w2, self.d3 = z(2, NK), z(2, NK, NK), z(2, NK, NK), z(2, S, S, S)
        self.T = torch.zeros(2, S, S, S, dtype=torch.complex128, device=dev)
        self.sh, self.dev = torch.as_tensor(G.shells3, device=dev), dev

    def add(self, H, X0, half):
        H, X0 = torch.as_tensor(H, device=self.dev), torch.as_tensor(X0, device=self.dev)   # (b, NK), X~_0 (b, NK)
        for h in (0, 1):
            m = torch.as_tensor(half == h, device=self.dev)
            Hh, Hs, Xs = H[m], H[m][:, self.sh], X0[m][:, self.sh]
            self.d1[h] += (Hh**2).sum(0)
            self.d2[h] += torch.einsum("ba,bc->ac", Hh**2, Hh**2)
            self.w2[h] += torch.einsum("ba,bc->ac", Hh, Hh)
            self.d3[h] += torch.einsum("ba,bd,bc->adc", Hs**2, Hs**2, Hs**2)
            Hc = Hs.to(Xs.dtype)                                                     # einsum needs one dtype
            self.T[h] += torch.einsum("ba,bd,bc->adc", Xs, Hc, Hc)


def estimate(acc, ex, G, Sig, P, halves=(0, 1), alpha=None):
    """Moments from the sums of the given halves (Sig: (L+1, NK, NK) q = 0..L; P: (2 Q3 + 1, S, S) q = -Q3..Q3).
    alpha (NK,): per-shell noise calibration, Sig(k1,k2) and P(b,c) scaled by sqrt(alpha alpha)."""
    h = list(halves)
    dev = ex.dev
    s1, s2, s3 = (x[h].sum(0) for x in (acc.s1, acc.s2, acc.s3))
    d1, d2, w2, d3, T = (x[h].sum(0) for x in (ex.d1, ex.d2, ex.w2, ex.d3, ex.T))
    if alpha is not None:
        r = np.sqrt(np.outer(alpha, alpha))
        Sig, P = Sig * r[None], P * r[np.ix_(G.shells3, G.shells3)][None]
    Sig, P = torch.as_tensor(Sig, device=dev), torch.as_tensor(P, device=dev)
    a = s1 / d1
    c2 = (s2 - Sig * w2[None]) / d2[None]
    Q3, qs = G.Q3, np.arange(-G.Q3, G.Q3 + 1)
    bias = torch.zeros_like(s3)
    bias[Q3] += torch.einsum("abc,qbc->qabc", T, P)                       # q1 = 0: T(a;b,c) P_q2(b,c)
    bias[:, Q3] += torch.einsum("bac,qac->qabc", T, P)                    # q2 = 0: T(b;a,c) P_q1(a,c)
    for k, q1 in enumerate(qs):                                           # q3 = 0: q2 = -q1, T(c;a,b) P_q1(a,b)
        bias[k, int(np.flatnonzero(qs == -q1)[0])] += torch.einsum("cab,ab->abc", T, P[k])
    c3 = (s3 - bias) / d3[None, None]
    return {k: v.cpu().numpy() for k, v in dict(a=a, c2=c2, c3=c3).items()}


class Detrend:
    """Least-squares plane mu + bx x + by y on the corner pixels (outside the inscribed circle), subtracted from the
    whole image: removes background offsets and ice gradients, which otherwise leak into the corner spectrum through
    the mask's transform and into the moments through the finite box."""

    def __init__(self, L):
        n = np.arange(L) - L / 2
        y, x = np.meshgrid(n, n, indexing="ij")
        self.m = ((x**2 + y**2) > (L / 2) ** 2).ravel()
        self.A = np.stack([np.ones(L * L), x.ravel() / L, y.ravel() / L], 1)           # (L^2, 3)
        self.pinv = np.linalg.pinv(self.A[self.m])                                    # (3, n_corner)
        self.stats = np.zeros(3)                                                      # sum mu^2, sum |b|^2, n

    def __call__(self, x):
        B, L, _ = x.shape
        f = x.reshape(B, -1)
        c = f[:, self.m] @ self.pinv.T                                                # (B, 3)
        self.stats += [(c[:, 0] ** 2).sum(), (c[:, 1:] ** 2).sum(), B]
        return (f - c @ self.A.T).reshape(B, L, L)


def origin_sign(G, meta, root, idx, fn, rows, n=3000):
    """Sign of the origins that best centres the particles: |sum_j H_j I_j0| on the shells, mid-weighted."""
    sel = rows[:n]
    score = {}
    for sgn in (0.0, 1.0, -1.0):
        num = 0
        for r, x in batches(root, idx, fn, sel, 1024):
            ok = np.isfinite(x).all((1, 2)) & (np.abs(x).max((1, 2)) < 1e3 * np.median(np.abs(x).std((1, 2))))
            r, x = r[ok], x[ok]
            x = Detrend(x.shape[-1])(x)
            H = ctf_of(meta, r, G.kappa)
            sh = None if sgn == 0 else sgn * np.stack([meta["ox"][r], meta["oy"][r]], 1)
            num = num + (H * G.coeffs(x, sh)[:, :, 0]).sum(0)
        score[sgn] = float((np.abs(num) * G.kappa**2).sum())
    return score


def ctf_of(meta, r, kappa):
    return radial_ctf(kappa, meta["dfU"][r], meta["dfV"][r], meta["kv"][r], meta["cs"][r], meta["Q"][r],
                      meta["phase"][r], meta["bfac"][r], meta["scale"][r], meta.get("astig_avg", True))


def cmd_stack(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    meta, idx, fn = read_particles(a.star)
    atoms = load_xstar(a.system)
    G = PolarGrid(atoms, meta["L"], meta["p"], nb3=a.nb3, Q3=a.q3, stride3=a.s3stride)
    rows = np.arange(len(idx))
    if a.n and a.n < len(rows):
        rows = np.sort(np.random.default_rng(a.seed).choice(rows, a.n, replace=False))
    half = (rows % 2).astype(int)
    lam, k = wavelength(meta["kv"][0]), G.kappa[-1]
    dphi = np.pi * lam * np.abs(meta["dfU"] - meta["dfV"])[rows] / 2 * k**2
    print(f"{len(rows)} particles, box {meta['L']} px x {meta['p']} A, r_max {G.r_max:.1f} A, L {G.L}, nphi {G.nphi}")
    print(f"defocus {np.percentile((meta['dfU'] + meta['dfV'])[rows] / 2, [1, 50, 99]).round(0)} A; astigmatic phase "
          f"error at {k:.3f} 1/A (radial CTF): median {np.median(dphi):.2f}, 99% {np.percentile(dphi, 99):.2f} rad")
    meta["astig_avg"] = a.ctf == "avg"
    A99 = np.percentile(np.abs(meta["dfU"] - meta["dfV"])[rows], 99) * np.pi * lam / 2 * G.kappa**2
    print(f"CTF: {'in-plane average sin(chi0) J0(A)' if meta['astig_avg'] else 'mean defocus sin(chi0)'}; "
          f"99% astigmatic amplitude A < 0.3 rad up to shell {int((A99 < 0.3).sum())} of {len(G.kappa)} "
          f"(1/{1 / G.kappa[max(int((A99 < 0.3).sum()) - 1, 0)]:.1f} A); third-order shells end at shell {G.shells3[-1] + 1}")
    sgn = 0.0
    if a.center == "origins":
        sc = origin_sign(G, meta, a.root, idx, fn, rows)
        sgn = max((1.0, -1.0), key=sc.get)
        print(f"centering score (none / +origins / -origins): {sc[0.0]:.4g} / {sc[1.0]:.4g} / {sc[-1.0]:.4g}; using {sgn:+.0f}")
    acc, ex, psd, det = Accumulator(G, 2, dev), Extra(G, dev), CornerPSD(meta["L"], meta["p"]), Detrend(meta["L"])
    t0, seen, ref_std, bad, nrej = time.time(), 0, None, [], {}
    for r, x in batches(a.root, idx, fn, rows, a.batch):
        # screen: non-finite pixels, or corner std off by > a.screen x the median of the first batch
        f = x.reshape(len(x), -1)
        fin = np.isfinite(f).all(1)
        f = np.where(fin[:, None], f, 0.0)
        cstd, fstd, fmax = f[:, det.m].std(1), f.std(1), np.abs(f).max(1)
        if ref_std is None:
            ref_std = float(np.median(cstd[fin]))
        crit = dict(nonfinite=~fin,
                    corner_std=(cstd > a.screen * ref_std) | (cstd < ref_std / a.screen),
                    full_std=(fstd > a.screen * ref_std) | (fstd < ref_std / a.screen),
                    max_abs=fmax > a.max_sigma * ref_std)
        for k, v in crit.items():
            nrej[k] = nrej.get(k, 0) + int(v.sum())
        ok = ~np.any(list(crit.values()), 0)
        bad += [(fn[j], int(idx[j]) + 1, float(c), float(m)) for j, c, m in zip(r[~ok], fstd[~ok], fmax[~ok])]
        r, x = r[ok], x[ok]
        seen += int((~ok).sum())
        if not len(r):
            continue
        if a.detrend:
            x = det(x)
        H = ctf_of(meta, r, G.kappa)
        sh = None if sgn == 0 else sgn * np.stack([meta["ox"][r], meta["oy"][r]], 1)
        X = G.coeffs(x, sh) * H[:, :, None]                                   # X~ = H I
        hr = (r % 2).astype(int)
        acc.add(X, hr); ex.add(H, X[:, :, 0], hr); psd.add(x)
        seen += len(r)
        if seen // a.log_every != (seen - len(r)) // a.log_every:
            el = time.time() - t0
            print(f"  {seen}/{len(rows)} ({el / 60:.1f} min, ~{el * (len(rows) / seen - 1) / 60:.0f} min left)", flush=True)
    S = psd.spectrum()
    rej = a.out.replace(".npz", "_rejected.txt")
    with open(rej, "w") as f:
        f.write("file\timage\tstd\tmax_abs\n")
        f.writelines(f"{n}\t{i}\t{c:.4g}\t{m:.4g}\n" for n, i, c, m in bad)
    nbf = len({b[0] for b in bad})
    print(f"screening: reference corner std {ref_std:.3f}; rejected {len(bad)} of {len(rows)} images in {nbf} files "
          f"(by criterion, overlapping: {nrej}) -> {rej}")
    if a.detrend:
        mu2, b2, nn = det.stats
        print(f"removed background: rms offset {np.sqrt(mu2 / nn):.3f}, rms gradient {np.sqrt(b2 / nn / 2):.3f} per box; "
              f"corner noise variance after detrending {S.mean():.3f}")
    SigF, P = noise_cov_psd(G, S, G.L, G.Q3, dev)
    Sig = SigF[G.L:]                                                          # q = 0..L
    full, h0, h1 = estimate(acc, ex, G, Sig, P), estimate(acc, ex, G, Sig, P, (0,)), estimate(acc, ex, G, Sig, P, (1,))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez(a.out, **full, kappa=G.kappa, nb3=G.nb3, Q3=G.Q3, L=G.L, shells3=G.shells3, n_images=len(rows) - len(bad), n_rejected=len(bad), ctf=a.ctf, detrend=a.detrend, psd_profile=S[0, : meta["L"] // 2],
             r_max=G.r_max, L_img=meta["L"], pixel=meta["p"], center=a.center, origin_sign=sgn, system=a.system,
             noise_var=float(S.mean()), **{f"h0_{k}": v for k, v in h0.items()}, **{f"h1_{k}": v for k, v in h1.items()},
             Sig=Sig, P=P, **{f"sum_{k}": getattr(acc, k).cpu().numpy() for k in ("s1", "s2", "s3")},
             **{f"sum_{k}": getattr(ex, k).cpu().numpy() for k in ("d1", "d2", "w2", "d3", "T")})
    print(f"wrote {a.out} ({(time.time() - t0) / 60:.1f} min; corner noise variance {S.mean():.3f})")


# ------------------------------------------------------------------------------------------------ comparison
class _Sums:
    def __init__(self, D, keys, dev):
        for k in keys:
            setattr(self, k, torch.as_tensor(D[f"sum_{k}"], device=dev))
        self.dev = dev


def recalibrate(D, G, atoms, dev, margin):
    """Per-shell noise scale alpha_k from in-plane orders beyond the particle's bandlimit, q > 2 pi k r_p + margin,
    where the raw sums S2_q(k,k) are noise only; then re-estimate with Sig, P scaled by sqrt(alpha alpha)."""
    X = np.asarray(atoms.coord, float)
    rp = np.linalg.norm(X - X.mean(0), axis=1).max()
    acc, ex = _Sums(D, ("s1", "s2", "s3"), dev), _Sums(D, ("d1", "d2", "w2", "d3", "T"), dev)
    S2 = D["sum_s2"].sum(0)                                                    # (L+1, NK, NK), both halves
    B2 = D["Sig"] * D["sum_w2"].sum(0)[None]
    qv, K = np.arange(S2.shape[0]), G.kappa
    alpha, snr = np.full(len(K), np.nan), np.full(len(K), np.nan)
    print(f"   noise calibration from signal-free orders (r_p = {rp:.1f} A, margin {margin}):")
    print("   shell  q_s  #q   alpha    signal/noise power (calibrated)")
    for k in range(len(K)):
        qs = int(np.ceil(2 * np.pi * K[k] * rp)) + margin
        free = qv > qs
        if free.sum() >= 3:
            alpha[k] = S2[free, k, k].real.sum() / B2[free, k, k].real.sum()
        sig = (S2[:, k, k].real - alpha[k] * B2[:, k, k].real).sum()
        snr[k] = sig / (alpha[k] * B2[:, k, k].real.sum())
        print(f"   {k + 1:>5}  {qs:>3}  {int(free.sum()):>3}  {alpha[k]:.4f}   {snr[k]:+.2e}")
    if np.isnan(alpha).any():                                                  # shells without free orders: neighbour
        good = ~np.isnan(alpha)
        alpha = np.interp(np.arange(len(K)), np.flatnonzero(good), alpha[good])
    est = lambda h: estimate(acc, ex, G, D["Sig"], D["P"], h, alpha)
    return est((0, 1)), [est((0,)), est((1,))]


def fit_scale(a_d, a_m, kappa, sl):
    """(s, dB) minimizing sum_k |a_d - s exp(-dB k^2 / 4) a_m|^2 over shells sl: dB on a grid, s in closed form."""
    best = None
    for dB in np.linspace(-300, 1500, 1801):
        u = np.exp(-dB * kappa[sl] ** 2 / 4) * a_m[sl]
        s = np.real(np.vdot(u, a_d[sl])) / np.vdot(u, u).real
        r = np.linalg.norm(a_d[sl] - s * u) ** 2
        if best is None or r < best[2]:
            best = (s, dB, r)
    return best[0], best[1]


def shell_envelope(d, m, sign):
    """Free per-shell envelope from the power: e_k^2 = sum_q c2d_q(k,k) / sum_q c2m_q(k,k) (Wilson-plot scaling);
    sign from the first moment's lowest shell. Returns signed e (NK,) with e = s E(kappa)."""
    pd = np.einsum("qkk->k", d["c2"]).real
    pm = np.einsum("qkk->k", m["c2"]).real
    if (pd <= 0).any():
        print(f"   WARNING: non-positive data power in shells {list(np.flatnonzero(pd <= 0) + 1)} (noise over-subtracted); "
              f"envelope set to NaN there")
    return sign * np.sqrt(np.where(pd > 0, pd, np.nan) / pm)


def normalize(m, s, dB, kappa, sh3, e=None):
    """Divide out s E(kappa): E = exp(-dB k^2 / 4) with scale s, or a given signed per-shell e (then s, dB unused)."""
    e = s * np.exp(-dB * kappa**2 / 4) if e is None else e
    es = e[sh3]
    return dict(a=m["a"] / e, c2=m["c2"] / (e[:, None] * e[None]),
                c3=m["c3"] / (es[:, None, None] * es[None, :, None] * es[None, None]))


# ------------------------------------------------------------------------------------------------ bias-free comparison
def biasfree_masks(G, dmin):
    """c2: shell pairs |k1 - k2| >= dmin (noise correlated only through the box kernel); c3: blocks with q1, q2, q3 != 0
    (the bias T P lives only in blocks with a zero index). Both free of noise bias for any stationary noise."""
    NK, qs = len(G.kappa), np.arange(-G.Q3, G.Q3 + 1)
    k = np.arange(NK)
    M2 = np.broadcast_to(np.abs(k[:, None] - k[None]) >= dmin, (G.L + 1, NK, NK))
    q3 = -(qs[:, None] + qs[None])
    M3 = ((qs[:, None] != 0) & (qs[None] != 0) & (q3 != 0))[:, :, None, None, None]
    return M2, np.broadcast_to(M3, (len(qs), len(qs)) + (G.nb3,) * 3)


def offdiag_envelope(d2, m2, M2, sign, iters=200):
    """Per-shell e minimizing sum_{M2} |c2d(k1,k2) - e1 e2 c2m(k1,k2)|^2 (alternating least squares), signed."""
    W = M2[0].astype(float)
    num = np.einsum("qij,qij->ij", np.conj(m2), d2).real * W                    # <c2m, c2d> per pair
    den = np.einsum("qij,qij->ij", np.conj(m2), m2).real * W
    e = np.sqrt(np.maximum(np.abs(num).sum(1) / den.sum(1), 1e-300))
    for _ in range(iters):     # simultaneous update maps c e -> e / c (period-2 cycle); the geometric mean removes it
        e = np.sqrt(e * np.maximum((num @ e) / (den @ e**2), 1e-12))
    return sign * e


def mrel(d, m, M):
    return float(np.linalg.norm((d - m)[M]) / np.linalg.norm(m[M]))


def pair_corr(x, y, axes):
    """Normalized correlation Re<x, y> / (|x| |y|) over `axes`, per remaining index (scale-invariant per block)."""
    num = np.sum(np.conj(x) * y, axis=axes).real
    return num / np.sqrt(np.sum(np.abs(x) ** 2, axis=axes) * np.sum(np.abs(y) ** 2, axis=axes) + 1e-300)


def correlations(data, halves, m, G, dmin):
    """rho2[i,j] over q for c2 shell pairs |i-j| >= dmin; rho3[a,b,c] over nonzero-q blocks for c3 shell triples.
    Also the half-vs-half correlations (reproducibility of each entry in the data)."""
    M2, M3 = biasfree_masks(G, dmin)
    pairs = M2[0]
    r2 = pair_corr(m["c2"], data["c2"], 0)
    h2 = pair_corr(halves[0]["c2"], halves[1]["c2"], 0)
    keep = M3[:, :, 0, 0, 0]
    c3m, c3d = m["c3"][keep], data["c3"][keep]                              # (nblocks, S, S, S)
    r3 = pair_corr(c3m, c3d, 0)
    h3 = pair_corr(halves[0]["c3"][keep], halves[1]["c3"][keep], 0)
    return r2, h2, pairs, r3, h3


def compare_biasfree(name, data, halves, m, G, dmin, rows):
    M2, M3 = biasfree_masks(G, dmin)
    sign = np.sign((data["a"][0] / m["a"][0]).real) or 1.0
    e = offdiag_envelope(data["c2"], m["c2"], M2, sign)
    dn = normalize(data, 0, 0, G.kappa, G.shells3, e)
    hn = [normalize(h, 0, 0, G.kappa, G.shells3, e) for h in halves]
    L2, L3 = mrel(dn["c2"], m["c2"], M2) ** 2, mrel(dn["c3"], m["c3"], M3) ** 2
    F2 = mrel(m["c2"] + hn[0]["c2"] - hn[1]["c2"], m["c2"], M2) ** 2 / 4
    F3 = mrel(m["c3"] + hn[0]["c3"] - hn[1]["c3"], m["c3"], M3) ** 2 / 4
    rows.append((name, L2, F2, L3, F3))
    print(f"\n== {name} (bias-free entries: c2 |k1-k2| >= {dmin}, c3 blocks q1,q2,q3 != 0)")
    print("   envelope e(k)/e(k1): " + " ".join(f"{v:.2f}" for v in np.abs(e / e[0])))
    K = G.kappa
    Bk = -2 * np.log(np.abs(e[1:] / e[0])) / (K[1:] ** 2 - K[0] ** 2)
    print("   implied B (A^2):     " + " ".join(f"{v:.0f}" for v in Bk))
    print("   shell  1/kappa(A)  c2 row rel err (half-split)   c3 diag-block rel err (half-split)")
    for i in range(len(K)):
        Mi = np.zeros_like(M2); Mi[:, i, :] = M2[:, i, :]
        e2 = mrel(dn["c2"], m["c2"], Mi)
        n2 = mrel(m["c2"] + hn[0]["c2"] - hn[1]["c2"], m["c2"], Mi) / 2
        line = f"   {i + 1:>5}  {1 / K[i]:>10.1f}  {e2:>10.3f} ({n2:.3f})"
        if i in G.shells3:
            j = int(np.flatnonzero(G.shells3 == i)[0])
            Mj = np.zeros_like(M3); Mj[..., j, j, j] = M3[..., j, j, j]
            e3 = mrel(dn["c3"], m["c3"], Mj)
            n3 = mrel(m["c3"] + hn[0]["c3"] - hn[1]["c3"], m["c3"], Mj) / 2
            line += f"          {e3:>10.3f} ({n3:.3f})"
        print(line)
    return dn, e


def losses(ref, test, dev):
    """Default image-space losses of the reward (m12 plain, m3 block) of `test` against `ref` (here the model, whose
    norms and nonzero c3 blocks define the normalization; noisy data would also keep its noise-only blocks)."""
    T = lambda d: {k: torch.as_tensor(np.asarray(d[k]), dtype=torch.complex128, device=dev) for k in ("a", "c2", "c3")}
    lo, t = ImageLoss(T(ref)), T(test)
    return dict(m12=float(lo.m12(t, "plain")), m3=float(lo.m3(t, "block")))


def structure_atoms(path):
    import biotite.structure.io as bio

    st = bio.load_structure(path)
    st = st[0] if hasattr(st, "stack_depth") else st
    return st[st.element != "H"]


def cmd_compare(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    D = dict(np.load(a.data))
    atoms = load_xstar(a.system)
    G = PolarGrid(atoms, int(D["L_img"]), float(D["pixel"]), nb3=int(D["nb3"]), Q3=int(D["Q3"]))
    if G.L != int(D["L"]) or not np.allclose(G.kappa, D["kappa"]) or not np.array_equal(G.shells3, D["shells3"]):
        raise SystemExit("data grid differs from the grid of this system")
    sh3, K = G.shells3, G.kappa
    sl = slice(a.fit_from, None)
    data = {k: D[k] for k in ("a", "c2", "c3")}
    halves = [{k: D[f"h{i}_{k}"] for k in ("a", "c2", "c3")} for i in (0, 1)]
    if "sum_s2" in D and not a.no_calib:
        data, halves = recalibrate(D, G, atoms, dev, a.rp_margin)
    models = [("x*", atoms)] + [(os.path.basename(p), structure_atoms(p)) for p in a.model]
    if "Sig" in D:
        Sg = np.abs(D["Sig"]).max(0)
        cor = Sg / np.sqrt(np.outer(np.diag(Sg), np.diag(Sg)))
        print("   max |noise correlation| between shells at separation d = 1, 2, 3: "
              + ", ".join(f"{max(cor[i, i + d] for i in range(len(cor) - d)):.1e}" for d in (1, 2, 3)))
    if a.biasfree:
        rows, cors = [], []
        for name, at in models:
            m, im, _ = model_moments(G, at, dev)
            m = {k: v.detach().cpu().numpy() for k, v in m.items() if k in ("a", "c2", "c3")}
            if name == "x*":
                dn, e = compare_biasfree(name, data, halves, m, G, a.dmin, rows)
                out = a.data.replace(".npz", f"_norm_{a.system}_biasfree.npz")
                np.savez(out, **{**D, **dn}, fit_e=e, dmin=a.dmin)
                print(f"   normalized data (envelope from x*) -> {out}")
            cors.append((name,) + correlations(data, halves, m, G, a.dmin))
        # envelope-free comparison: correlation per bias-free entry, on entries the data reproduce
        _, _, h2, pairs, _, h3 = cors[0]
        rel2 = pairs & (h2 > a.min_repro)
        rel3 = h3 > a.min_repro
        NK = len(G.kappa)
        print(f"\n   envelope-free correlation with the data (c2 pairs |k1-k2| >= {a.dmin}, c3 triples, nonzero-q blocks)")
        print(f"   entries reproduced by the data (half-vs-half corr > {a.min_repro}): c2 {int(rel2.sum())} of "
              f"{int(pairs.sum())} pairs, c3 {int(rel3.sum())} of {rel3.size} triples; "
              f"median half-vs-half corr c2 {np.median(h2[pairs]):.3f}, c3 {np.median(h3):.3f}")
        print("   structure                        c2 mean corr (reproduced / all)   c3 mean corr (reproduced / all)")
        for name, r2, _, _, r3, _ in cors:
            print(f"   {name[:32]:<32} {r2[rel2].mean() if rel2.any() else np.nan:>+8.3f} / {r2[pairs].mean():+.3f}"
                  f"               {r3[rel3].mean() if rel3.any() else np.nan:>+8.3f} / {r3.mean():+.3f}")
        print("   per shell (c2 row mean corr over reproduced pairs): structure: " + " ".join(f"{i + 1:>5}" for i in range(NK)))
        for name, r2, _, _, _, _ in cors:
            row = [r2[i][rel2[i]].mean() if rel2[i].any() else np.nan for i in range(NK)]
            print(f"   {name[:24]:<24} " + " ".join(f"{v:+.2f}" for v in row))
        hrow = [h2[i][pairs[i]].mean() for i in range(NK)]
        print(f"   {'(half vs half)':<24} " + " ".join(f"{v:+.2f}" for v in hrow))
        return
    print(f"{a.data}: {int(D['n_images'])} images, center={D['center']}; fit (s, dB) on shells {a.fit_from + 1}..{len(K)}")
    for name, at in models:
        m, im, _ = model_moments(G, at, dev)
        m = {k: v.detach().cpu().numpy() for k, v in m.items() if k in ("a", "c2", "c3")}  # x* or --model
        s, dB = fit_scale(data["a"], m["a"], K, sl)
        e = None
        if a.envelope == "shell":
            e = shell_envelope(data, m, np.sign((data["a"][0] / m["a"][0]).real) or 1.0)
            s = e[0]
            Bk = -2 * np.log(np.abs(e[1:] / e[0])) / (K[1:] ** 2 - K[0] ** 2)             # implied B per shell
            print(f"   per-shell envelope e(k) / e(k1): " + " ".join(f"{v:.2f}" for v in np.abs(e / e[0])))
            print(f"   implied B (A^2) from shell 1 to k: " + " ".join(f"{v:.0f}" for v in Bk))
        dn, hn = normalize(data, s, dB, K, sh3, e), [normalize(h, s, dB, K, sh3, e) for h in halves]
        L_dm = losses(m, dn, dev)
        L_hh = losses(m, {k: m[k] + hn[0][k] - hn[1][k] for k in m}, dev)      # |h0 - h1|^2 / |model|^2
        print(f"\n== {name}: " + (f"s = {s:+.4g}, dB = {dB:.0f} A^2" if e is None else "per-shell (Wilson) envelope"))
        print("   losses data vs model: " + "  ".join(f"{k} {v:.3e}" for k, v in L_dm.items()))
        print("   half vs half / 4 (sampling floor of the full data): "
              + "  ".join(f"{k} {v / 4:.3e}" for k, v in L_hh.items()))
        if name != "x*":
            continue
        print("   shell  1/kappa(A)  a_data/a_model   c2 diag: rel err (half-split)   c3 diag: rel err (half-split)")
        for i in range(len(K)):
            r_a = (dn["a"][i] / m["a"][i]).real
            e2, n2 = rel(dn["c2"][:, i, i], m["c2"][:, i, i]), rel(hn[0]["c2"][:, i, i], hn[1]["c2"][:, i, i]) / 2
            line = f"   {i + 1:>5}  {1 / K[i]:>10.1f}  {r_a:>14.3f}   {e2:>10.3f} ({n2:.3f})"
            if i in sh3:
                j = int(np.flatnonzero(sh3 == i)[0])
                e3 = rel(dn["c3"][..., j, j, j], m["c3"][..., j, j, j])
                n3 = rel(hn[0]["c3"][..., j, j, j], hn[1]["c3"][..., j, j, j]) / 2
                line += f"          {e3:>10.3f} ({n3:.3f})"
            print(line)
        out = a.data.replace(".npz", f"_norm_{a.system}_{a.envelope}.npz")
        np.savez(out, **{**D, **dn}, fit_s=s, fit_dB=dB, envelope=a.envelope, **({} if e is None else dict(fit_e=e)))
        print(f"   normalized data (with x*'s s, dB; usable as DATA=) -> {out}")


# ------------------------------------------------------------------------------------------------ export for the reward
def cmd_export(a):
    """Data file for MOMENT_M12=corr / MOMENT_M3=corr: noise-calibrated moments plus the masks of the entries used
    (c2 pairs |k1-k2| >= dmin and c3 triples, each reproduced by the two halves above min_repro) and sign3.
    Uses no structure except x*'s radius (grid, as the reward itself, and the calibration's bandlimit)."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    D = dict(np.load(a.data))
    atoms = load_xstar(a.system)
    G = PolarGrid(atoms, int(D["L_img"]), float(D["pixel"]), nb3=int(D["nb3"]), Q3=int(D["Q3"]))
    data, halves = recalibrate(D, G, atoms, dev, a.rp_margin)
    M2b, _ = biasfree_masks(G, a.dmin)
    keep3 = biasfree_masks(G, a.dmin)[1][:, :, 0, 0, 0]
    h2 = pair_corr(halves[0]["c2"], halves[1]["c2"], 0)
    h3 = pair_corr(halves[0]["c3"][keep3], halves[1]["c3"][keep3], 0)
    M2, M3t = M2b[0] & (h2 > a.min_repro), h3 > a.min_repro
    out = a.out or a.data.replace(".npz", "_corr.npz")
    np.savez(out, **data, kappa=D["kappa"], nb3=D["nb3"], Q3=D["Q3"], L=D["L"], shells3=D["shells3"],
             M2=M2, M3t=M3t, sign3=float(a.sign3), dmin=a.dmin, min_repro=a.min_repro, n_images=D["n_images"],
             source=a.data)
    print(f"wrote {out}: {int(M2.sum())} c2 pairs, {int(M3t.sum())} of {M3t.size} c3 triples, sign3 {a.sign3:+.0f}")


# ------------------------------------------------------------------------------------------------ self-test
def cmd_selftest(a):
    """1. Per-particle estimator == defocus-group estimator (moments.image_data.Accumulator) for discrete defoci, on
    the same images (random; an algebraic identity, so ~1e-12). 2. Pure white noise: the bias-corrected moments
    are ~0 (sampling level), the uncorrected ones are not."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)
    G = PolarGrid(load_xstar(a.system), a.box, a.pixel, nb3=8, Q3=16)
    n, dfs = a.n, np.array([8000.0, 12000.0, 16000.0, 20000.0])
    g = rng.integers(0, len(dfs), n)
    one = np.ones(n)
    meta = dict(dfU=dfs[g], dfV=dfs[g], kv=300 * one, cs=2.7 * one, Q=0.1 * one, phase=0 * one, bfac=0 * one, scale=one)
    imgs = rng.standard_normal((n, a.box, a.box))
    S = np.full((a.box, a.box), 1.0)                                         # white, unit variance
    SigF, P = noise_cov_psd(G, S, G.L, G.Q3, dev)
    Sig = SigF[G.L:]
    acc, ex, accg = Accumulator(G, 2, dev), Extra(G, dev), Accumulator(G, len(dfs), dev)
    for i in range(0, n, 256):
        r = np.arange(i, min(i + 256, n))
        H = ctf_of(meta, r, G.kappa)
        I = G.coeffs(imgs[r])
        X = I * H[:, :, None]
        acc.add(X, r % 2); ex.add(H, X[:, :, 0], r % 2); accg.add(I, g[r])
    mine = estimate(acc, ex, G, Sig, P)
    Hg = ctf_of(dict(dfU=dfs, dfV=dfs, kv=300 * np.ones(4), cs=2.7 * np.ones(4), Q=0.1 * np.ones(4), phase=np.zeros(4),
                     bfac=np.zeros(4), scale=np.ones(4)), np.arange(4), G.kappa)
    ref = accg.estimate(Hg, 1.0, Sig, P)
    print("1. per-particle vs group estimator, rel diff: "
          + "  ".join(f"{k} {rel(mine[k], ref[k]):.1e}" for k in ("a", "c2", "c3")), "(expect ~1e-12)")
    raw = estimate(acc, ex, G, 0 * Sig, 0 * P)
    print(f"2. pure noise, |corrected c2| / |uncorrected c2|: {np.linalg.norm(mine['c2']) / np.linalg.norm(raw['c2']):.2e}"
          f" (expect << 1, ~ n^-1/2 = {n ** -0.5:.1e})")

    # 3. c3 noise bias needs a signal (bias = signal x noise x noise): fixed centred blob + noise; the same estimator
    #    on the noiseless images is the reference (the blob is not CTF-modulated, but that is common to both)
    x = (np.arange(a.box) - a.box / 2) * a.pixel
    blob = np.exp(-(x[:, None] ** 2 + x[None] ** 2) / (2 * 8.0**2))
    blob *= 3.0 / blob.std()
    out = {}
    for name, noisy in (("clean", False), ("noisy", True)):
        acc3, ex3 = Accumulator(G, 2, dev), Extra(G, dev)
        for i in range(0, n, 256):
            r = np.arange(i, min(i + 256, n))
            H = ctf_of(meta, r, G.kappa)
            I = G.coeffs(np.broadcast_to(blob, (len(r), a.box, a.box)) + (imgs[r] if noisy else 0))
            X = I * H[:, :, None]
            acc3.add(X, r % 2); ex3.add(H, X[:, :, 0], r % 2)
        out[name] = (estimate(acc3, ex3, G, Sig, P), estimate(acc3, ex3, G, 0 * Sig, 0 * P))
    truth = out["clean"][1]
    cor, unc = out["noisy"]
    print("3. blob + noise, rel err vs noiseless (corrected / uncorrected): "
          + "  ".join(f"{k} {rel(cor[k], truth[k]):.2e} / {rel(unc[k], truth[k]):.2e}" for k in ("a", "c2", "c3")),
          "(expect corrected << uncorrected for c2, c3; a has no bias)")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("stack")
    s.add_argument("--star", required=True); s.add_argument("--root", required=True); s.add_argument("--system", required=True)
    s.add_argument("--center", choices=("none", "origins"), default="none")
    s.add_argument("--ctf", choices=("avg", "mean"), default="avg", help="radial CTF: in-plane average (default) or mean defocus")
    s.add_argument("--no-detrend", dest="detrend", action="store_false", help="keep the per-image background plane")
    s.add_argument("--screen", type=float, default=10.0, help="reject images whose corner or full std is off by this factor")
    s.add_argument("--max-sigma", type=float, default=30.0, help="reject images with a pixel beyond this many reference stds")
    s.add_argument("--n", type=int, default=0, help="random subset of particles (0 = all)")
    s.add_argument("--seed", type=int, default=0); s.add_argument("--batch", type=int, default=512)
    s.add_argument("--nb3", type=int, default=8); s.add_argument("--q3", type=int, default=16)
    s.add_argument("--s3stride", type=int, default=0); s.add_argument("--log-every", type=int, default=10000)
    s.add_argument("--out", default=None)
    c = sp.add_parser("compare")
    c.add_argument("--data", required=True); c.add_argument("--system", required=True)
    c.add_argument("--model", nargs="*", default=[]); c.add_argument("--fit-from", type=int, default=0,
                                                                     help="first shell index used in the (s, dB) fit")
    c.add_argument("--no-calib", action="store_true", help="use the corner noise estimate as is")
    c.add_argument("--diag", dest="biasfree", action="store_false", help="old comparison incl. noise-biased entries")
    c.add_argument("--dmin", type=int, default=2, help="minimum shell separation of the c2 entries used")
    c.add_argument("--min-repro", type=float, default=0.5, help="half-vs-half correlation for an entry to count as reproduced")
    c.add_argument("--rp-margin", type=int, default=6, help="orders beyond 2 pi k r_p treated as signal")
    c.add_argument("--envelope", choices=("gauss", "shell"), default="shell",
                   help="shell: free per-shell envelope from the c2 power (default); gauss: s exp(-dB k^2/4) fitted to a")
    x = sp.add_parser("export")
    x.add_argument("--data", required=True); x.add_argument("--system", required=True)
    x.add_argument("--dmin", type=int, default=2); x.add_argument("--min-repro", type=float, default=0.5)
    x.add_argument("--rp-margin", type=int, default=6); x.add_argument("--sign3", type=float, default=1.0)
    x.add_argument("--out", default=None)
    t = sp.add_parser("selftest")
    t.add_argument("--system", default="7N6N"); t.add_argument("--n", type=int, default=4000)
    t.add_argument("--box", type=int, default=128); t.add_argument("--pixel", type=float, default=1.2)
    a = ap.parse_args()
    if a.cmd == "selftest":
        cmd_selftest(a)
    elif a.cmd == "stack":
        a.out = a.out or f"out/real/{os.path.basename(a.star).split('.')[0]}_{a.center}{'_n%d' % a.n if a.n else ''}.npz"
        cmd_stack(a)
    elif a.cmd == "export":
        cmd_export(a)
    elif a.cmd == "compare":
        cmd_compare(a)


if __name__ == "__main__":
    main()

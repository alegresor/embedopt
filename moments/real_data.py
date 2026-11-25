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


def estimate(acc, ex, G, Sig, P, halves=(0, 1)):
    """Moments from the sums of the given halves (Sig: (L+1, NK, NK) q = 0..L; P: (2 Q3 + 1, S, S) q = -Q3..Q3)."""
    h = list(halves)
    dev = ex.dev
    s1, s2, s3 = (x[h].sum(0) for x in (acc.s1, acc.s2, acc.s3))
    d1, d2, w2, d3, T = (x[h].sum(0) for x in (ex.d1, ex.d2, ex.w2, ex.d3, ex.T))
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


def origin_sign(G, meta, root, idx, fn, rows, n=3000):
    """Sign of the origins that best centres the particles: |sum_j H_j I_j0| on the shells, mid-weighted."""
    sel = rows[:n]
    score = {}
    for sgn in (0.0, 1.0, -1.0):
        num = 0
        for r, x in batches(root, idx, fn, sel, 1024):
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
    acc, ex, psd = Accumulator(G, 2, dev), Extra(G, dev), CornerPSD(meta["L"], meta["p"])
    t0, seen = time.time(), 0
    for r, x in batches(a.root, idx, fn, rows, a.batch):
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
    SigF, P = noise_cov_psd(G, S, G.L, G.Q3, dev)
    Sig = SigF[G.L:]                                                          # q = 0..L
    full, h0, h1 = estimate(acc, ex, G, Sig, P), estimate(acc, ex, G, Sig, P, (0,)), estimate(acc, ex, G, Sig, P, (1,))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez(a.out, **full, kappa=G.kappa, nb3=G.nb3, Q3=G.Q3, L=G.L, shells3=G.shells3, n_images=len(rows), ctf=a.ctf,
             r_max=G.r_max, L_img=meta["L"], pixel=meta["p"], center=a.center, origin_sign=sgn, system=a.system,
             noise_var=float(S.mean()), **{f"h0_{k}": v for k, v in h0.items()}, **{f"h1_{k}": v for k, v in h1.items()})
    print(f"wrote {a.out} ({(time.time() - t0) / 60:.1f} min; corner noise variance {S.mean():.3f})")


# ------------------------------------------------------------------------------------------------ comparison
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
    return sign * np.sqrt(np.maximum(pd, 0) / pm)


def normalize(m, s, dB, kappa, sh3, e=None):
    """Divide out s E(kappa): E = exp(-dB k^2 / 4) with scale s, or a given signed per-shell e (then s, dB unused)."""
    e = s * np.exp(-dB * kappa**2 / 4) if e is None else e
    es = e[sh3]
    return dict(a=m["a"] / e, c2=m["c2"] / (e[:, None] * e[None]),
                c3=m["c3"] / (es[:, None, None] * es[None, :, None] * es[None, None]))


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
    models = [("x*", atoms)] + [(os.path.basename(p), structure_atoms(p)) for p in a.model]
    print(f"{a.data}: {int(D['n_images'])} images, center={D['center']}; fit (s, dB) on shells {a.fit_from + 1}..{len(K)}")
    for name, at in models:
        m, im, _ = model_moments(G, at, dev)
        m = {k: v.detach().cpu().numpy() for k, v in m.items() if k in ("a", "c2", "c3")}  # x* or --model
        s, dB = fit_scale(data["a"], m["a"], K, sl)
        e = None
        if a.envelope == "shell":
            e = shell_envelope(data, m, np.sign((data["a"][0] / m["a"][0]).real))
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
    s.add_argument("--n", type=int, default=0, help="random subset of particles (0 = all)")
    s.add_argument("--seed", type=int, default=0); s.add_argument("--batch", type=int, default=512)
    s.add_argument("--nb3", type=int, default=8); s.add_argument("--q3", type=int, default=16)
    s.add_argument("--s3stride", type=int, default=0); s.add_argument("--log-every", type=int, default=10000)
    s.add_argument("--out", default=None)
    c = sp.add_parser("compare")
    c.add_argument("--data", required=True); c.add_argument("--system", required=True)
    c.add_argument("--model", nargs="*", default=[]); c.add_argument("--fit-from", type=int, default=0,
                                                                     help="first shell index used in the (s, dB) fit")
    c.add_argument("--envelope", choices=("gauss", "shell"), default="shell",
                   help="shell: free per-shell envelope from the c2 power (default); gauss: s exp(-dB k^2/4) fitted to a")
    t = sp.add_parser("selftest")
    t.add_argument("--system", default="7N6N"); t.add_argument("--n", type=int, default=4000)
    t.add_argument("--box", type=int, default=128); t.add_argument("--pixel", type=float, default=1.2)
    a = ap.parse_args()
    if a.cmd == "selftest":
        cmd_selftest(a)
    elif a.cmd == "stack":
        a.out = a.out or f"out/real/{os.path.basename(a.star).split('.')[0]}_{a.center}{'_n%d' % a.n if a.n else ''}.npz"
        cmd_stack(a)
    elif a.cmd == "compare":
        cmd_compare(a)


if __name__ == "__main__":
    main()
"""Image moments (a, c2, c3) estimated from noisy, CTF-modulated pixel images, on the reward's polar grid.

Polar sampling. Each L x L image (pixel p, centre at index L/2, ASPIRE / RELION convention) is evaluated in Fourier
space at the reward's shells kappa_i and n_phi angles by a type-2 NUFFT (finufft),
    I(xi) = s sum_x img(x) exp(-2 pi i xi . x),   x = (n - L/2) p,   s = L p^3,
where s undoes ASPIRE's projection scaling (image = line integral / (L p)) and the pixel area, so that noiseless
CTF-free images give I = Vhat[x*](g xi) in the units of the model. In-plane Fourier coefficients I_q(kappa) are
then taken by an FFT over the angles, as in image_checks.

Data model. I_jq(kappa) = H_j(kappa) S_jq(kappa) + N_jq(kappa): radial CTF H_j (one of a few defocus groups g),
signal S, and the polar coefficients N of white pixel noise of variance sigma^2. For unit-variance noise
    E[N(xi) conj N(xi')] = s^2 D(xi - xi'),  E[N(xi) N(xi')] = s^2 D(xi + xi'),  D(u) = prod_d sum_n e^{-2 pi i u_d p n},
from which Sigma0_q(k1, k2) = E[N_q(k1) conj N_q(k2)] and P0_q(k1, k2) = E[N_q(k1) N_{-q}(k2)] follow by a 2-D DFT
over the two angles (noise_polar_cov).

Estimators (least squares over defocus groups g with counts n_g; sums over images in g):
    a        = sum_g H_g sum_j I_j0 / sum_g n_g H_g^2
    c2_q     = sum_g H_g(k1) H_g(k2) [sum_j I_jq(k1) conj I_jq(k2) - n_g sigma^2 Sigma0_q] / sum_g n_g H_g(k1)^2 H_g(k2)^2
    c3_q1q2  = sum_g H_a H_b H_c [sum_j I_q1(a) I_q2(b) I_q3(c) - sigma^2 bias_g] / sum_g n_g H_a^2 H_b^2 H_c^2,
    bias_g   = d_{q1,0} T_g(a) P0_q2(b,c) + d_{q2,0} T_g(b) P0_q1(a,c) + d_{q3,0} T_g(c) P0_q1(a,b),  T_g = sum_j I_j0,
the terms with one signal and two noise factors (E[I_0 N N] = H a P0; Gaussian noise has no third moment).
All are unbiased; the CTF enters only through its value per shell and group.

Commands:
  python -m moments.image_data stack  --sim DIR                 moments of DIR/particles.star -> DIR/moments_N<n>.npz
  python -m moments.image_data stream --sim DIR --n 1000000      fresh images (same volume, CTF, noise) -> npz
  python -m moments.image_data checks --sim DIR                 calibration, CTF, noise covariance, floors
The .npz files are read by moments.image_moments.load_data (keys a, c2, c3, kappa, L, nb3, Q3, shells3).
"""

import argparse
import glob
import json
import os
import time

import numpy as np
import torch

from moments.image_moments import ImageLoss, ImageMoments, grid, third_shells
R_SAFETY = 1.2  # as moments.reward (r_max = 1.2 * radius of x*)

RES, NKAPPA = 5.0, 16


# ------------------------------------------------------------------------------------------------ grid / sampling
class PolarGrid:
    """The reward's polar grid for a system: shells, n_phi, L, third-order shells; and the NUFFT sampler."""

    def __init__(self, atoms, L_img, p, nb3=8, Q3=16, stride3=0):
        x = atoms.coord.astype(np.float64)
        x -= x.mean(0)
        self.kappa = np.arange(1, NKAPPA + 1) / (NKAPPA * RES)
        self.r_max = R_SAFETY * np.linalg.norm(x, axis=1).max()
        self.shells3 = third_shells(NKAPPA, nb3, stride3)
        self.L, self.LB, self.nphi = grid(self.kappa, self.r_max, self.shells3)
        self.nb3, self.Q3, self.L_img, self.p = nb3, Q3, L_img, p
        self.s = L_img * p**3
        phi = 2 * np.pi * np.arange(self.nphi) / self.nphi
        kx = (self.kappa[:, None] * np.cos(phi)[None]).ravel()
        ky = (self.kappa[:, None] * np.sin(phi)[None]).ravel()
        self.xn, self.yn = 2 * np.pi * ky * p, 2 * np.pi * kx * p  # finufft: first coordinate <-> first axis (y)
        if 2 * np.pi * self.kappa[-1] * p > np.pi:
            raise ValueError("shells beyond the image Nyquist frequency")

    def coeffs(self, imgs):
        """imgs (B, L, L) [y, x] -> in-plane coefficients I_q(kappa), (B, NKAPPA, n_phi) complex128 (q in fft order)."""
        import finufft

        f = np.ascontiguousarray(imgs, dtype=np.complex128)
        v = finufft.nufft2d2(self.xn, self.yn, f, isign=-1, eps=1e-9, modeord=0)
        v = self.s * v.reshape(len(imgs), NKAPPA, self.nphi)
        return np.fft.fft(v, axis=-1) / self.nphi


def ctf_values(defoci, kappa, p):
    """H_g(kappa) for the radial CTFs of the simulation (ASPIRE's own evaluation; omega = 2 pi kappa p)."""
    from aspire.operators import RadialCTFFilter

    om = np.vstack([2 * np.pi * kappa * p, np.zeros_like(kappa)])
    return np.stack([RadialCTFFilter(voltage=300, defocus=float(d), Cs=2.7, alpha=0.1).evaluate(om, pixel_size=p)
                     for d in defoci])


def noise_polar_cov(G, qmax2, Q3m):
    """Unit-variance white-noise covariance of the polar coefficients: Sigma0 (L+1, NK, NK) for q = 0..L, and
    P0 (2 Q3m + 1, S3, S3) for q = -Q3m..Q3m on the third-order shells (P0_q(k1,k2) = E[N_q(k1) N_{-q}(k2)])."""
    nphi, L_img, p, s = G.nphi, G.L_img, G.p, G.s
    phi = 2 * np.pi * np.arange(nphi) / nphi
    e = np.stack([np.cos(phi), np.sin(phi)], 1)
    n = np.arange(-L_img // 2, L_img // 2)

    def g1(u):  # sum_n exp(-2 pi i u p n), vectorized, exact
        th = 2 * np.pi * u * p
        return np.exp(-1j * th[..., None] * n).sum(-1)

    def D(d):  # d (..., 2)
        return g1(d[..., 0]) * g1(d[..., 1])

    NK = len(G.kappa)
    Sig = np.zeros((G.L + 1, NK, NK), complex)
    for i in range(NK):
        for j in range(i, NK):
            M = s**2 * D(G.kappa[i] * e[:, None, :] - G.kappa[j] * e[None, :, :])
            X = np.fft.fft(np.fft.ifft(M, axis=1) * nphi, axis=0) / nphi**2          # E[N_q conj N_q']
            d = X[np.arange(G.L + 1) % nphi, np.arange(G.L + 1) % nphi]
            Sig[:, i, j], Sig[:, j, i] = d, d.conj()
    sh = G.shells3
    qs = np.arange(-Q3m, Q3m + 1)
    P = np.zeros((len(qs), len(sh), len(sh)), complex)
    for a, i in enumerate(sh):
        for b, j in enumerate(sh):
            M = s**2 * D(G.kappa[i] * e[:, None, :] + G.kappa[j] * e[None, :, :])
            X = np.fft.fft2(M) / nphi**2                                              # E[N_q N_q']
            P[:, a, b] = X[qs % nphi, (-qs) % nphi]
    return Sig, P


# ------------------------------------------------------------------------------------------------ accumulation
class Accumulator:
    """Per defocus group: n_g, sum I_0 (all shells), sum I_q conj I_q (q = 0..L), sum of third-order products."""

    def __init__(self, G, n_groups, dev):
        self.G, self.dev = G, dev
        NK, S, Q = len(G.kappa), G.nb3, 2 * G.Q3 + 1
        z = lambda *sh: torch.zeros(sh, dtype=torch.complex128, device=dev)
        self.n = np.zeros(n_groups)
        self.s1, self.s2, self.s3 = z(n_groups, NK), z(n_groups, G.L + 1, NK, NK), z(n_groups, Q, Q, S, S, S)
        qs = np.arange(-G.Q3, G.Q3 + 1)
        self.iq = torch.as_tensor(np.arange(G.L + 1) % G.nphi, device=dev)
        self.i1 = torch.as_tensor(qs % G.nphi, device=dev)
        self.i3 = torch.as_tensor((-qs[:, None] - qs[None, :]) % G.nphi, device=dev)
        self.sh = torch.as_tensor(G.shells3, device=dev)

    def add(self, Iq, groups):
        Iq = torch.as_tensor(Iq, device=self.dev)
        groups = np.asarray(groups)
        for g in np.unique(groups):
            X = Iq[torch.as_tensor(np.flatnonzero(groups == g), device=self.dev)]   # (b, NK, nphi)
            self.n[g] += len(X)
            self.s1[g] += X[:, :, 0].sum(0)
            Y = X[:, :, self.iq]                                                      # (b, NK, L+1)
            self.s2[g] += torch.einsum("baq,bcq->qac", Y, Y.conj())
            Z = X[:, self.sh]                                                         # (b, S, nphi)
            A, A3 = Z[:, :, self.i1], Z[:, :, self.i3]                                # (b,S,Q), (b,S,Q,Q)
            Q, S = A.shape[-1], A.shape[1]
            Yo = torch.einsum("baq,bdr->qradb", A, A).reshape(Q, Q, S * S, -1)
            self.s3[g] += (Yo @ A3.permute(2, 3, 0, 1)).reshape(Q, Q, S, S, S)

    def estimate(self, H, sigma2, Sig0=None, P0=None):
        """H (n_groups, NK) CTF per group and shell; sigma2 pixel noise variance; Sig0, P0 from noise_polar_cov."""
        G, dev = self.G, self.dev
        Ht = torch.as_tensor(H, dtype=torch.float64, device=dev)
        n = torch.as_tensor(self.n, dtype=torch.float64, device=dev)
        keep = n > 0
        Ht, n = Ht[keep], n[keep]
        s1, s2, s3 = self.s1[keep], self.s2[keep], self.s3[keep]
        a = (Ht * s1).sum(0) / (n[:, None] * Ht**2).sum(0)
        if sigma2 > 0:
            s2 = s2 - sigma2 * n[:, None, None, None] * torch.as_tensor(Sig0, device=dev)[None]
        W2 = Ht[:, :, None] * Ht[:, None, :]                                          # (g, k1, k2)
        c2 = (W2[:, None] * s2).sum(0) / (n[:, None, None] * W2**2).sum(0)[None]
        Hs = Ht[:, self.sh]                                                           # (g, S)
        W3 = Hs[:, :, None, None] * Hs[:, None, :, None] * Hs[:, None, None, :]
        if sigma2 > 0:
            P = torch.as_tensor(P0, device=dev)                                       # (2Q3m+1, S, S), q from -Q3m
            Q3, Q3m = G.Q3, (P.shape[0] - 1) // 2
            T = s1[:, self.sh]                                                        # (g, S) sums of I_0
            qs = np.arange(-Q3, Q3 + 1)
            bias = torch.zeros_like(s3)
            i0, Pq = Q3, P[qs + Q3m]                                                  # Pq: (Q, S, S), q in -Q3..Q3
            bias[:, i0] += torch.einsum("ga,qbc->gqabc", T, Pq)                       # q1 = 0: T(a) P_q2(b, c)
            bias[:, :, i0] += torch.einsum("gb,qac->gqabc", T, Pq)                    # q2 = 0: T(b) P_q1(a, c)
            for k, q1 in enumerate(qs):                                               # q3 = 0: q2 = -q1, T(c) P_q1(a,b)
                k2 = int(np.flatnonzero(qs == -q1)[0])
                bias[:, k, k2] += torch.einsum("gc,ab->gabc", T, P[q1 + Q3m])
            s3 = s3 - sigma2 * bias
        c3 = (W3[:, None, None] * s3).sum(0) / (n[:, None, None, None] * W3**2).sum(0)[None, None]
        return {k: v.cpu().numpy() for k, v in dict(a=a, c2=c2, c3=c3).items()}

    def save(self, path, H, sigma2, Sig0, P0, extra=None):
        m = self.estimate(H, sigma2, Sig0, P0)
        G = self.G
        np.savez(path, **m, kappa=G.kappa, nb3=G.nb3, Q3=G.Q3, L=G.L, shells3=G.shells3, n_images=int(self.n.sum()),
                 sigma2=sigma2, **(extra or {}))
        return m


# ------------------------------------------------------------------------------------------------ sources
def read_star(path):
    """Minimal RELION STAR reader for the particles loop: list of dicts (column name -> string)."""
    rows, cols, inside = [], [], False
    for line in open(path):
        s = line.strip()
        if s.startswith("data_"):
            inside = s == "data_particles"
            cols = []
            continue
        if not inside or not s or s == "loop_":
            continue
        if s.startswith("_"):
            cols.append(s.split()[0][1:])
        else:
            rows.append(dict(zip(cols, s.split())))
    return rows


def stack_batches(sim_dir, batch=1024):
    """Yields (images (B, L, L), defocus (B,)) from DIR/particles.star."""
    import mrcfile

    rows = read_star(f"{sim_dir}/particles.star")
    files = {}
    for r in rows:
        idx, fn = r["rlnImageName"].split("@")
        files.setdefault(fn, []).append((int(idx) - 1, float(r["rlnDefocusU"])))
    for fn, lst in files.items():
        with mrcfile.mmap(os.path.join(sim_dir, fn), mode="r") as m:
            data = m.data if m.data.ndim == 3 else m.data[None]
            for i in range(0, len(lst), batch):
                sel = lst[i:i + batch]
                yield np.asarray(data[[k for k, _ in sel]], np.float64), np.array([d for _, d in sel])


def stream_batches(sim_dir, n, seed, batch=1024, clean=False, ctf=True):
    """Yields freshly simulated batches (same volume, CTF set and noise level as DIR/sim.json)."""
    import mrcfile

    from moments.simulate import make_sim

    meta = json.load(open(f"{sim_dir}/sim.json"))
    vol = mrcfile.read(f"{sim_dir}/truth.mrc")
    defoci = meta["defoci"] if ctf else []
    nv = None if clean else meta["noise_var"]
    sim = make_sim(vol, meta["pixel"], n, defoci, seed, nv) if ctf else make_sim_noctf(vol, meta["pixel"], n, seed, nv)
    for i in range(0, n, batch):
        imgs = (sim.clean_images if clean else sim.images)[i:i + batch].asnumpy().astype(np.float64)
        d = np.asarray(defoci)[sim.filter_indices[i:i + batch]] if ctf else np.zeros(len(imgs))
        yield imgs, d


def make_sim_noctf(vol, p, n, seed, noise_var=None):
    from aspire.noise import WhiteNoiseAdder
    from aspire.source import Simulation
    from aspire.volume import Volume

    kw = dict(L=vol.shape[0], n=n, vols=Volume(vol, pixel_size=p), unique_filters=None,
              offsets=np.zeros((n, 2), np.float32), amplitudes=np.ones(n, np.float32), seed=seed, pixel_size=p)
    if noise_var:
        kw["noise_adder"] = WhiteNoiseAdder(var=float(noise_var), seed=seed + 1000)
    return Simulation(**kw)


def accumulate(G, batches, defoci, dev, log_every=100000):
    defoci = np.asarray(defoci) if len(defoci) else np.zeros(1)
    acc = Accumulator(G, len(defoci), dev)
    t0, seen = time.time(), 0
    for imgs, d in batches:
        grp = np.abs(d[:, None] - defoci[None]).argmin(1)
        acc.add(G.coeffs(imgs), grp)
        seen += len(imgs)
        if seen % log_every < len(imgs):
            print(f"    {seen} images ({time.time() - t0:.0f}s)", flush=True)
    return acc


# ------------------------------------------------------------------------------------------------ model side
def model_moments(G, atoms, dev, coords=None):
    """Exact image moments of a structure on G's grid (float64), and its coefficients."""
    from moments import image_checks as ic
    from moments.invariants import form_factors

    x = (atoms.coord if coords is None else coords).astype(np.float64)
    x = x - x.mean(0)
    rho = form_factors(atoms.element, G.kappa, 50.0)
    omega, WYc = ic.s2_quadrature64(G.L)
    A = ic.coefficients64(x, rho, G.kappa, omega, WYc)
    im = ImageMoments(G.kappa, G.r_max, L=G.L, nb3=G.nb3, Q3=G.Q3, dev=dev, dtype=torch.float64)
    m = im.moments(torch.as_tensor(A, dtype=torch.complex128, device=dev))
    return {k: v for k, v in m.items() if v is not None}, im, A


def rel(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)) / np.linalg.norm(np.asarray(b)))


def losses_vs(ref, im, test):
    """Image-space losses (m12 plain/white, m3 block/wblock) of `test` moments against `ref` data moments."""
    T = lambda d: {k: torch.as_tensor(np.asarray(v), dtype=torch.complex128, device=im.beta.device) for k, v in d.items()
                   if k in ("a", "c2", "c3", "C")}
    lo = ImageLoss(T(ref), im, all_variants=True)
    v = lo.variants(T(test))
    return {k: float(v[k]) for k in ("m12_img", "m12w_img", "m3_img", "m3b_img")}


# ------------------------------------------------------------------------------------------------ commands
def setup(sim_dir):
    from moments.simulate import load_xstar

    meta = json.load(open(f"{sim_dir}/sim.json"))
    atoms = load_xstar(meta["system"])
    G = PolarGrid(atoms, meta["L"], meta["pixel"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return meta, atoms, G, dev


def noise_terms(G, meta, sim_dir):
    path = f"{sim_dir}/noise_cov.npz"
    if os.path.isfile(path):
        d = np.load(path)
        return d["Sig0"], d["P0"]
    Sig0, P0 = noise_polar_cov(G, G.L, 2 * G.Q3)
    np.savez(path, Sig0=Sig0, P0=P0)
    return Sig0, P0


def pose_defocus_correlation(sim_dir):
    """Largest normalized mean resultant of the first Euler angle within a defocus group of DIR/particles.star (about 1
    for independent assignment; >> 1 when each group covers a slice of poses, as with ASPIRE's default assignment)."""
    rows = read_star(f"{sim_dir}/particles.star")
    rot = np.deg2rad([float(r["rlnAngleRot"]) for r in rows])
    df = np.array([float(r["rlnDefocusU"]) for r in rows])
    z = [np.abs(np.exp(1j * rot[df == d]).mean()) * np.sqrt((df == d).sum()) for d in np.unique(df)]
    return float(max(z))


def cmd_stack(a):
    meta, atoms, G, dev = setup(a.sim)
    z = pose_defocus_correlation(a.sim)
    if z > 5:
        raise SystemExit(f"{a.sim}/particles.star: poses are correlated with defocus groups (statistic {z:.0f} >> 1; "
                         "stack written before the fix in moments.simulate.make_sim). Its CTF-weighted moments would be "
                         "biased: use `stream` instead, or regenerate the stack.")
    Sig0, P0 = noise_terms(G, meta, a.sim)
    acc = accumulate(G, stack_batches(a.sim, a.batch), meta["defoci"], dev)
    out = f"{a.sim}/moments_N{int(acc.n.sum())}.npz"
    acc.save(out, ctf_values(meta["defoci"], G.kappa, G.p), meta["noise_var"], Sig0, P0)
    print(f"wrote {out}")


def cmd_stream(a):
    meta, atoms, G, dev = setup(a.sim)
    Sig0, P0 = noise_terms(G, meta, a.sim)
    acc = accumulate(G, stream_batches(a.sim, a.n, a.seed, a.batch), meta["defoci"], dev)
    out = f"{a.sim}/moments_N{a.n}_stream.npz"
    acc.save(out, ctf_values(meta["defoci"], G.kappa, G.p), meta["noise_var"], Sig0, P0)
    print(f"wrote {out}")


def cmd_checks(a):
    """1. calibration: noiseless, CTF-free images vs the exact model (scale, discretization);
    2. CTF correction: noiseless, CTF images; 3. noise covariance: analytic vs Monte Carlo;
    4. floors of x* vs the data files present, against the sensitivity of x* + noise."""
    meta, atoms, G, dev = setup(a.sim)
    res = dict(system=meta["system"], snr=meta["snr"], L=G.L, nphi=G.nphi)
    exact, im, A = model_moments(G, atoms, dev)
    ex = {k: v.cpu().numpy() for k, v in exact.items()}
    print(f"=== {meta['system']} SNR {meta['snr']:g} (noise var {meta['noise_var']:.3e}); grid L={G.L}, n_phi={G.nphi}")
    for name, kw in (("noiseless, no CTF", dict(clean=True, ctf=False)), ("noiseless, CTF", dict(clean=True, ctf=True))):
        acc = accumulate(G, stream_batches(a.sim, a.ncal, 777, a.batch, **kw), meta["defoci"] if kw["ctf"] else [0.0],
                         dev, log_every=10**9)
        H = ctf_values(meta["defoci"], G.kappa, G.p) if kw["ctf"] else np.ones((1, len(G.kappa)))
        m = acc.estimate(H, 0.0)
        ratio = np.median((m["a"].real / ex["a"].real)[:8])
        e = {k: rel(m[k], ex[k]) for k in ("a", "c2", "c3")}
        lv = losses_vs(m, im, ex)
        res[name] = dict(scale=float(ratio), **{f"rel_{k}": v for k, v in e.items()}, **lv)
        print(f"  {name:<18} N={a.ncal}: scale a_data/a_model {ratio:.4f}; rel err a {e['a']:.1e} c2 {e['c2']:.1e} "
              f"c3 {e['c3']:.1e}; x* vs data: m12w {lv['m12w_img']:.1e}  m3 {lv['m3_img']:.1e}  m3wblock {lv['m3b_img']:.1e}")
    Sig0, P0 = noise_terms(G, meta, a.sim)
    nm = 2000
    acc = Accumulator(G, 1, dev)
    rng = np.random.default_rng(5)
    for i in range(0, nm, 500):
        acc.add(G.coeffs(rng.standard_normal((500, meta["L"], meta["L"]))), np.zeros(500, int))
    mc2 = np.diagonal(acc.s2[0].cpu().numpy() / nm, axis1=1, axis2=2).real
    an2 = np.diagonal(Sig0, axis1=1, axis2=2).real
    ok = an2 > 1e-3 * an2.max()
    e2 = np.abs(mc2[ok] / an2[ok] - 1)
    res["noise_cov_mc_err"] = dict(median=float(np.median(e2)), max=float(e2.max()))
    print(f"  noise covariance (diagonal, unit pixel variance): Monte Carlo (N={nm}) vs analytic, relative difference "
          f"median {np.median(e2):.1e}, max {e2.max():.1e}  (sampling ~ {np.sqrt(1 / nm):.1e} per entry)")
    rng = np.random.default_rng(0)
    sens = {}
    for s in (0.25, 0.5, 1.0):
        mm, _, _ = model_moments(G, atoms, dev, coords=atoms.coord + s * rng.standard_normal(atoms.coord.shape))
        sens[s] = losses_vs(ex, im, {k: v.cpu().numpy() for k, v in mm.items()})
    res["sensitivity"] = sens
    print("  sensitivity (x* + noise vs exact): " + "; ".join(
        f"{s} A: m12w {v['m12w_img']:.1e} m3 {v['m3_img']:.1e} m3wblock {v['m3b_img']:.1e}" for s, v in sens.items()))
    for f in sorted(glob.glob(f"{a.sim}/moments_N*.npz")):
        d = dict(np.load(f))
        lv = losses_vs(d, im, ex)
        e = {k: rel(d[k], ex[k]) for k in ("a", "c2", "c3")}
        res[os.path.basename(f)] = dict(**lv, **{f"rel_{k}": v for k, v in e.items()})
        print(f"  {os.path.basename(f):<28} rel err a {e['a']:.1e} c2 {e['c2']:.1e} c3 {e['c3']:.1e}; "
              f"x* vs data: m12w {lv['m12w_img']:.1e} m3 {lv['m3_img']:.1e} m3wblock {lv['m3b_img']:.1e}")
    json.dump(res, open(f"{a.sim}/image_data_checks.json", "w"), indent=1, default=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["stack", "stream", "checks"])
    ap.add_argument("--sim", required=True)
    ap.add_argument("--n", type=int, default=1000000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--ncal", type=int, default=20000, help="images for the calibration checks")
    ap.add_argument("--batch", type=int, default=512)
    a = ap.parse_args()
    dict(stack=cmd_stack, stream=cmd_stream, checks=cmd_checks)[a.cmd](a)


if __name__ == "__main__":
    main()

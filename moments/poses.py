"""Non-uniform viewing-direction densities (writeup, section "Unknown pose density").

Poses g = R_z(alpha) R_y(beta) R_z(gamma) with the in-plane angle gamma uniform and the viewing direction
n = g e_z = (sin b cos a, sin b sin a, cos b) drawn from a density nu on S^2 (mean 1 over the sphere). ASPIRE uses the
same convention: an image simulated with rotation matrix R samples Vhat on the plane R (xi_x, xi_y, 0) (checked to
9e-4, the pixel discretization), i.e. its viewing direction is R's third column.

Density used for tests: an antipodally symmetric mixture of von Mises-Fisher distributions,
    nu(n) = sum_k w_k [vMF(n; m_k, c_k) + vMF(n; -m_k, c_k)] / 2,   vMF(n; m, c) = c e^{c (m.n - 1)} / (1 - e^{-2c}) * 2,
normalized to mean 1 over S^2. (Antipodal symmetry: only even spherical-harmonic degrees, as in the writeup's model.)

Model moments under nu, exactly, by quadrature over viewing directions: Gauss-Legendre in cos(beta) (nb nodes), uniform
in alpha (na nodes), weights w_j / (2 na) nu(n_ja); at each node the image's polar Fourier samples are computed from
the atoms by a type-3 NUFFT of the projected atom positions (no spherical-harmonic truncation):
    I(kappa, phi; g) = sum_atoms f_e(kappa) e^{-B kappa^2/4} exp(-2 pi i kappa (x'_n cos phi + y'_n sin phi)),
with (x', y') the first two components of g^T q_n. This is the reference for the convention check and for tests; the
fast model side used in the reward (weighted quadrature on the spherical-harmonic node images) follows in turn 2.

Usage:
    python -m moments.poses check  --sim out/8K9Z_A/noise_snr16 --n 20000   (noiseless check vs the model)
    python -m moments.poses stream --sim out/8K9Z_A/noise_snr16 --n 100000  (data with CTF and noise)
"""

import argparse
import json
import time

import numpy as np

B_FACTOR = 50.0


# ------------------------------------------------------------------------------------------------ densities, sampling
class VMFMixture:
    def __init__(self, means, concentrations, weights):
        m = np.asarray(means, float)
        self.m = m / np.linalg.norm(m, axis=1, keepdims=True)
        self.c = np.asarray(concentrations, float)
        w = np.asarray(weights, float)
        self.w = w / w.sum()

    @classmethod
    def default(cls):
        """Two preferred orientations (top and side views), moderately concentrated."""
        return cls([[0, 0, 1], [1, 0, 0.3]], [4.0, 3.0], [0.6, 0.4])

    def spec(self):
        return dict(means=self.m.tolist(), concentrations=self.c.tolist(), weights=self.w.tolist())

    def __call__(self, n):
        """Density at unit vectors n (..., 3), mean 1 over S^2."""
        out = 0
        for m, c, w in zip(self.m, self.c, self.w):
            norm = 2 * c / (1 - np.exp(-2 * c))  # so that the mean of e^{c (m.n - 1)} norm over S^2 is 1
            out = out + w * 0.5 * norm * (np.exp(c * (n @ m - 1)) + np.exp(c * (-n @ m - 1)))
        return out

    def sample_dirs(self, k, rng):
        comp = rng.choice(len(self.w), k, p=self.w)
        out = np.zeros((k, 3))
        for j in range(len(self.w)):
            sel = np.flatnonzero(comp == j)
            if len(sel):
                out[sel] = sample_vmf(self.m[j], self.c[j], len(sel), rng)
        return out * rng.choice([-1.0, 1.0], k)[:, None]  # antipodal symmetrization


def sample_vmf(m, c, k, rng):
    """Exact sampling of vMF on S^2 (Wood 1994, d = 3: closed-form inverse CDF of t = m.n)."""
    u = rng.uniform(size=k)
    t = 1 + np.log(u + (1 - u) * np.exp(-2 * c)) / c
    phi = rng.uniform(0, 2 * np.pi, k)
    s = np.sqrt(np.clip(1 - t**2, 0, None))
    a = np.array([1.0, 0, 0]) if abs(m[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(m, a); e1 /= np.linalg.norm(e1)
    e2 = np.cross(m, e1)
    return t[:, None] * m[None] + s[:, None] * (np.cos(phi)[:, None] * e1[None] + np.sin(phi)[:, None] * e2[None])


def rotations_from_dirs(n, rng):
    """g = R_z(alpha) R_y(beta) R_z(gamma), viewing direction n = g e_z, gamma uniform."""
    a = np.arctan2(n[:, 1], n[:, 0])
    b = np.arccos(np.clip(n[:, 2], -1, 1))
    g = rng.uniform(0, 2 * np.pi, len(n))
    return Rz(a) @ Ry(b) @ Rz(g)


def Rz(t):
    c, s = np.cos(t), np.sin(t)
    R = np.zeros((len(t), 3, 3)); R[:, 0, 0] = c; R[:, 0, 1] = -s; R[:, 1, 0] = s; R[:, 1, 1] = c; R[:, 2, 2] = 1
    return R


def Ry(t):
    c, s = np.cos(t), np.sin(t)
    R = np.zeros((len(t), 3, 3)); R[:, 0, 0] = c; R[:, 0, 2] = s; R[:, 2, 0] = -s; R[:, 2, 2] = c; R[:, 1, 1] = 1
    return R


# ------------------------------------------------------------------------------------------------ model moments under nu
def form_factor(elements, kappa, B=B_FACTOR):
    import gemmi

    out = {}
    for el in np.unique(elements):
        c = gemmi.Element(str(el)).c4322
        out[el] = sum(a * np.exp(-(b + B) * kappa**2 / 4) for a, b in zip(c.a, c.b))
    return out


def model_moments_nu(q, elements, kappa, nphi, L, shells3, Q3, nu=None, nb=None, na=None, chunk=64, log=None):
    """a (NK,), c2 (L+1, NK, NK), c3 (Q, Q, S, S, S) of atoms q (centred, A) under the viewing density nu (callable on
    unit vectors; None = uniform), by quadrature over viewing directions and exact per-node rendering."""
    import finufft

    nb = nb or L + 10
    na = na or 2 * L + 20
    cb, wb = np.polynomial.legendre.leggauss(nb)
    al = 2 * np.pi * np.arange(na) / na
    be = np.arccos(cb)
    phi = 2 * np.pi * np.arange(nphi) / nphi
    kx = (kappa[:, None] * np.cos(phi)[None]).ravel()
    ky = (kappa[:, None] * np.sin(phi)[None]).ravel()
    ff = form_factor(elements, kappa)
    groups = {el: np.flatnonzero(elements == el) for el in ff}
    NK, S = len(kappa), len(shells3)
    qs = np.arange(-Q3, Q3 + 1)
    i1 = qs % nphi
    i3 = (-qs[:, None] - qs[None, :]) % nphi
    a = np.zeros(NK, complex)
    c2 = np.zeros((L + 1, NK, NK), complex)
    c3 = np.zeros((len(qs), len(qs), S, S, S), complex)
    nodes = [(j, t) for j in range(nb) for t in range(na)]
    t0 = time.time()
    for i0 in range(0, len(nodes), chunk):
        blk = nodes[i0:i0 + chunk]
        A_ = np.array([al[t] for _, t in blk]); B_ = np.array([be[j] for j, _ in blk])
        g = Rz(A_) @ Ry(B_)                                                    # (b, 3, 3)
        n = g[:, :, 2]
        w = np.array([wb[j] / (2 * na) for j, _ in blk]) * (nu(n) if nu is not None else 1.0)
        X = np.zeros((len(blk), NK * nphi), complex)
        for k, gk in enumerate(g):
            P = q @ gk                                                         # rows: g^T q (as row vectors)
            for el, idx in groups.items():
                v = finufft.nufft2d3(2 * np.pi * P[idx, 0], 2 * np.pi * P[idx, 1], np.ones(len(idx), complex),
                                     kx, ky, isign=-1, eps=1e-10)
                X[k] += (ff[el][:, None] * v.reshape(NK, nphi)).ravel()
        Xq = np.fft.fft(X.reshape(len(blk), NK, nphi), axis=-1) / nphi        # (b, NK, nphi), q in fft order
        a += (w[:, None] * Xq[:, :, 0]).sum(0)
        Y = Xq[:, :, np.arange(L + 1) % nphi]                                  # (b, NK, L+1)
        c2 += np.einsum("b,baq,bcq->qac", w, Y, Y.conj())
        Z = Xq[:, shells3]
        A1, A3 = Z[:, :, i1], Z[:, :, i3]                                      # (b,S,Q), (b,S,Q,Q)
        c3 += np.einsum("n,naq,nbr,ncqr->qrabc", w, A1, A1, A3)
        if log and (i0 // chunk) % 50 == 0:
            log(f"    model moments: {i0 + len(blk)}/{len(nodes)} nodes ({time.time() - t0:.0f}s)")
    return dict(a=a, c2=c2, c3=c3)


# ------------------------------------------------------------------------------------------------ data with non-uniform poses
def make_sim_poses(vol, p, rots, defoci=(), seed=0, noise_var=None):
    """ASPIRE simulation with explicit rotations (n, 3, 3), radial CTFs with independent group assignment, white noise."""
    from aspire.noise import WhiteNoiseAdder
    from aspire.operators import RadialCTFFilter
    from aspire.source import Simulation
    from aspire.utils import Rotation
    from aspire.volume import Volume

    n = len(rots)
    fs = [RadialCTFFilter(voltage=300, defocus=float(d), Cs=2.7, alpha=0.1) for d in defoci] or None
    kw = dict(L=vol.shape[0], n=n, vols=Volume(vol, pixel_size=p), unique_filters=fs,
              angles=Rotation.from_matrix(np.asarray(rots, np.float32)).angles.astype(np.float32),
              offsets=np.zeros((n, 2), np.float32), amplitudes=np.ones(n, np.float32), seed=seed, pixel_size=p)
    if fs:
        kw["filter_indices"] = np.random.default_rng(seed + 7919).integers(0, len(fs), n)
    if noise_var:
        kw["noise_adder"] = WhiteNoiseAdder(var=float(noise_var), seed=seed + 1000)
    return Simulation(**kw)


def check_rotations(sim, rots):
    R = np.asarray(sim.rotations, float)
    return float(np.abs(R - rots).max())


# ------------------------------------------------------------------------------------------------ CLI
def cmd_check(a):
    """Noiseless CTF-free images with poses from the default density vs the model with the true density and with the
    uniform density (relative errors of a, c2, c3, and the image-space losses through moments.image_data)."""
    import mrcfile

    from moments import image_data as D

    meta, atoms, G, dev = D.setup(a.sim)
    vol = mrcfile.read(f"{a.sim}/truth.mrc")
    nu = VMFMixture.default()
    rng = np.random.default_rng(a.seed)
    acc = D.Accumulator(G, 1, dev)
    t0 = time.time()
    for i in range(0, a.n, a.batch):
        k = min(a.batch, a.n - i)
        rots = rotations_from_dirs(nu.sample_dirs(k, rng), rng)
        sim = make_sim_poses(vol, meta["pixel"], rots, seed=a.seed + i)
        if i == 0:
            print(f"  ASPIRE keeps the rotations: max diff {check_rotations(sim, rots):.1e}")
        acc.add(G.coeffs(sim.projections[:].asnumpy().astype(np.float64)), np.zeros(k, int))
    data = acc.estimate(np.ones((1, len(G.kappa))), 0.0)
    print(f"  {a.n} images ({time.time() - t0:.0f}s)")
    x = atoms.coord.astype(np.float64); x -= x.mean(0)
    log = lambda s: print(s, flush=True)
    mod_nu = model_moments_nu(x, np.asarray(atoms.element), G.kappa, G.nphi, G.L, G.shells3, G.Q3, nu, log=log)
    mod_u = model_moments_nu(x, np.asarray(atoms.element), G.kappa, G.nphi, G.L, G.shells3, G.Q3, None, log=log)
    ex, im, _ = D.model_moments(G, atoms, dev)
    exu = {k: v.cpu().numpy() for k, v in ex.items()}
    out = dict(n=a.n, density=nu.spec())
    rel = lambda u, v: float(np.linalg.norm(u - v) / np.linalg.norm(v))
    out["quadrature_uniform_vs_analytic"] = {k: rel(mod_u[k], exu[k]) for k in ("a", "c2", "c3")}
    out["data_vs_model_true_density"] = {k: rel(data[k], mod_nu[k]) for k in ("a", "c2", "c3")}
    out["data_vs_model_uniform"] = {k: rel(data[k], mod_u[k]) for k in ("a", "c2", "c3")}
    for k, v in out.items():
        if isinstance(v, dict) and "a" in v:
            print(f"  {k:<34} " + "  ".join(f"{q} {v[q]:.1e}" for q in ("a", "c2", "c3")))
    np.savez(f"{a.sim}/poses_check_model_nu.npz", **mod_nu, kappa=G.kappa, nb3=G.nb3, Q3=G.Q3, L=G.L, shells3=G.shells3)
    np.savez(f"{a.sim}/poses_check_data.npz", **data, kappa=G.kappa, nb3=G.nb3, Q3=G.Q3, L=G.L, shells3=G.shells3)
    json.dump(out, open(f"{a.sim}/poses_check.json", "w"), indent=1)


def cmd_stream(a):
    """Particle images of x* with viewing directions from the default test density (moments.poses.VMFMixture), CTFs
    and white noise as in DIR/sim.json, and their CTF-corrected, noise-debiased moments -> DIR/moments_N<n>_poses.npz."""
    import mrcfile

    from moments import image_data as D

    meta, atoms, G, dev = D.setup(a.sim)
    vol = mrcfile.read(f"{a.sim}/truth.mrc")
    nu = VMFMixture.default()
    defoci = np.asarray(meta["defoci"], float)
    rng = np.random.default_rng(a.seed)
    acc = D.Accumulator(G, len(defoci), dev)
    t0 = time.time()
    for i in range(0, a.n, a.batch):
        k = min(a.batch, a.n - i)
        rots = rotations_from_dirs(nu.sample_dirs(k, rng), rng)
        sim = make_sim_poses(vol, meta["pixel"], rots, defoci, seed=a.seed + i, noise_var=meta["noise_var"])
        acc.add(G.coeffs(sim.images[:].asnumpy().astype(np.float64)), np.asarray(sim.filter_indices))
        if (i // a.batch) % 50 == 0:
            print(f"    {i + k} images ({time.time() - t0:.0f}s)", flush=True)
    Sig0, P0 = D.noise_terms(G, meta, a.sim)
    out = a.out or f"{a.sim}/moments_N{a.n}_poses{a.suffix}.npz"
    acc.save(out, D.ctf_values(defoci, G.kappa, G.p), meta["noise_var"], Sig0, P0, extra=dict(density=json.dumps(nu.spec())))
    print(f"wrote {out} ({time.time() - t0:.0f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "stream"])
    ap.add_argument("--sim", required=True)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dict(check=cmd_check, stream=cmd_stream)[a.cmd](a)


if __name__ == "__main__":
    main()

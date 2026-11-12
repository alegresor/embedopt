"""Simulated cryo-EM particle stacks of x* with ASPIRE: CTF, white noise, uniform poses; RELION-ready STAR + MRCS.

Volume. The electron potential of the heavy atoms of x* (same 5-Gaussian electron form factors and B-factor as the
moment reward), band-limited to the box Nyquist frequency: it is evaluated exactly on the 3D DFT grid and inverse-
transformed, so there is no real-space aliasing,
    rho[n] = (1/p^3) Re ifftn( Vhat(eta_k) (-1)^{k_x + k_y + k_z} ),
    Vhat(eta) = sum_atoms f_e(|eta|) exp(-B |eta|^2 / 4) exp(-2 pi i eta . q),   eta_k = k / (L p),
with q the atom positions relative to the centroid, which sits at voxel index L/2 (the rotation centre of ASPIRE and
RELION). In map coordinates (voxel i at i * p Angstrom, as MapReward and SFC_Torch read MRC files) x* is at
q + (L/2) p; this frame is written to xstar_mapframe.pdb.

Images. ASPIRE projects the volume (image = (1/L) sum_z vol, i.e. the line integral divided by L p), applies a radial
CTF (300 kV, Cs 2.7 mm, 10% amplitude contrast; NDEF defocus values evenly spaced in [DEF_MIN, DEF_MAX] Angstrom,
default 0.8-2.0 um, with a 192 x 1.5 A box so that the CTF-delocalized signal stays inside the box,
assigned at random) and adds white noise of variance noise_var = P_signal / SNR, P_signal the mean squared pixel of
the clean CTF-modulated images (estimated on 2000 images). Poses uniform on SO(3), no shifts, unit amplitudes.

Outputs in --out: truth.mrc (the volume), xstar_mapframe.pdb, particles.star + particles_*.mrcs (if --n > 0),
sim.json (all parameters, noise_var, defocus values).

Usage: python -m moments.simulate --system 8K9Z_A --snr 0.0625 --n 100000 --out out/8K9Z_A/noise_snr16
"""

import argparse
import json
import os
import time

import numpy as np

BENCH = "examples/synthetic_map_benchmark"
B_FACTOR = 50.0


def load_xstar(system, bench=BENCH):
    import biotite.structure.io as bio

    a = bio.load_structure(f"{bench}/pdb_clean/{system}.pdb")
    return a[a.element != "H"]


def potential_volume(q, elements, L, p, B=B_FACTOR, dev=None, chunk=256):
    """Band-limited electron potential (z, y, x) of atoms at q (Angstrom, relative to the box centre), float32."""
    import gemmi
    import torch

    dev = dev or ("cuda" if torch.cuda.is_available() else "cpu")
    k = torch.fft.fftfreq(L, d=p, dtype=torch.float64, device=dev)                 # Angstrom^-1, fft order
    kint = torch.round(torch.fft.fftfreq(L, dtype=torch.float64, device=dev) * L)  # integer index
    e2 = k[:, None, None] ** 2 + k[None, :, None] ** 2 + k[None, None, :] ** 2   # [kz, ky, kx]
    V = torch.zeros((L, L, L), dtype=torch.complex128, device=dev)
    q = torch.as_tensor(np.asarray(q, np.float64), device=dev)
    elements = np.asarray(elements)
    for el in np.unique(elements):
        c = gemmi.Element(str(el)).c4322
        f = sum(a * torch.exp(-(b + B) * e2 / 4) for a, b in zip(c.a, c.b))      # f_e(|eta|) exp(-B eta^2/4)
        S = torch.zeros((L * L, L), dtype=torch.complex128, device=dev)
        for idx in np.array_split(np.flatnonzero(elements == el), max(1, int(np.ceil((elements == el).sum() / chunk)))):
            qq = q[torch.as_tensor(idx, device=dev)]
            Ex, Ey, Ez = (torch.exp(-2j * np.pi * qq[:, d:d + 1] * k[None, :]) for d in range(3))
            T = (Ez[:, :, None] * Ey[:, None, :]).reshape(len(idx), L * L)             # (n, z*y)
            S += T.T @ Ex                                                            # (z*y, x)
        V += f * S.reshape(L, L, L)
    sign = (-1.0) ** (kint[:, None, None] + kint[None, :, None] + kint[None, None, :])
    rho = torch.fft.ifftn(V * sign).real / p**3
    return rho.float().cpu().numpy()


def write_mrc(path, vol, p):
    import mrcfile

    with mrcfile.new(path, overwrite=True) as m:
        m.set_data(np.ascontiguousarray(vol, np.float32))
        m.voxel_size = p
        m.header.origin = (0.0, 0.0, 0.0)


def write_pdb(path, atoms, box):
    from biotite.structure.io.pdb import PDBFile

    a = atoms.copy()
    a.box = np.eye(3) * box
    f = PDBFile()
    f.set_structure(a)
    f.write(path)


def make_sim(vol, p, n, defoci, seed, noise_var=None):
    from aspire.noise import WhiteNoiseAdder
    from aspire.operators import RadialCTFFilter
    from aspire.source import Simulation
    from aspire.volume import Volume

    L = vol.shape[0]
    fs = [RadialCTFFilter(voltage=300, defocus=float(d), Cs=2.7, alpha=0.1) for d in defoci]
    # Filter (defocus) assignment from an independent stream: ASPIRE draws it with the same seed as the rotations, which
    # makes each defocus group cover a different slice of the first Euler angle (pose-defocus correlation). That is
    # harmless for reconstruction but biases any CTF-weighted moment estimator.
    fi = np.random.default_rng(seed + 7919).integers(0, len(fs), n) if fs else None
    kw = dict(L=L, n=n, vols=Volume(vol, pixel_size=p), unique_filters=fs, filter_indices=fi,
              offsets=np.zeros((n, 2), np.float32), amplitudes=np.ones(n, np.float32), seed=seed, pixel_size=p)
    if noise_var is not None and noise_var > 0:
        kw["noise_adder"] = WhiteNoiseAdder(var=float(noise_var), seed=seed + 1000)
    return Simulation(**kw)


def signal_power(vol, p, defoci, n=2000, seed=12345):
    sim = make_sim(vol, p, n, defoci, seed)
    return float(np.mean([np.mean(sim.clean_images[i:i + 500].asnumpy() ** 2) for i in range(0, n, 500)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True)
    ap.add_argument("--snr", type=float, required=True, help="signal-to-noise ratio (inf: no noise)")
    ap.add_argument("--n", type=int, default=100000, help="images to write (0: volume and metadata only)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--L", type=int, default=192)
    ap.add_argument("--pixel", type=float, default=1.5)
    ap.add_argument("--def-min", type=float, default=8000.0)
    ap.add_argument("--def-max", type=float, default=20000.0)
    ap.add_argument("--ndef", type=int, default=7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=1024)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    atoms = load_xstar(a.system)
    cen = atoms.coord.astype(np.float64).mean(0)
    q = atoms.coord.astype(np.float64) - cen
    rmax = np.linalg.norm(q, axis=1).max()
    if rmax > 0.45 * a.L * a.pixel:
        raise ValueError(f"particle radius {rmax:.0f} A does not fit a {a.L} x {a.pixel} A box")
    # CTF delocalization: at defocus df and frequency k the signal spreads by lambda df k; keep it inside the box,
    # otherwise ASPIRE's circular CTF wraps it around and the off-grid (polar) Fourier samples are biased
    lam = 12.2643 / np.sqrt(300e3 * (1 + 0.978466e-6 * 300e3))
    deloc = lam * a.def_max * (1.0 / 5.0)
    if rmax + deloc > 0.5 * a.L * a.pixel:
        print(f"WARNING: radius {rmax:.0f} A + CTF delocalization {deloc:.0f} A (at 5 A, {a.def_max:.0f} A defocus) "
              f"exceeds the half box {0.5 * a.L * a.pixel:.0f} A: enlarge --L or lower --def-max")
    vol = potential_volume(q, atoms.element, a.L, a.pixel)
    write_mrc(f"{a.out}/truth.mrc", vol, a.pixel)
    moved = atoms.copy()
    moved.coord = (q + a.L / 2 * a.pixel).astype(np.float32)
    write_pdb(f"{a.out}/xstar_mapframe.pdb", moved, a.L * a.pixel)
    defoci = np.linspace(a.def_min, a.def_max, a.ndef)
    ps = signal_power(vol, a.pixel, defoci)
    nv = 0.0 if not np.isfinite(a.snr) else ps / a.snr
    meta = dict(system=a.system, snr=a.snr, n=a.n, L=a.L, pixel=a.pixel, defoci=defoci.tolist(), seed=a.seed,
                signal_power=ps, noise_var=nv, centroid=cen.tolist(), particle_radius=float(rmax), B=B_FACTOR,
                voltage=300, Cs=2.7, alpha=0.1)
    json.dump(meta, open(f"{a.out}/sim.json", "w"), indent=1)
    print(f"{a.system}: volume {a.L}^3 at {a.pixel} A ({time.time() - t0:.0f}s), radius {rmax:.1f} A, "
          f"signal power {ps:.3e}, SNR {a.snr:g} -> noise var {nv:.3e}")
    if a.n > 0:
        sim = make_sim(vol, a.pixel, a.n, defoci, a.seed, nv)
        cwd = os.getcwd()
        os.chdir(a.out)  # STAR image paths relative to the STAR file
        try:
            sim.save("particles.star", batch_size=a.batch, overwrite=True)
        finally:
            os.chdir(cwd)
        print(f"  wrote {a.n} images to {a.out}/particles.star ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()

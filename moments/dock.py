"""Rigid-body docking of an atomic model into a cryo-EM map, without the true structure (replaces MapReward's oracle
Kabsch frame), and map cleaning.

Docking. The model's band-limited potential (moments.simulate.potential_volume, centred in the box) and the map are
band-passed between LOWPASS and HIGHCUT (15) A; without removing the lowest frequencies, the overall mass distribution
lets many wrong rotations correlate at ~0.99. For every rotation R of a quasi-uniform set (coarse search at half resolution) the
rotated model volume V_R(r) = V(R^T r) is resampled (trilinear) and the translation d maximizing the cross-correlation
sum_r map(r) V_R(r - d) is found by FFT; both volumes are zero-mean and unit-norm, so the score is a correlation
coefficient. The TOP best distinct rotations (low resolution has near-symmetric false optima) are each refined on two
nested local grids at full resolution, low-passed to REFINE A, and the best refined pose is kept. Both hands are tried: the map and its
mirror image (x -> -x about the box centre); an ab-initio reconstruction has a random hand, and the mirrored map is
used when it fits better. Output: the docked model (coordinates R (q - centroid) + centre + d p, in the map frame:
voxel i at i * p Angstrom, as MapReward reads MRC files), with CRYST1 = box, and the (possibly mirrored) map.

Cleaning (as in Li et al. for real maps): voxels below a threshold are set to zero; the threshold keeps the
VOL_FACTOR x expected particle volume (1.21 A^3 / Da x 110 Da per residue) of highest density, after a light low-pass.

Usage:
  python -m moments.dock dock --map MAP.mrc --model PRED.cif --out DIR    -> DIR/docked.pdb, DIR/map.mrc, DIR/dock.json
  python -m moments.dock dock --map truth.mrc --model x*.pdb --out DIR --truth xstar_mapframe.pdb   (self-test: RMSD)
  python -m moments.dock fsc --map A.mrc --ref B.mrc [--out fsc.json]
  python -m moments.dock threshold --map A.mrc --nres N          (threshold enclosing the expected particle volume)
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from moments.simulate import potential_volume, write_mrc, write_pdb


def read_map(path):
    import mrcfile

    with mrcfile.open(path) as m:
        return np.asarray(m.data, np.float32).copy(), float(m.voxel_size.x)


def lowpass(vol, p, res):
    L = vol.shape[0]
    k = np.fft.fftfreq(L, d=p)
    k2 = k[:, None, None] ** 2 + k[None, :, None] ** 2 + k[None, None, :] ** 2
    return np.fft.ifftn(np.fft.fftn(vol) * np.exp(-k2 * res**2 * 2.0)).real.astype(np.float32)  # Gaussian, ~res


def fourier_crop(vol, f):
    """Downsample by an integer factor f by cropping the centred spectrum (L divisible by f)."""
    L = vol.shape[0]
    n = L // f
    F = np.fft.fftshift(np.fft.fftn(vol))
    c = L // 2
    F = F[c - n // 2:c + n // 2, c - n // 2:c + n // 2, c - n // 2:c + n // 2]
    return (np.fft.ifftn(np.fft.ifftshift(F)).real / f**3).astype(np.float32)


def mirror(vol):
    """x -> -x about the centre index L/2 (array axes z, y, x)."""
    return np.roll(vol[:, :, ::-1], 1, axis=2)


def normalize(v):
    v = v - v.mean()
    return v / np.linalg.norm(v)


class Rotator:
    """Resamples a centred volume (z, y, x) under rotations of the coordinates: V_R(r) = V(R^T r), centre index L/2."""

    def __init__(self, vol, dev):
        self.L = L = vol.shape[0]
        self.v = torch.as_tensor(vol, device=dev)[None, None]
        n = torch.arange(L, device=dev, dtype=torch.float32) - L / 2
        z, y, x = torch.meshgrid(n, n, n, indexing="ij")
        self.r = torch.stack([x, y, z], -1).reshape(-1, 3)                                # (xyz) per output voxel
        self.dev = dev

    def __call__(self, R):
        """R (B, 3, 3) acting on (x, y, z) -> (B, L, L, L)."""
        R = torch.as_tensor(R, dtype=torch.float32, device=self.dev)
        rin = torch.einsum("nj,bjk->bnk", self.r, R)                                     # r_in = R^T r_out (rows)
        u = (2 * (rin + self.L / 2) + 1) / self.L - 1                                      # align_corners=False
        g = u.reshape(len(R), self.L, self.L, self.L, 3)
        out = torch.nn.functional.grid_sample(self.v.expand(len(R), -1, -1, -1, -1), g, mode="bilinear",
                                              padding_mode="zeros", align_corners=False)
        return out[:, 0]


def search(target, model, rots, dev, batch=None):
    """Best (score, rotation index, shift (z, y, x) in voxels) of model rotated by rots against target."""
    L = target.shape[0]
    batch = batch or int(max(1, min(32, 2.5e7 // L**3)))                                  # ~3 GB of GPU memory
    Ft = torch.fft.fftn(torch.as_tensor(normalize(target), device=dev))
    rot = Rotator(model, dev)
    best = (-np.inf, -1, None)
    scores = np.zeros(len(rots))
    for i in range(0, len(rots), batch):
        V = rot(rots[i:i + batch])
        V = V - V.mean(dim=(1, 2, 3), keepdim=True)
        V = V / V.flatten(1).norm(dim=1)[:, None, None, None]
        dims = (-3, -2, -1)                                                                # not the batch axis
        cc = torch.fft.ifftn(Ft[None] * torch.fft.fftn(V, dim=dims).conj(), dim=dims).real  # cc[d] = sum t(r) V(r - d)
        m, idx = cc.flatten(1).max(dim=1)
        scores[i:i + batch] = m.cpu().numpy()
        j = int(m.argmax())
        if m[j].item() > best[0]:
            d = np.array(np.unravel_index(int(idx[j]), (L, L, L)))
            d = np.where(d >= L // 2, d - L, d)
            best = (m[j].item(), i + j, d)
    return best, scores


def local_rotations(R0, max_deg, step_deg):
    from scipy.spatial.transform import Rotation

    a = np.deg2rad(np.arange(-max_deg, max_deg + 1e-9, step_deg))
    rv = np.stack(np.meshgrid(a, a, a, indexing="ij"), -1).reshape(-1, 3)
    return (Rotation.from_rotvec(rv) * Rotation.from_matrix(R0)).as_matrix()


def distinct_top(rots, scores, k, min_deg):
    """Indices of the k best-scoring rotations, each at least min_deg from the ones already picked."""
    from scipy.spatial.transform import Rotation

    picked = []
    for i in np.argsort(scores)[::-1]:
        if all(Rotation.from_matrix(rots[i] @ rots[j].T).magnitude() > np.deg2rad(min_deg) for j in picked):
            picked.append(i)
        if len(picked) == k:
            break
    return picked


def bandpass(v, p, lo, hi):
    """Keep roughly the band between hi and lo Angstrom (difference of Gaussian low-passes)."""
    return lowpass(v, p, lo) - lowpass(v, p, hi)


def dock(map_vol, p, atoms, dev, lowpass_res=8.0, n_coarse=50000, coarse=2, seed=0, top=12, refine_res=6.0, highcut=25.0,
         hands=("original", "mirrored"), log=print):
    """Coarse global search (low-passed to lowpass_res, downsampled by `coarse`), then local refinement of the `top`
    best distinct rotations at full resolution (low-passed to refine_res); both hands."""
    from scipy.spatial.transform import Rotation

    L = map_vol.shape[0]
    q = atoms.coord.astype(np.float64)
    cen = q.mean(0)
    model = potential_volume(q - cen, atoms.element, L, p, dev=dev)
    # band-pass (drop > highcut A): the overall ellipsoidal mass otherwise makes many wrong rotations score ~0.99
    mlp, mrf = bandpass(model, p, lowpass_res, highcut), bandpass(model, p, refine_res, highcut)
    rots = Rotation.random(n_coarse, random_state=seed).as_matrix()
    spacing = np.rad2deg((8 * np.pi**2 / n_coarse) ** (1 / 3))                          # typical gap of the coarse set
    out = {}
    t0 = time.time()
    for hand in hands:
        tv = map_vol if hand == "original" else mirror(map_vol)
        tlp, trf = bandpass(tv, p, lowpass_res, highcut), bandpass(tv, p, refine_res, highcut)
        _, scores = search(fourier_crop(tlp, coarse), fourier_crop(mlp, coarse), rots, dev)
        best = (-np.inf, None, None)
        for i in distinct_top(rots, scores, top, 2 * spacing):
            R = rots[i]
            for maxd, step in ((spacing, spacing / 4), (spacing / 4, spacing / 12)):
                loc = local_rotations(R, maxd, step)
                (s, j, d), _ = search(trf, mrf, loc, dev)
                R = loc[j]
            if s > best[0]:
                best = (s, R, d)
        out[hand] = dict(score=float(best[0]), R=best[1], d=best[2])
        log(f"  {hand:<8} hand: correlation {best[0]:.4f}  ({time.time() - t0:.0f}s)")
    hand = max(out, key=lambda h: out[h]["score"])
    R, d = out[hand]["R"], out[hand]["d"]
    X = (q - cen) @ R.T + (L / 2 + d[::-1]) * p                                         # d is (z, y, x)
    return X, hand, out


def clean_map(vol, p, n_res, vol_factor=2.0, res=6.0):
    """Zero voxels outside the vol_factor x expected-volume region of highest (low-passed) density."""
    v_exp = 1.21 * 110.0 * n_res * vol_factor
    nvox = int(min(vol.size, v_exp / p**3))
    lp = lowpass(vol, p, res)
    t = np.sort(lp.ravel())[::-1][nvox]
    out = np.where(lp > t, vol, 0.0).astype(np.float32)
    return out, float(t), nvox


def fsc(v1, v2, p, nbins=None):
    """Fourier shell correlation of two maps on the same grid; returns (shell centres in 1/A, FSC)."""
    L = v1.shape[0]
    F1, F2 = np.fft.fftn(v1), np.fft.fftn(v2)
    k = np.fft.fftfreq(L, d=p)
    r = np.sqrt(k[:, None, None] ** 2 + k[None, :, None] ** 2 + k[None, None, :] ** 2)
    nb = nbins or L // 2
    edges = np.linspace(0, 0.5 / p, nb + 1)
    idx = np.digitize(r.ravel(), edges) - 1
    ok = (idx >= 0) & (idx < nb)
    num = np.bincount(idx[ok], (F1.ravel() * F2.ravel().conj())[ok].real, nb)
    d1 = np.bincount(idx[ok], (np.abs(F1.ravel()) ** 2)[ok], nb)
    d2 = np.bincount(idx[ok], (np.abs(F2.ravel()) ** 2)[ok], nb)
    return (edges[1:] + edges[:-1]) / 2, num / np.sqrt(d1 * d2 + 1e-300)


def resolution_at(k, f, thr):
    """Resolution (A) where the FSC first drops below thr; inf if already below at the first shell, Nyquist if never."""
    below = np.flatnonzero(f < thr)
    if not len(below):
        return float(1 / k[-1])
    return float(1 / k[below[0]]) if below[0] > 0 else float("inf")


def expected_threshold(vol, p, n_res, vol_factor=1.0, res=6.0):
    """Density threshold enclosing vol_factor x the expected particle volume (1.21 A^3/Da x 110 Da/residue)."""
    nvox = int(min(vol.size - 1, 1.21 * 110.0 * n_res * vol_factor / p**3))
    return float(np.sort(lowpass(vol, p, res).ravel())[::-1][nvox])


def main():
    import biotite.structure.io as bio

    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dock", help="dock a model into a map (both hands), clean the map")
    d.add_argument("--map", required=True)
    d.add_argument("--model", required=True, help="atomic model to dock (e.g. the unguided prior prediction)")
    d.add_argument("--out", required=True)
    d.add_argument("--truth", default=None, help="x* in the frame of the map (for tests: CA RMSD without superposition)")
    d.add_argument("--lowpass", type=float, default=8.0)
    d.add_argument("--highcut", type=float, default=25.0)
    d.add_argument("--n-coarse", type=int, default=50000)
    d.add_argument("--top", type=int, default=12)
    d.add_argument("--no-clean", action="store_true")
    f = sub.add_parser("fsc", help="FSC of a map against a reference map on the same grid")
    f.add_argument("--map", required=True)
    f.add_argument("--ref", required=True)
    f.add_argument("--out", default=None, help="json file")
    t = sub.add_parser("threshold", help="density threshold enclosing the expected particle volume")
    t.add_argument("--map", required=True)
    t.add_argument("--nres", type=int, required=True)
    t.add_argument("--factor", type=float, default=1.0)
    a = ap.parse_args()
    if a.cmd == "threshold":
        vol, p = read_map(a.map)
        print(f"{expected_threshold(vol, p, a.nres, a.factor):.6g}")
        return
    if a.cmd == "fsc":
        v1, p = read_map(a.map)
        v2, _ = read_map(a.ref)
        k, fc = fsc(v1, v2, p)
        r5, r143 = resolution_at(k, fc, 0.5), resolution_at(k, fc, 0.143)
        cc = float(np.corrcoef(v1.ravel(), v2.ravel())[0, 1])
        print(f"FSC({a.map}, {a.ref}): 0.5 at {r5:.2f} A, 0.143 at {r143:.2f} A; real-space CC {cc:.3f}")
        if a.out:
            json.dump(dict(fsc05=r5, fsc0143=r143, cc=cc, k=k.tolist(), fsc=fc.tolist()), open(a.out, "w"), indent=1)
        return
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    vol, p = read_map(a.map)
    atoms = bio.load_structure(a.model)
    atoms = atoms[atoms.element != "H"]
    print(f"docking {a.model} ({len(atoms)} atoms) into {a.map} ({vol.shape[0]}^3, {p} A), {a.n_coarse} rotations, {dev}")
    X, hand, res = dock(vol, p, atoms, dev, a.lowpass, a.n_coarse, top=a.top, highcut=a.highcut)
    docked = atoms.copy()
    docked.coord = X.astype(np.float32)
    write_pdb(f"{a.out}/docked.pdb", docked, vol.shape[0] * p)
    mv = vol if hand == "original" else mirror(vol)
    info = dict(map=a.map, model=a.model, hand=hand, **{h: dict(score=r["score"], R=r["R"].tolist(), d=r["d"].tolist())
                                                          for h, r in res.items()})
    if not a.no_clean:
        n_res = int((atoms.atom_name == "CA").sum())
        mv, thr, nvox = clean_map(mv, p, n_res)
        info.update(clean_threshold=thr, clean_voxels=nvox)
    write_mrc(f"{a.out}/map.mrc", mv, p)
    if a.truth:
        tr = bio.load_structure(a.truth)
        tr = tr[tr.element != "H"]
        ca_t, ca_d = tr.coord[tr.atom_name == "CA"], X[atoms.atom_name == "CA"]
        if len(ca_t) == len(ca_d):
            info["rmsd_to_truth_frame"] = float(np.sqrt(((ca_t - ca_d) ** 2).sum(1).mean()))
            print(f"  CA RMSD of the docked model to x* in the map frame (no superposition): {info['rmsd_to_truth_frame']:.2f} A")
    json.dump(info, open(f"{a.out}/dock.json", "w"), indent=1)
    print(f"  hand {hand}; wrote {a.out}/docked.pdb, {a.out}/map.mrc")


if __name__ == "__main__":
    main()

"""How accurately can the viewing density be fitted from predicted structures? A sweep over fitting variants.

For the final pass of a moment chain (scored by moments.check against the non-uniform data, sel_<run>.csv), take the
k lowest-m3_img seeds. Fit theta by the exact QP (moments.pose_moments.fit_theta) jointly from those structures,
J(theta) = sum_s J_s(theta), with
    terms : both (m12 / m12_0 + m3 / m3_0), m12 only, m3 only        (normalizers per structure at theta = 0)
    ridge : 0, 1e-3, 1e-2, 1e-1 x mean(diag H) (shrinks towards the uniform density)
    P     : 2 (5 parameters) or 4 (14)
    k     : 1, 3, 5 structures
Accuracy is measured in x*'s frame: every structure is rotated onto x* (Kabsch on CA) before the diagnostic fit and the
error |nu_fit - nu_true| is computed on the sphere (L2 over directions, relative to |nu_true - 1|), which is comparable
across P. For use in a chain the same variant is fitted with the structures rotated onto the best one instead (the
frame of POSE_REF): the fits differ by one global rotation.
Usage: python -m moments.pose_sweep --sim out/8K9Z_A/noise_snr16 --run <final run> [--out DIR]
Writes DIR/sweep.json (all variants) and, with --write VARIANT (e.g. "k3_both_r1e-2_P4"), DIR/theta_<VARIANT>.json and
DIR/ref_<VARIANT>.cif (the best structure, for POSE_REF).
"""

import argparse
import glob
import itertools
import json
import os
import shutil

import numpy as np
import pandas as pd
import torch

from moments import image_data as D
from moments.image_moments import ImageLoss, ImageMoments, load_data, tables
from moments.pose_moments import PoseMoments, SHDensity, collocation, combine, dirs, fit_theta
from moments.poses import VMFMixture


def kabsch_onto(st, ref_ca):
    ca = st.coord[st.atom_name == "CA"].astype(float)
    P, Q = ca - ca.mean(0), ref_ca - ref_ca.mean(0)
    U, _, Vt = np.linalg.svd(P.T @ Q)
    R = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
    out = st.copy()
    out.coord = ((st.coord - ca.mean(0)) @ R.T + ref_ca.mean(0)).astype(np.float32)
    return out


def main():
    import biotite.structure.io as bio

    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", required=True)
    ap.add_argument("--run", required=True, help="final-pass run directory name (under out/<SYS>)")
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--kmax", type=int, default=5)
    ap.add_argument("--write", default=None, help="variant to write theta + ref for (e.g. k3_both_r1e-2_P4)")
    a = ap.parse_args()
    meta, atoms, G, dev = D.setup(a.sim)
    sys_ = meta["system"]
    out = a.out or f"out/{sys_}/pose_sweep_{a.run}"
    os.makedirs(out, exist_ok=True)
    data = a.data or f"{a.sim}/moments_N100000_poses.npz"
    sel = pd.read_csv(f"out/{sys_}/sel_{a.run}.csv").sort_values("m3_img")
    seeds = list(sel.seed.values[:a.kmax])
    files = [sorted(glob.glob(f"out/{sys_}/{a.run}/*/{s}/predictions/*_sample_0.cif"))[0] for s in seeds]
    print(f"{sys_}: top {len(files)} seeds by m3_img: {seeds} (core {np.round(sel.core.values[:a.kmax], 2).tolist()})")
    sts = []
    for f in files:
        st = bio.load_structure(f)
        sts.append(st[st.element != "H"])
    xca = atoms.coord[atoms.atom_name == "CA"].astype(float)
    best_ca = sts[0].coord[sts[0].atom_name == "CA"].astype(float)
    im = ImageMoments(G.kappa, G.r_max, L=G.L, nb3=G.nb3, Q3=G.Q3, dev=dev, dtype=torch.float64)
    T = lambda x: torch.as_tensor(x, dtype=torch.complex128, device=dev)
    loss = ImageLoss(load_data(data, im, None), im, m12="white", m3="wblock")
    beta, *_ = tables(im.L, im.LB, im.nphi, im.Q3)
    Binv = T(np.linalg.inv(beta.T / (4 * np.pi)))
    withC = lambda m: dict(m, C=(Binv @ m["c2"].reshape(m["c2"].shape[0], -1)).reshape(m["c2"].shape))
    nu_true = VMFMixture.default()
    cb, wb = np.polynomial.legendre.leggauss(40)
    al = 2 * np.pi * np.arange(80) / 80
    nq = dirs(np.arccos(cb)[:, None], al[None, :])
    wq = np.broadcast_to(wb[:, None] / 2 / 80, nq.shape[:2])
    vt = nu_true(nq)
    err = lambda dens, th: float(np.sqrt((wq * (dens.values(th, nq) - vt) ** 2).sum() / (wq * (vt - 1) ** 2).sum()))
    pms, denss, bases = {}, {}, {}
    for P in (2, 4):
        pms[P] = PoseMoments(im.L, im.LB, im.nphi, len(G.kappa), im.shells3, im.Q3, P=P, dtype=np.complex128,
                             backend="torch", device=dev)
        denss[P] = SHDensity(P)
        for frame, ref in (("xstar", xca), ("best", best_ca)):
            for i, st in enumerate(sts):
                with torch.no_grad():
                    bases[(P, frame, i)] = pms[P].basis_moments(T(D.coef_of(G, kabsch_onto(st, ref))), denss[P])
        print(f"  bases for P = {P} done", flush=True)

    def fit(P, frame, k, terms, rr):
        dens = denss[P]
        Js = []
        for i in range(k):
            b = bases[(P, frame, i)]
            with torch.no_grad():
                m0 = withC(combine(b, np.zeros(dens.K)))
                n12, n3 = float(loss.m12(m0)), float(loss.m3(m0))
            w12, w3 = {"both": (1, 1), "m12": (1, 0), "m3": (0, 1)}[terms]
            Js.append((b, n12, n3, w12, w3))

        def J_of(th):  # a sum of per-structure quadratic forms: evaluated through each structure's basis
            tot = 0.0
            with torch.no_grad():
                for b, n12, n3, w12, w3 in Js:
                    m = combine(b, th)
                    tot += (w12 * float(loss.m12(withC(m))) / n12 if w12 else 0.0) + \
                           (w3 * float(loss.m3(m)) / n3 if w3 else 0.0)
            return tot
        # fit_theta needs a single basis: pass a dummy basis of theta unit vectors and evaluate J_of on combinations
        unit = [dict(t=np.zeros(dens.K))] + [dict(t=e) for e in np.eye(dens.K)]
        th, info = fit_theta(unit, lambda m: J_of(m["t"]), dens, collocation(400), ridge_rel=rr)
        return th, info

    rows = []
    for P, k, terms, rr in itertools.product((2, 4), (1, 3, 5), ("both", "m12", "m3"), (0.0, 1e-3, 1e-2, 1e-1)):
        if k > len(sts):
            continue
        name = f"k{k}_{terms}_r{rr:g}_P{P}"
        th, info = fit(P, "xstar", k, terms, rr)
        e = err(denss[P], th)
        rows.append(dict(variant=name, P=P, k=k, terms=terms, ridge=rr, err=e, min_nu=info["min_nu"]))
        print(f"  {name:<22} density error (x* frame) {e:.3f}  min nu {info['min_nu']:.3f}", flush=True)
    e_true4 = err(denss[4], denss[4].project(nu_true)); e_true2 = err(denss[2], denss[2].project(nu_true))
    print(f"  (true density projected: P = 2 {e_true2:.3f}, P = 4 {e_true4:.3f}; uniform density 1.000)")
    json.dump(dict(system=sys_, run=a.run, seeds=[str(s) for s in seeds], rows=rows,
                   projection_err={"2": e_true2, "4": e_true4}), open(f"{out}/sweep.json", "w"), indent=1)
    if a.write:
        r = next(r for r in rows if r["variant"] == a.write)
        th, info = fit(r["P"], "best", r["k"], r["terms"], r["ridge"])
        json.dump(th.tolist(), open(f"{out}/theta_{a.write}.json", "w"))
        shutil.copy(files[0], f"{out}/ref_{a.write}.cif")
        print(f"  wrote {out}/theta_{a.write}.json and ref_{a.write}.cif (frame of seed {seeds[0]})")


if __name__ == "__main__":
    main()

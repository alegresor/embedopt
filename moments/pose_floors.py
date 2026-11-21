"""Floors of x* against moments from non-uniform-pose images, through the reward's own losses (whitened m12, wblock m3):
    uniform  : the reward's uniform-pose model (ImageMoments), and the same with PoseMoments at nu = 1 (must agree)
    oracle   : PoseMoments with the true test density (moments.poses.VMFMixture.default(), alpha-Fourier to P = 12)
    fitted   : PoseMoments with theta from the exact QP using x* (P = 4), and its error against the true density's
               degree-4 projection; plus the true degree-4 projection itself
    x* + 0.5 A under the oracle density, for scale. Also the uniform model against the uniform-pose data, for reference.
Usage: python -m moments.pose_floors --sim out/8K9Z_A/noise_snr16
"""

import argparse

import numpy as np
import torch

from moments import image_data as D
from moments.image_moments import ImageLoss, ImageMoments, load_data, tables
from moments.pose_moments import PoseMoments, SHDensity, collocation, combine, fit_theta
from moments.poses import VMFMixture


def fit_from_structure(a):
    """theta (P = 4) by the exact QP for a given structure file (e.g. the best current prediction) -> json."""
    import json

    import biotite.structure.io as bio

    meta, atoms, G, dev = D.setup(a.sim)
    st = bio.load_structure(a.model)
    st = st[st.element != "H"]
    data = a.data or f"{a.sim}/moments_N100000_poses.npz"
    im = ImageMoments(G.kappa, G.r_max, L=G.L, nb3=G.nb3, Q3=G.Q3, dev=dev, dtype=torch.float64)
    T = lambda x: torch.as_tensor(x, dtype=torch.complex128, device=dev)
    A = T(D.coef_of(G, st))
    loss = ImageLoss(load_data(data, im, None), im, m12="white", m3="wblock")
    beta, *_ = tables(im.L, im.LB, im.nphi, im.Q3)
    Binv = T(np.linalg.inv(beta.T / (4 * np.pi)))
    withC = lambda m: dict(m, C=(Binv @ m["c2"].reshape(m["c2"].shape[0], -1)).reshape(m["c2"].shape))
    dens = SHDensity(4)
    pm4 = PoseMoments(im.L, im.LB, im.nphi, len(G.kappa), im.shells3, im.Q3, P=4, dtype=np.complex128,
                      backend="torch", device=dev)
    with torch.no_grad():
        basis = pm4.basis_moments(A, dens)
        m0 = withC(combine(basis, np.zeros(dens.K)))
        n12, n3 = float(loss.m12(m0)), float(loss.m3(m0))
        J = lambda m: float(loss.m12(withC(m))) / n12 + float(loss.m3(m)) / n3
        th, info = fit_theta(basis, J, dens, collocation(400))
    th_true = dens.project(VMFMixture.default())
    json.dump(th.tolist(), open(a.out, "w"))
    print(f"theta from {a.model}: |theta - theta_true| / |theta_true| = "
          f"{np.linalg.norm(th - th_true) / np.linalg.norm(th_true):.3f}, min nu {info['min_nu']:.3f} -> {a.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", required=True)
    ap.add_argument("--data", default=None, help="non-uniform moments (default SIM/moments_N100000_poses.npz)")
    ap.add_argument("--model", default=None, help="fit theta from this structure (with --out) instead of floors")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.model:
        return fit_from_structure(a)
    meta, atoms, G, dev = D.setup(a.sim)
    data = a.data or f"{a.sim}/moments_N100000_poses.npz"
    im = ImageMoments(G.kappa, G.r_max, L=G.L, nb3=G.nb3, Q3=G.Q3, dev=dev, dtype=torch.float64)
    T = lambda x: torch.as_tensor(x, dtype=torch.complex128, device=dev)
    A = T(D.coef_of(G, atoms))
    rng = np.random.default_rng(0)
    Ap = T(D.coef_of(G, _perturbed(atoms, 0.5, rng)))
    lossN = ImageLoss(load_data(data, im, None), im, m12="white", m3="wblock")
    lossU = ImageLoss(load_data(f"{a.sim}/moments_N100000.npz", im, None), im, m12="white", m3="wblock")
    beta, *_ = tables(im.L, im.LB, im.nphi, im.Q3)
    Binv = T(np.linalg.inv(beta.T / (4 * np.pi)))
    withC = lambda m: dict(m, C=(Binv @ m["c2"].reshape(m["c2"].shape[0], -1)).reshape(m["c2"].shape))
    f = lambda lo, m: (float(lo.m12(m)), float(lo.m3(m)))
    rows = []
    with torch.no_grad():
        mu = im.moments(A)
        rows.append(("uniform model vs uniform-pose data (reference)", f(lossU, mu)))
        rows.append(("uniform model (ImageMoments) vs non-uniform data", f(lossN, mu)))
        pm12 = PoseMoments(im.L, im.LB, im.nphi, len(G.kappa), im.shells3, im.Q3, P=12, dtype=np.complex128,
                           backend="torch", device=dev)
        m1 = withC(pm12.moments(A, *pm12.uniform()))
        rows.append(("PoseMoments, nu = 1, vs non-uniform data", f(lossN, m1)))
        rel = float((m1["c3"] - mu["c3"]).abs().pow(2).sum().sqrt() / mu["c3"].abs().pow(2).sum().sqrt())
        rel2 = float((m1["c2"] - mu["c2"]).abs().pow(2).sum().sqrt() / mu["c2"].abs().pow(2).sum().sqrt())
        nu = VMFMixture.default()
        mo = withC(pm12.moments(A, *pm12.of_function(nu)))
        rows.append(("oracle density vs non-uniform data", f(lossN, mo)))
        mop = withC(pm12.moments(Ap, *pm12.of_function(nu)))
        rows.append(("x* + 0.5 A, oracle density (scale)", f(lossN, mop)))
        dens = SHDensity(4)
        pm4 = PoseMoments(im.L, im.LB, im.nphi, len(G.kappa), im.shells3, im.Q3, P=4, dtype=np.complex128,
                          backend="torch", device=dev)
        th_true = dens.project(nu)
        mt = withC(pm4.moments(A, *pm4.of_theta(dens, th_true)))
        rows.append(("true density, degree-4 projection", f(lossN, mt)))
        basis = pm4.basis_moments(A, dens)
        m0 = withC(combine(basis, np.zeros(dens.K)))
        n12, n3 = f(lossN, m0)
        J = lambda m: float(lossN.m12(withC(m))) / n12 + float(lossN.m3(m)) / n3
        th, info = fit_theta(basis, J, dens, collocation(400))
        mf = withC(combine(basis, th))
        rows.append(("QP-fitted density (from x*)", f(lossN, mf)))
    print(f"=== {meta['system']}: floors of x* (m12 whitened, m3 wblock), non-uniform data {data}")
    print(f"  PoseMoments at nu = 1 vs ImageMoments: c2 {rel2:.1e}, c3 {rel:.1e} (should be ~1e-7 or less)")
    for lab, (a12, a3) in rows:
        print(f"  {lab:<52} m12 {a12:.2e}   m3 {a3:.2e}")
    e = np.linalg.norm(th - th_true) / np.linalg.norm(th_true)
    print(f"  QP from x*: |theta - theta_true| / |theta_true| = {e:.2f}; min nu {info['min_nu']:.3f}; "
          f"loss {info['loss0']:.3e} -> {info['loss']:.3e}")
    print(f"  theta_true = {np.round(th_true, 3).tolist()}\n  theta_fit  = {np.round(th, 3).tolist()}")


def _perturbed(atoms, s, rng):
    b = atoms.copy()
    b.coord = (b.coord + s * rng.standard_normal(b.coord.shape)).astype(np.float32)
    return b


if __name__ == "__main__":
    main()

"""Where does the CTF calibration error come from? Run on a simulation directory (moments.simulate):

    python -m moments.ctf_diag --sim out/8K9Z_A/noise_snr16

Same poses, with and without CTF (noiseless), for each defocus value of the simulation:
  1. polar sampling: per-shell relative error of the polar Fourier samples of CTF images against H(kappa) times the
     samples of the CTF-free projections. Nonzero only through CTF wrap-around in the box (circular CTF) and leakage.
  2. estimator: moments from the CTF images against moments from the analytically CTF-weighted CTF-free samples
     (same poses, same groups): isolates 1. from everything else.
  3. moments of each defocus group alone (CTF-corrected) against the exact model moments of x*, and the same for the
     CTF-free images: shows whether the error grows with defocus.
"""

import argparse
import json

import mrcfile
import numpy as np

from moments import image_data as D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", required=True)
    ap.add_argument("--n", type=int, default=2100, help="images (split evenly over the defocus values)")
    ap.add_argument("--batch", type=int, default=150)
    a = ap.parse_args()
    from aspire.operators import RadialCTFFilter
    from aspire.source import Simulation
    from aspire.volume import Volume

    meta, atoms, G, dev = D.setup(a.sim)
    vol, p, defoci = mrcfile.read(f"{a.sim}/truth.mrc"), meta["pixel"], np.asarray(meta["defoci"])
    ng, n = len(defoci), a.n - a.n % len(defoci)
    fi = np.arange(n) % ng
    kw = dict(L=vol.shape[0], n=n, vols=Volume(vol, pixel_size=p), offsets=np.zeros((n, 2), np.float32),
              amplitudes=np.ones(n, np.float32), seed=4242, pixel_size=p)
    s0 = Simulation(**kw, unique_filters=None)
    s1 = Simulation(**kw, angles=s0.angles, unique_filters=[RadialCTFFilter(voltage=300, defocus=float(d), Cs=2.7,
                                                                          alpha=0.1) for d in defoci], filter_indices=fi)
    H = D.ctf_values(defoci, G.kappa, p)
    A1, A2, A0 = D.Accumulator(G, ng, dev), D.Accumulator(G, ng, dev), D.Accumulator(G, ng, dev)
    num, den = np.zeros((ng, len(G.kappa))), np.zeros((ng, len(G.kappa)))
    for i in range(0, n, a.batch):
        g = fi[i:i + a.batch]
        X0 = G.coeffs(s0.projections[i:i + a.batch].asnumpy())
        X1 = G.coeffs(s1.clean_images[i:i + a.batch].asnumpy())
        ref = X0 * H[g][:, :, None]
        for k in range(ng):
            s = g == k
            num[k] += (np.abs(X1[s] - ref[s]) ** 2).sum((0, 2))
            den[k] += (np.abs(ref[s]) ** 2).sum((0, 2))
        A1.add(X1, g)
        A2.add(ref, g)
        A0.add(X0, g)
    print(f"=== {meta['system']}: box {vol.shape[0]} x {p} A, {n} images, defocus {defoci.min():.0f}-{defoci.max():.0f} A")
    print("1. polar-sample error of CTF images vs H x CTF-free samples, per defocus (A) at shells 1, 4, 8, 12, 16:")
    out = dict(defoci=defoci.tolist(), polar_err={})
    for k, d in enumerate(defoci):
        e = np.sqrt(num[k] / den[k])
        tot = float(np.sqrt(num[k].sum() / den[k].sum()))
        out["polar_err"][f"{d:.0f}"] = dict(total=tot, per_shell=e.tolist())
        print(f"   {d:7.0f}: total {tot:.1e};  " + "  ".join(f"{e[j]:.1e}" for j in (0, 3, 7, 11, 15)))
    m1, m2 = A1.estimate(H, 0.0), A2.estimate(H, 0.0)
    e2 = {k: D.rel(m1[k], m2[k]) for k in ("a", "c2", "c3")}
    out["estimator"] = e2
    print("2. CTF images vs CTF-weighted CTF-free samples (same poses): " + "  ".join(f"{k} {v:.1e}" for k, v in e2.items()))
    exact, im, _ = D.model_moments(G, atoms, dev)
    ex = {k: v.cpu().numpy() for k, v in exact.items()}
    print("3. one defocus group at a time vs the exact model (sampling error ~ the CTF-free row):")
    out["per_group"] = {}
    m0 = A0.estimate(np.ones((ng, len(G.kappa))), 0.0)
    print(f"   all, CTF-free      : " + "  ".join(f"{k} {D.rel(m0[k], ex[k]):.1e}" for k in ("a", "c2", "c3")))
    for k, d in enumerate(defoci):
        Hk = np.zeros_like(H)
        Hk[k] = H[k]
        mk = A1.estimate(np.where(Hk == 0, 1e-12, Hk) * (np.arange(ng)[:, None] == k), 0.0)
        r = {q: D.rel(mk[q], ex[q]) for q in ("a", "c2", "c3")}
        out["per_group"][f"{d:.0f}"] = r
        print(f"   defocus {d:7.0f}  : " + "  ".join(f"{q} {v:.1e}" for q, v in r.items()))
    json.dump(out, open(f"{a.sim}/ctf_diag.json", "w"), indent=1)


if __name__ == "__main__":
    main()
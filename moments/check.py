"""Analysis of oracle-moment experiments: one row per final prediction, aggregated per run
(base / map_lr* / moment2_lr* / moment3_lr*[_res*]).

Per prediction: CA RMSD to x* (and mirrored), 80%-core RMSD, fraction of residues within 2/4/8 A after the core fit,
Rg and max radius, and, scored identically for every run (5 A, default bispectrum settings): the second-order moment
loss m12, the third-order bispectrum loss m3, and the whitened per-degree error of C_l.
Outputs: a printed per-run summary (median [min, max] and success rates), a CSV with every row, and a figure.

Usage: python -m moments.check --ref <x*.pdb> --runs out/SYS/base out/SYS/moment2_lr0.1 ... --out fig.png --csv rows.csv
"""

import argparse
import glob
import json
import os

import biotite.structure.io as bio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from moments.reward import OracleMomentReward

SIGMAS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
SUCCESS = {"rmsd<5": ("rmsd", 5.0), "core<2": ("core", 2.0)}  # success-rate definitions (A)


def ca(a):
    return a.coord[a.atom_name == "CA"]


def kabsch_apply(P, Q, idx, sign=1.0):
    """Superimpose Q onto P using rows idx; sign=-1 gives the improper (mirror) solution."""
    p0, q0 = P[idx].mean(0), Q[idx].mean(0)
    U, _, Vt = np.linalg.svd((Q[idx] - q0).T @ (P[idx] - p0))
    D = np.diag([1, 1, sign * np.sign(np.linalg.det(U @ Vt))])
    return (Q - q0) @ U @ D @ Vt + p0


def structure_metrics(ref, pred, frac=0.8, iters=20):
    P, Q = ca(ref), ca(pred)
    if len(P) != len(Q):
        return None
    allidx = np.arange(len(P))
    rmsd = np.sqrt(((kabsch_apply(P, Q, allidx) - P) ** 2).sum(1).mean())
    mirror = np.sqrt(((kabsch_apply(P, Q, allidx, -1.0) - P) ** 2).sum(1).mean())
    idx = allidx
    for _ in range(iters):  # iterative core fit on the best `frac` of residues
        d = np.linalg.norm(kabsch_apply(P, Q, idx) - P, axis=1)
        idx = np.argsort(d)[: int(frac * len(P))]
    x = pred.coord[pred.element != "H"]
    r = np.linalg.norm(x - x.mean(0), axis=1)
    m = dict(rmsd=rmsd, mirror=mirror, core=np.sqrt(np.mean(d[idx] ** 2)), rg=np.sqrt((r**2).mean()), rmax=r.max())
    m.update({f"within{c}": (d < c).mean() for c in (2, 4, 8)})
    return m, d


@torch.no_grad()
def scores(rew, x):
    """x: (N_atom, 3) in model order. Returns m12, m3, whitened per-degree errors of C_l."""
    x = torch.as_tensor(x, dtype=torch.float32, device=rew.dev)[rew.keep]
    A = rew.inv.coefficients(x - x.mean(0), rew.rho)
    _, C = rew.inv.power(A)
    d = 1 / sum(cr.diagonal().real for cr in rew.C_ref).sqrt()  # whitening by the radial power of x*
    D = d[:, None] * d[None, :]
    per_l = np.array([((D * (c - cr)).abs().pow(2).sum() / (D * cr).abs().pow(2).sum()).item() for c, cr in zip(C, rew.C_ref)])
    return rew.m12(A).item(), rew.m3(A).item(), per_l


def run_checks():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--resolution", type=float, default=5.0)
    ap.add_argument("--out", default="out/check.png")
    ap.add_argument("--csv", default="out/summary.csv")
    a = ap.parse_args()
    ref = bio.load_structure(a.ref)
    runs = [r.rstrip("/") for r in a.runs]
    color = {os.path.basename(r): f"C{i}" for i, r in enumerate(runs)}
    fig, ax = plt.subplots(2, 4, figsize=(22, 10))

    # Traces (map runs log the map reward, not the moment loss: different scale)
    for run in runs:
        g = os.path.basename(run)
        for i, f in enumerate(sorted(glob.glob(f"{run}/**/reward_history.json", recursive=True))):
            h = json.load(open(f))
            step = [e["step"] for e in h]
            ax[0, 0].semilogy(step, [-e["reward"] for e in h], color=color[g], lw=0.6, label=g if i == 0 else None)

    ax[0, 0].set(title="loss as optimized (moment2: m12, moment3: m12 m3^lambda, map: -map reward)", xlabel="diffusion step")
    ax[0, 1].set(title="does m3 predict accuracy?", xlabel="m3 (bispectrum loss)", ylabel="80%-core RMSD (A)")

    # Sensitivity of the moment loss to per-atom noise on x*
    rew_ref = OracleMomentReward(a.ref, ref, resolution=a.resolution, order=3)
    rng = np.random.default_rng(0)
    sens = np.array([scores(rew_ref, ref.coord + s * rng.standard_normal(ref.coord.shape))[:2] for s in SIGMAS])
    ax[1, 0].loglog(SIGMAS, sens[:, 0], "ko-", label="m12: x* + noise")
    ax[1, 0].loglog(SIGMAS, sens[:, 1], "ks--", label="m3: x* + noise")
    print("\nSensitivity (x* + per-atom noise):")
    for s, (m12_, m3_) in zip(SIGMAS, sens):
        print(f"  sigma {s:5.2f} A   m12 {m12_:.3e}   m3 {m3_:.3e}")

    # Final predictions; one scoring reward for all (Protenix atom order is identical across runs and seeds)
    rows, per_l_all, dev_all, rew_pred, n_atom = [], {}, {}, None, None
    for run in runs:
        g = os.path.basename(run)
        for f in sorted(glob.glob(f"{run}/**/predictions/*.cif", recursive=True)):
            pred = bio.load_structure(f)
            if rew_pred is None or len(pred) != n_atom:
                rew_pred, n_atom = OracleMomentReward(a.ref, pred, resolution=a.resolution, order=3), len(pred)
            sm = structure_metrics(ref, pred)
            if sm is None:
                print(f"  skipping {f}: CA count mismatch")
                continue
            m, dev = sm
            m12, m3, per_l = scores(rew_pred, pred.coord)
            seed = next((p for p in f.split(os.sep) if p.startswith("seed_")), "?")
            rows.append(dict(run=g, seed=seed, m12=m12, m3=m3, **m))
            per_l_all.setdefault(g, []).append(per_l)
            dev_all.setdefault(g, []).append(dev)
    df = pd.DataFrame(rows)
    for name, (col, thr) in SUCCESS.items():
        df[name] = df[col] < thr
    os.makedirs(os.path.dirname(a.csv) or ".", exist_ok=True)
    df.to_csv(a.csv, index=False)

    # Summary table
    order = [os.path.basename(r) for r in runs if os.path.basename(r) in set(df.run)]
    print(f"\nx*: Rg {structure_metrics(ref, ref)[0]['rg']:.2f} A\n")
    fmt = lambda s: f"{s.median():6.2f} [{s.min():5.2f}, {s.max():5.2f}]"
    print(f"{'run':<20}{'n':>3}  {'RMSD med [min,max]':<22}{'core med [min,max]':<22}{'within 2A':>10}{'Rg':>7}"
          f"{'m12':>11}{'m3':>11}{'rmsd<5':>8}{'core<2':>8}")
    for g in order:
        d = df[df.run == g]
        print(f"{g:<20}{len(d):>3}  {fmt(d.rmsd):<22}{fmt(d.core):<22}{d.within2.median():>9.0%}{d.rg.median():>7.1f}"
              f"{d.m12.median():>11.2e}{d.m3.median():>11.2e}{d['rmsd<5'].mean():>8.0%}{d['core<2'].mean():>8.0%}")
    print(f"\nRows written to {a.csv}")

    # Aggregate panels
    for j, (col, title) in enumerate([("rmsd", "CA RMSD to x* (A)"), ("core", "80%-core RMSD (A)")]):
        axi = ax[0, 2 + j]
        axi.boxplot([df[df.run == g][col] for g in order], showfliers=False)
        axi.set_xticks(range(1, len(order) + 1), order, rotation=45)
        for i, g in enumerate(order):
            v = df[df.run == g][col]
            axi.scatter(np.full(len(v), i + 1) + 0.08 * rng.standard_normal(len(v)), v, s=12, color=color[g], zorder=3)
        axi.set(title=title)
    for g in order:
        d = df[df.run == g]
        ax[1, 0].scatter(np.full(len(d), SIGMAS[-1] * 1.3), d.m12, s=10, color=color[g], label=g)
        ax[1, 0].scatter(np.full(len(d), SIGMAS[-1] * 1.6), d.m3, s=10, color=color[g], marker="s")
        ax[1, 1].semilogy(np.median(per_l_all[g], axis=0), color=color[g], label=g)
        ax[1, 2].semilogy(ref.res_id[ref.atom_name == "CA"], np.median(dev_all[g], axis=0), color=color[g], lw=0.8, label=g)
        ax[1, 3].loglog(d.m12, d.core, "o", color=color[g], ms=4, label=g)
        ax[0, 1].loglog(d.m3, d.core, "s", color=color[g], ms=4, label=g)
    ax[1, 0].set(title="m12 (o) and m3 (s): x* + noise vs predictions (right)", xlabel="per-atom noise sigma (A)")
    ax[1, 1].set(title="median whitened relative error of C_l", xlabel="degree l")
    ax[1, 2].set(title="median per-residue CA deviation after core fit (A)", xlabel="res_id (x*)")
    ax[1, 3].set(title="does m12 predict accuracy?", xlabel="m12 (second-order loss)", ylabel="80%-core RMSD (A)")
    for axi in ax.ravel():
        if axi.get_legend_handles_labels()[0]:
            axi.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(a.out, dpi=130)
    print(f"Saved {a.out}")


if __name__ == "__main__":
    run_checks()

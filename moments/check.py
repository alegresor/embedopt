"""Analysis of oracle-moment experiments: one row per final prediction, aggregated per run.

Runs: base, map_lr*, moment{2,3}[dw]_lr*[_res*]. Every prediction is scored identically (5 A, default bispectrum
settings, no Debye-Waller weighting): CA RMSD to x* (and mirrored), 80%-core RMSD, fraction of residues within 2/4/8 A
after the core fit, Rg, max radius, second-order loss m12 and third-order loss m3.
Figure: colors distinguish runs, line styles / markers distinguish methods (base, map, moment2, moment3; DW runs bold).
Per-step traces (t, s_t, m12, m3, RMSD) come from moment_trace.json, written by moment runs.
With --data (exact | image_checks .npz), every prediction is also scored by the image-space losses against those data
(columns m12_img, m3_img), and the m12 panel of the figure is replaced by m3_img.

Usage: python -m moments.check --ref <x*.pdb> --runs out/SYS/base out/SYS/moment3_lr0.1 ... --out fig.png --csv rows.csv
       [--data exact|out/SYS/image_moments_N1000000.npz]
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
DW_WIDTHS = [0.0, 2.0, 4.0, 8.0]  # posterior widths s for the basin test
SUCCESS = {"rmsd<5": ("rmsd", 5.0), "core<2": ("core", 2.0)}
STYLE = {"base": ("-", "x"), "map": ("-", "o"), "moment2": ("--", "s"), "moment3": (":", "^")}
IMG = ("m12_img", "m3_img")


def method(run):
    return next(k for k in ("moment2", "moment3", "map", "base") if run.startswith(k))


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
def losses(rew, x, s=0.0):
    m12, m3 = rew.losses(torch.as_tensor(x, dtype=torch.float32, device=rew.dev), s, space="invariant", third=True)
    return m12.item(), m3.item()


@torch.no_grad()
def img_losses(rew, x):
    m12, m3 = rew.losses(torch.as_tensor(x, dtype=torch.float32, device=rew.dev), space="image", third=True)
    return m12.item(), m3.item()


def run_checks():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--resolution", type=float, default=5.0)
    ap.add_argument("--out", default="out/check.png")
    ap.add_argument("--csv", default="out/summary.csv")
    ap.add_argument("--data", default=None, help="image-space scoring data: exact | image_checks .npz")
    a = ap.parse_args()
    ref = bio.load_structure(a.ref)
    runs = [r.rstrip("/") for r in a.runs if os.path.isdir(r)]
    names = [os.path.basename(r) for r in runs]
    cmap = plt.get_cmap("tab20" if len(runs) > 10 else "tab10")
    color = {g: cmap(i % cmap.N) for i, g in enumerate(names)}
    sty = lambda g: dict(color=color[g], ls=STYLE[method(g)][0], lw=2.0 if "dw" in g else 1.2)
    fig, ax = plt.subplots(3, 4, figsize=(24, 15))
    rng = np.random.default_rng(0)

    # Sensitivity and Debye-Waller basin test on x* + per-atom noise
    make_reward = lambda pred: OracleMomentReward(a.ref, pred, resolution=a.resolution, order=3, dw=False, space="invariant",
                                         data=a.data)
    rew_ref = make_reward(ref)
    noisy = [ref.coord + s * rng.standard_normal(ref.coord.shape) for s in SIGMAS]
    print("\nSensitivity (x* + per-atom noise), unweighted and with Debye-Waller width s:")
    for s in DW_WIDTHS:
        v = np.array([losses(rew_ref, x, s) for x in noisy])
        print(f"  s = {s:4.1f} A   m12: " + " ".join(f"{u:.1e}" for u in v[:, 0]) + "   m3: " + " ".join(f"{u:.1e}" for u in v[:, 1]))
        ax[1, 0].loglog(SIGMAS, v[:, 1] / v[0, 1], "s-", label=f"m3, s = {s:g} A")
        if s == 0:
            sens = v
    print(f"  (columns: sigma = {SIGMAS} A)")
    if a.data:
        v = np.array([img_losses(rew_ref, x) for x in noisy])
        print(f"  image space ({a.data})   m12: " + " ".join(f"{u:.1e}" for u in v[:, 0]) + "   m3: " + " ".join(f"{u:.1e}" for u in v[:, 1]))
        sens_img = v
    ax[1, 0].set(title="basin test: m3(x* + noise) / m3(x* + 0.25 A), by DW width s", xlabel="per-atom noise sigma (A)")

    # Final predictions; one scoring reward for all (Protenix atom order is identical across runs and seeds)
    rows, dev_all, rew_pred, n_atom = [], {}, None, None
    for run, g in zip(runs, names):
        for f in sorted(glob.glob(f"{run}/**/predictions/*.cif", recursive=True)):
            pred = bio.load_structure(f)
            if rew_pred is None or len(pred) != n_atom:
                rew_pred, n_atom = make_reward(pred), len(pred)
            sm = structure_metrics(ref, pred)
            if sm is None:
                print(f"  skipping {f}: CA count mismatch")
                continue
            m, dev = sm
            m12, m3 = losses(rew_pred, pred.coord)
            seed = next((p for p in f.split(os.sep) if p.startswith("seed_")), "?")
            extra = dict(zip(IMG, img_losses(rew_pred, pred.coord))) if a.data else {}
            rows.append(dict(run=g, seed=seed, m12=m12, m3=m3, **extra, **m))
            dev_all.setdefault(g, []).append(dev)
    df = pd.DataFrame(rows)
    for name, (col, thr) in SUCCESS.items():
        df[name] = df[col] < thr
    os.makedirs(os.path.dirname(a.csv) or ".", exist_ok=True)
    df.to_csv(a.csv, index=False)

    # Summary table
    order = [g for g in names if g in set(df.run)]
    print(f"\nx*: Rg {structure_metrics(ref, ref)[0]['rg']:.2f} A\n")
    fmt = lambda s: f"{s.median():6.2f} [{s.min():5.2f}, {s.max():5.2f}]"
    w = max(22, max(len(g) for g in order) + 1)
    print(f"{'run':<{w}}{'n':>3}  {'RMSD med [min,max]':<22}{'core med [min,max]':<22}{'within 2A':>10}{'Rg':>7}"
          f"{'m12':>11}{'m3':>11}" + ("".join(f"{c:>11}" for c in IMG) if a.data else "") + f"{'rmsd<5':>8}{'core<2':>8}")
    for g in order:
        d = df[df.run == g]
        print(f"{g:<{w}}{len(d):>3}  {fmt(d.rmsd):<22}{fmt(d.core):<22}{d.within2.median():>9.0%}{d.rg.median():>7.1f}"
              f"{d.m12.median():>11.2e}{d.m3.median():>11.2e}" + ("".join(f"{d[c].median():>11.2e}" for c in IMG) if a.data else "")
              + f"{d['rmsd<5'].mean():>8.0%}{d['core<2'].mean():>8.0%}")
    print(f"\nRows written to {a.csv}")

    # Row 0: reward as optimized, accuracy distributions
    for run, g in zip(runs, names):
        for i, f in enumerate(sorted(glob.glob(f"{run}/**/reward_history.json", recursive=True))):
            h = json.load(open(f))
            ax[0, 0].semilogy([e["step"] for e in h], [abs(e["reward"]) for e in h], alpha=0.6, label=g if i == 0 else None, **sty(g))
    ax[0, 0].set(title="|reward| as optimized (map runs: map reward)", xlabel="diffusion step")
    for j, (col, title) in enumerate([("rmsd", "CA RMSD to x* (A)"), ("core", "80%-core RMSD (A)"), ("within2", "fraction within 2 A")]):
        axi = ax[0, 1 + j]
        axi.boxplot([df[df.run == g][col] for g in order], showfliers=False)
        axi.set_xticks(range(1, len(order) + 1), order, rotation=60, fontsize=8)
        for i, g in enumerate(order):
            v = df[df.run == g][col]
            axi.scatter(np.full(len(v), i + 1) + 0.08 * rng.standard_normal(len(v)), v, s=14, color=color[g],
                        marker=STYLE[method(g)][1], zorder=3)
        axi.set(title=title)

    # Row 1: sensitivity (above), per-residue deviation, loss vs accuracy
    for g in order:
        d = df[df.run == g]
        mk = STYLE[method(g)][1]
        ax[1, 1].semilogy(ref.res_id[ref.atom_name == "CA"], np.median(dev_all[g], axis=0), label=g, **{**sty(g), "lw": 0.9})
        ax[1, 2].loglog(d.m3_img if a.data else d.m12, d.core, mk, color=color[g], ms=5, label=g)
        ax[1, 3].loglog(d.m3, d.core, mk, color=color[g], ms=5, label=g)
    for v in (sens_img[:, 1] if a.data else sens[:, 0]):
        ax[1, 2].axvline(v, color="0.85", lw=0.8, zorder=0)
    for v in sens[:, 1]:
        ax[1, 3].axvline(v, color="0.85", lw=0.8, zorder=0)
    ax[1, 1].set(title="median per-residue CA deviation after core fit (A)", xlabel="res_id (x*)")
    ax[1, 2].set(title=f"does {'image-space m3' if a.data else 'm12'} predict accuracy? (grey: x* + 0.25..8 A noise)",
                 xlabel="m3_img" if a.data else "m12", ylabel="80%-core RMSD (A)")
    ax[1, 3].set(title="does m3 predict accuracy? (grey: x* + 0.25..8 A noise)", xlabel="m3", ylabel="80%-core RMSD (A)")

    # Row 2: per-step traces from moment runs (median over seeds)
    for run, g in zip(runs, names):
        tr = [json.load(open(f)) for f in sorted(glob.glob(f"{run}/**/moment_trace.json", recursive=True))]
        if not tr:
            continue
        n = min(len(t) for t in tr)
        for j, key in enumerate(["m12", "m3", "rmsd"]):
            if tr[0][0][key] is None:
                continue
            M = np.median([[t[k][key] for k in range(n)] for t in tr], axis=0)
            ax[2, j].semilogy(range(n), M, label=g, **sty(g)) if key != "rmsd" else ax[2, j].plot(range(n), M, label=g, **sty(g))
        ax[2, 3].semilogy(range(n), [max(tr[0][k]["s"], 1e-3) for k in range(n)], label=g, **sty(g))
    for j, t in enumerate(["median m12 along the trajectory (as optimized)", "median m3 along the trajectory (as optimized)",
                           "median CA RMSD to x* along the trajectory (A)", "DW posterior width s_t (A)"]):
        ax[2, j].set(title=t, xlabel="diffusion step")
    for axi in ax.ravel():
        if axi.get_legend_handles_labels()[0]:
            axi.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(a.out, dpi=120)
    print(f"Saved {a.out}")


if __name__ == "__main__":
    run_checks()
"""Checks for Stage 3a runs.

1. Reward / noise / embedding-shift traces from every reward_history.json under --runs.
2. CA RMSD (proper and mirrored) of every final prediction to x*.
3. Reward of every final prediction, under two weightings:
     global   : the reward used in stage3a_oracle.py (relative Frobenius over all entries)
     whitened : per-degree relative errors of C_l after whitening shells by the radial power of x*,
                averaged over l (gives high-l / high-kappa structure equal say)
4. Sensitivity: the same two scores for x* + N(0, s^2) per-atom noise, s in SIGMAS (Angstrom).
   If the 'global' score of x* + 4 A noise is already ~ the final reward, the reward cannot discriminate.
5. Per-degree whitened error of each prediction (where in l the mismatch lives).

Usage: python stage3a_check.py --ref <x*.pdb> --runs out/stage3a out/stage3a_base --out out/stage3a_check.png
"""

import argparse
import glob
import json
import os

import biotite.structure as bs
import biotite.structure.io as bio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from stage3a_oracle import Stage3aOracleReward

SIGMAS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]


def ca_rmsd(ref, pred):
    ca_r, ca_p = ref[ref.atom_name == "CA"], pred[pred.atom_name == "CA"]
    if len(ca_r) != len(ca_p):
        return np.nan, np.nan
    mir = ca_p.copy()
    mir.coord = -mir.coord
    return bs.rmsd(ca_r, bs.superimpose(ca_r, ca_p)[0]), bs.rmsd(ca_r, bs.superimpose(ca_r, mir)[0])


def radius(a):  # radius of gyration and max atom radius about the centroid (heavy atoms)
    x = a.coord[a.element != "H"]
    r = np.linalg.norm(x - x.mean(0), axis=1)
    return np.sqrt((r**2).mean()), r.max()


def worst_residues(ref, pred, k=5):
    ca_r, ca_p = ref[ref.atom_name == "CA"], pred[pred.atom_name == "CA"]
    if len(ca_r) != len(ca_p):
        return "n/a"
    d = np.linalg.norm(bs.superimpose(ca_r, ca_p)[0].coord - ca_r.coord, axis=1)
    return ", ".join(f"{ca_r.res_id[i]}: {d[i]:.1f}" for i in np.argsort(d)[::-1][:k])


@torch.no_grad()
def scores(rew, x):
    """x: (N_atom, 3) in rew's model-atom order. Returns global score, whitened score, per-l whitened errors."""
    x = torch.as_tensor(x, dtype=torch.float32, device=rew.dev)[rew.keep]
    x = x - x.mean(0)
    a0, C = rew.invariants(x, rew.rho_pred)
    g = -(
        (a0 - rew.a0_ref).abs().pow(2).sum() / rew.a0_ref.abs().pow(2).sum()
        + sum((c - cr).abs().pow(2).sum() for c, cr in zip(C, rew.C_ref)) / sum(cr.abs().pow(2).sum() for cr in rew.C_ref)
    ).item()
    d = 1 / sum(cr.diagonal().real for cr in rew.C_ref).sqrt()  # whitening: 1 / sqrt(radial power of x*)
    D = d[:, None] * d[None, :]
    per_l = np.array([((D * (c - cr)).abs().pow(2).sum() / (D * cr).abs().pow(2).sum()).item() for c, cr in zip(C, rew.C_ref)])
    return g, -per_l.mean(), per_l


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--resolution", type=float, default=5.0)
    ap.add_argument("--out", default="out/stage3a_check.png")
    a = ap.parse_args()
    ref = bio.load_structure(a.ref)
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))

    # 1. traces
    for run in a.runs:
        for f in sorted(glob.glob(f"{run}/**/reward_history.json", recursive=True)):
            h, lab = json.load(open(f)), os.path.relpath(f, run).split(os.sep)[0]
            step = [e["step"] for e in h]
            ax[0, 0].semilogy(step, [-e["reward"] for e in h], label=lab)
            ax[0, 1].semilogy(step, [e["noise_level"] for e in h], label=lab)
            if "cumulative_shift_z" in h[0]:
                ax[0, 2].plot(step, [e["cumulative_shift_s"] for e in h], label=f"{lab} s")
                ax[0, 2].plot(step, [e["cumulative_shift_z"] for e in h], "--", label=f"{lab} z")
    for axi, t in zip(ax[0], ["-reward (global, as optimized)", "noise level t_hat", "cumulative embedding shift"]):
        axi.set(title=t, xlabel="diffusion step")
        axi.legend(fontsize=7)

    # 4. sensitivity of both scores to per-atom noise on x*
    rew_ref = Stage3aOracleReward(a.ref, ref, resolution=a.resolution)
    rng = np.random.default_rng(0)
    sens = np.array([scores(rew_ref, ref.coord + s * rng.standard_normal(ref.coord.shape))[:2] for s in SIGMAS])
    ax[1, 0].loglog(SIGMAS, -sens[:, 0], "o-", label="global")
    ax[1, 0].loglog(SIGMAS, -sens[:, 1], "s-", label="whitened")
    print("\nSensitivity (x* + per-atom noise):")
    for s, (g, w) in zip(SIGMAS, sens):
        print(f"  sigma {s:5.2f} A   global {-g:.3e}   whitened {-w:.3e}")

    # 2, 3, 5. final predictions
    print("\nFinal predictions:")
    for k, f in enumerate(sorted(sum([glob.glob(f"{r}/**/predictions/*.cif", recursive=True) for r in a.runs], []))):
        pred = bio.load_structure(f)
        rp, rm = ca_rmsd(ref, pred)
        g, w, per_l = scores(Stage3aOracleReward(a.ref, pred, resolution=a.resolution), pred.coord)
        lab = os.path.basename(f).replace(".cif", "")
        print(f"  {lab}\n    CA RMSD {rp:.2f} A (mirror {rm:.2f} A)   global {-g:.3e}   whitened {-w:.3e}")
        print(f"    radius: Rg {radius(pred)[0]:.2f} A, max {radius(pred)[1]:.2f} A   (x*: Rg {radius(ref)[0]:.2f}, max {radius(ref)[1]:.2f})")
        print(f"    largest CA deviations (res_id: A): {worst_residues(ref, pred)}")
        for j, (val, mk) in enumerate([(-g, "-"), (-w, ":")]):
            ax[1, 0].axhline(val, color=f"C{k + 2}", ls=mk, lw=1, label=f"{lab[:25]} {'global' if j == 0 else 'whitened'}")
        ax[1, 1].semilogy(per_l, label=f"{lab[:25]} ({rp:.1f} A)")
        ax[1, 2].bar(k, rp, label=lab[:25])
    ax[1, 0].set(title="sensitivity: -score of x* + noise (lines: predictions)", xlabel="per-atom noise sigma (A)")
    ax[1, 1].set(title="whitened relative error of C_l", xlabel="degree l")
    ax[1, 2].set(title="CA RMSD to x* (A)", xticks=[])
    for axi in ax[1]:
        axi.legend(fontsize=6)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(a.out, dpi=130)
    print(f"\nSaved {a.out}")


if __name__ == "__main__":
    main()
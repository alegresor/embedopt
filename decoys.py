"""Decoys of a multi-chain x* with known errors, to measure how strongly a reward responds to structural change.

    python decoys.py 7N6N [--move B C] [--angles 5 15 30 60] [--pred pred.cif]
Output: out/decoys/7N6N_rotB_<angle>.pdb (chains --move rotated rigidly about their centroid, fixed random axis),
        out/decoys/7N6N_monomer.pdb (chains --move removed); prints the C-alpha RMSD of each to x* (no superposition:
        the moments are invariant to global pose, so the relevant change is internal; RMSD after optimal
        superposition is printed too).
"""
import argparse, os
import numpy as np
import biotite.structure as bs
import biotite.structure.io as bio

ap = argparse.ArgumentParser()
ap.add_argument("system")
ap.add_argument("--move", nargs="+", default=["B", "C"])
ap.add_argument("--angles", nargs="+", type=float, default=[5, 15, 30, 60])
ap.add_argument("--pred", default=None, help="also report this prediction's RMSD to x*")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--scales", nargs="*", type=float, default=[], help="x* scaled isotropically about its centroid")
a = ap.parse_args()

x = bio.load_structure(f"examples/synthetic_map_benchmark/pdb_clean/{a.system}.pdb")
x = x[0] if isinstance(x, bs.AtomArrayStack) else x
os.makedirs("out/decoys", exist_ok=True)
mv = np.isin(x.chain_id, a.move)
c = x.coord[mv].mean(0)
u = np.random.default_rng(a.seed).standard_normal(3); u /= np.linalg.norm(u)
K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
ca = x.atom_name == "CA"


def fitted(fixed, mobile):
    """mobile optimally superimposed on fixed (biotite < 1 returns (fitted, T); newer versions return T only)."""
    r = bs.superimpose(fixed, mobile)
    return r[0] if isinstance(r, tuple) else r.apply(mobile)


def rmsd(y, z):
    p, q = y.coord[ca], z.coord[ca]
    return np.sqrt(((p - q) ** 2).sum(1).mean()), bs.rmsd(z[ca], fitted(z[ca], y[ca]))


print(f"{a.system}: moving chains {a.move} ({mv.sum()} atoms) about their centroid, axis {np.round(u, 3)}")
for t in a.angles:
    R = np.eye(3) + np.sin(np.deg2rad(t)) * K + (1 - np.cos(np.deg2rad(t))) * K @ K
    y = x.copy()
    y.coord[mv] = (x.coord[mv] - c) @ R.T + c
    f = f"out/decoys/{a.system}_rot{''.join(a.move)}_{t:g}.pdb"
    bio.save_structure(f, y)
    r0, r1 = rmsd(y, x)
    print(f"  {f}: CA RMSD to x* {r0:.2f} A (after superposition {r1:.2f} A)")
rg = lambda X: float(np.sqrt(((X - X.mean(0)) ** 2).sum(1).mean()))
print(f"  x*: radius of gyration (CA) {rg(x.coord[ca]):.2f} A")
for sc in a.scales:
    y = x.copy()
    cx = x.coord.mean(0)
    y.coord = (x.coord - cx) * sc + cx
    f = f"out/decoys/{a.system}_scale{sc:g}.pdb"
    bio.save_structure(f, y)
    print(f"  {f}: x* scaled by {sc:g} (CA Rg {rg(y.coord[ca]):.2f} A)")
f = f"out/decoys/{a.system}_monomer.pdb"
bio.save_structure(f, x[~mv])
print(f"  {f}: chains {sorted(set(x.chain_id[~mv]))} only")
if a.pred:
    p = bio.load_structure(a.pred)
    p = p[0] if isinstance(p, bs.AtomArrayStack) else p
    p = p[p.element != "H"]
    pc, xc = p[p.atom_name == "CA"], x[ca]
    if len(pc) == len(xc):
        fit = fitted(xc, pc)
        print(f"  prediction {a.pred}: CA RMSD to x* after superposition {bs.rmsd(xc, fit):.2f} A; "
              f"CA Rg {rg(pc.coord):.2f} A vs x* {rg(xc.coord):.2f} A (ratio {rg(pc.coord) / rg(xc.coord):.4f})")
        # best isotropic scale of the prediction onto x* (after superposition): s = <x, y> / <y, y> about centroids
        X0, Y0 = xc.coord - xc.coord.mean(0), fit.coord - fit.coord.mean(0)
        sbest = float((X0 * Y0).sum() / (Y0 * Y0).sum())
        print(f"  prediction rescaled by {sbest:.4f}: CA RMSD {np.sqrt(((X0 - sbest * Y0) ** 2).sum(1).mean()):.2f} A")
        # where does it deviate? per chain (x* chain order = prediction chain order, residue order within chains)
        cid = xc.chain_id
        print("  per chain (each superposed on its own):")
        dev = np.zeros(len(xc))
        for c in dict.fromkeys(cid):
            m = cid == c
            if m.sum() < 3:
                continue
            fc = fitted(xc[m], pc[m])
            d = np.linalg.norm(fc.coord - xc.coord[m], axis=1)
            dev[m] = d
            print(f"    chain {c}: {m.sum()} CA, RMSD {np.sqrt((d ** 2).mean()):.2f} A, "
                  f"90th pct {np.percentile(d, 90):.2f} A, max {d.max():.2f} A")
        chains = list(dict.fromkeys(cid))
        if len(chains) >= 2:
            mA, mB = cid == chains[0], cid == chains[1]
            T = bs.superimpose(xc[mA], pc[mA])
            T = T[1] if isinstance(T, tuple) else T
            pB = T.apply(pc[mB])                                   # prediction in x*'s frame of chain A
            TB = bs.superimpose(xc[mB], pB)
            TB = TB[1] if isinstance(TB, tuple) else TB
            R = np.asarray(TB.rotation).reshape(3, 3)
            ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
            shift = np.linalg.norm(pB.coord.mean(0) - xc.coord[mB].mean(0))
            print(f"  chain {chains[1]} relative to chain {chains[0]} (after superposing {chains[0]}): "
                  f"rigid rotation {ang:.1f} deg, centroid shift {shift:.2f} A")
        res = xc.res_id
        top = np.argsort(dev)[::-1][:15]
        print("  largest per-chain deviations (chain:residue deviation):  "
              + "  ".join(f"{cid[i]}:{res[i]} {dev[i]:.1f}" for i in sorted(top, key=lambda i: (cid[i], res[i]))))
        for c in chains:
            m = cid == c
            if m.sum() < 20:
                continue
            d, r = dev[m], res[m]
            seg = [(int(r[i]), float(d[i])) for i in range(len(d)) if d[i] > 2.0]
            print(f"    chain {c}: {len(seg)} of {m.sum()} residues deviate > 2 A"
                  + (f"; first/last such residue {seg[0][0]}..{seg[-1][0]}" if seg else ""))
    else:
        print(f"  prediction has {len(pc)} CA vs {len(xc)} in x*: lengths differ, RMSD skipped")
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
        print(f"  prediction {a.pred}: CA RMSD to x* after superposition {bs.rmsd(xc, fit):.2f} A")
    else:
        print(f"  prediction has {len(pc)} CA vs {len(xc)} in x*: lengths differ, RMSD skipped")

"""Per-pass summary of the multi-pass chains (experiments/chain.sh), read from out/$SYS/check_chain.csv.

Prints, for each chain, the median full / core RMSD, fraction within 2 A and success rates after each pass, and
paired per-seed comparisons of the final pass: moment3 vs moment2 (third-order effect), moment3 vs map (moments vs
map at equal compute), and each chain's final pass vs its own pass 1.
"""

import os

import pandas as pd

SYS = os.environ["SYS"]
T = ["", "_seq80", "_seq80_seq30", "_seq80_seq30_seq15"]
CHAINS = {
    "map (m: map reward)": ["map_lr0.5" + t for t in T],
    "moment2 (m12)": ["moment2_lr0.1" + t for t in T],
    "moment3 (m12, then m12 m3)": ["moment2_lr0.1"] + ["moment3_lr0.1" + t for t in T[1:]],
}

d = pd.read_csv(f"out/{SYS}/check_chain.csv")
core, full = (d.pivot_table(index="seed", columns="run", values=v) for v in ("core", "rmsd"))
within = d.pivot_table(index="seed", columns="run", values="within2")
print(f"\n{SYS}: medians per pass (core / full RMSD in A; success rates over seeds)")
print(f"{'chain':<28}{'pass':>5}{'core':>8}{'full':>8}{'<2A':>7}{'full<5':>8}{'core<2':>8}")
for name, runs in CHAINS.items():
    for k, r in enumerate(runs, 1):
        if r not in core:
            print(f"{name:<28}{k:>5}   (missing: {r})")
            continue
        c, f, w = core[r], full[r], within[r]
        print(f"{name:<28}{k:>5}{c.median():>8.2f}{f.median():>8.2f}{w.median():>7.0%}{(f < 5).mean():>8.0%}{(c < 2).mean():>8.0%}")

print("\nPaired core RMSD at the final pass (negative = first better):")
fin = {k: v[-1] for k, v in CHAINS.items()}
pairs = [("moment3 vs moment2", fin["moment3 (m12, then m12 m3)"], fin["moment2 (m12)"]),
         ("moment3 vs map", fin["moment3 (m12, then m12 m3)"], fin["map (m: map reward)"]),
         ("moment2 vs map", fin["moment2 (m12)"], fin["map (m: map reward)"])]
pairs += [(f"{k.split()[0]}: final vs pass 1", v[-1], v[0]) for k, v in CHAINS.items()]
for lab, a, b in pairs:
    if a in core and b in core:
        dlt = (core[a] - core[b]).dropna()
        print(f"  {lab:<26} median {dlt.median():+.2f} A, first better in {(dlt < 0).sum()}/{len(dlt)} seeds")

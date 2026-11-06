"""Per-pass summary of the multi-pass chains (experiments/chain.sh, experiments/image_chain.sh).

Environment: SYS (system), CSV (default out/$SYS/check_chain$TAG.csv), TAG (moment-run tag: "" = invariant space,
"_imgx" / "_imgd" = image space with exact / image data), SELECT (selection column, default m3; m3_img for image runs),
TAGS3 (space-separated tags of moment3 chains to show, all sharing pass 1 moment2$TAG; default $TAG; e.g.
"_imgx _imgx_rep _imgx_c3s2q24"; the first is the reference for paired comparisons of the others).

Prints, for each chain, the median full / core RMSD, fraction within 2 A and success rates after each pass; paired
per-seed comparisons of the final pass (moment3 vs moment2, moments vs map, final vs pass 1, and, for TAG != "", the
image-space chains vs the invariant-space ones if present); and selection of the lowest-SELECT seed per run.
"""

import os

import pandas as pd

SYS, TAG = os.environ["SYS"], os.environ.get("TAG", "")
CSV = os.environ.get("CSV", f"out/{SYS}/check_chain{TAG}.csv")
SELECT = os.environ.get("SELECT", "m3")
TAGS3 = os.environ.get("TAGS3", TAG).split() or [TAG]
T = ["", "_seq80", "_seq80_seq30", "_seq80_seq30_seq15"]


def chains(tag, tags3):
    out = {f"moment2{tag} (m12)": [f"moment2{tag}_lr0.1" + t for t in T]}
    for t3 in tags3:
        out[f"moment3{t3} (m12, then m12 m3)"] = [f"moment2{tag}_lr0.1"] + [f"moment3{t3}_lr0.1" + t for t in T[1:]]
    return out


CHAINS = {"map (map reward)": ["map_lr0.5" + t for t in T], **chains(TAG, TAGS3), **(chains("", [""]) if TAG else {})}

d = pd.read_csv(CSV)
core, full, within = (d.pivot_table(index="seed", columns="run", values=v) for v in ("core", "rmsd", "within2"))
w = max(len(k) for k in CHAINS) + 2
print(f"\n{SYS} ({CSV}): medians per pass (core / full RMSD in A; success rates over seeds)")
print(f"{'chain':<{w}}{'pass':>5}{'core':>8}{'full':>8}{'<2A':>7}{'full<5':>8}{'core<2':>8}")
for name, runs in CHAINS.items():
    for k, r in enumerate(runs, 1):
        if r not in core:
            print(f"{name:<{w}}{k:>5}   (missing: {r})")
            continue
        c, f, wi = core[r], full[r], within[r]
        print(f"{name:<{w}}{k:>5}{c.median():>8.2f}{f.median():>8.2f}{wi.median():>7.0%}{(f < 5).mean():>8.0%}{(c < 2).mean():>8.0%}")

print("\nPaired core RMSD at the final pass (negative = first better):")
fin = {k.split()[0]: v[-1] for k, v in CHAINS.items()}
m2 = f"moment2{TAG}"
pairs = [(f"{m2} vs map", fin[m2], fin["map"])] + ([(f"{m2} vs moment2", fin[m2], fin["moment2"])] if TAG else [])
for t3 in TAGS3:
    m3 = f"moment3{t3}"
    pairs += [(f"{m3} vs {m2}", fin[m3], fin[m2]), (f"{m3} vs map", fin[m3], fin["map"])]
    pairs += [(f"{m3} vs moment3", fin[m3], fin["moment3"])] if TAG else []
    pairs += [(f"{m3} vs moment3{TAGS3[0]}", fin[m3], fin[f"moment3{TAGS3[0]}"])] if t3 != TAGS3[0] else []
pairs += [(f"{k.split()[0]}: final vs pass 1", v[-1], v[0]) for k, v in CHAINS.items()]
for lab, a, b in pairs:
    if a in core and b in core:
        dlt = (core[a] - core[b]).dropna()
        print(f"  {lab:<44} median {dlt.median():+.2f} A, first better in {(dlt < 0).sum()}/{len(dlt)} seeds")

if SELECT in d:
    print(f"\nSelection of the lowest-{SELECT} seed per run (core RMSD in A; Spearman of {SELECT} with core within the run):")
    print(f"{'run':<{w + 8}}{'selected':>9}{'best':>7}{'median':>8}{'rho':>7}")
    for r in dict.fromkeys(r for v in CHAINS.values() for r in v):
        x = d[d.run == r]
        if len(x) == 0:
            continue
        print(f"{r:<{w + 8}}{x.loc[x[SELECT].idxmin(), 'core']:>9.2f}{x.core.min():>7.2f}{x.core.median():>8.2f}"
              f"{x[SELECT].corr(x.core, method='spearman'):>7.2f}")
    g = d[d.run != "base"]
    for col in dict.fromkeys(["m12", "m3", SELECT] + [c for c in ("m12_img", "m3_img") if c in d]):
        print(f"  pooled over guided predictions: Spearman({col}, core) = {g[col].corr(g.core, method='spearman'):+.2f}")
else:
    print(f"\n(no column {SELECT} in {CSV}; selection skipped)")

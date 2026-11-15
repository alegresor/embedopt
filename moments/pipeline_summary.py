"""Summary of experiments/full_pipeline.sh over systems. Environment: SYSTEMS, TAG (e.g. snr16).

Per system and arm (final pass): core / full RMSD medians, success rates, and the lowest-m3_img selection; plus the
RELION, docking and data-check diagnostics.
"""

import json
import os

import pandas as pd

SYSTEMS, TAG = os.environ["SYSTEMS"].split(), os.environ["TAG"]
F = "_seq80_seq30_seq15"
ARMS = [
    ("unguided prior", "base", "-"),
    ("map, oracle map + per-step alignment to x*", f"map_lr0.5{F}", "x*"),
    ("moments, oracle invariants", f"moment3_lr0.1{F}", "x*"),
    ("moments, exact image moments", f"moment3_imgx_w12_m3wblock_lr0.1{F}", "x*"),
    (f"moments, {TAG} images", f"moment3_imgd_{TAG}n1e5_w12_m3wblock_lr0.1{F}", "none"),
    ("moments -> known-pose RELION map", "map_relpose_m3_lr0.5_from_moments", "poses"),
    ("moments -> ab-initio RELION map", "map_relfull_m3_lr0.5_from_moments", "none"),
]

for sys_ in SYSTEMS:
    sim = f"out/{sys_}/noise_{TAG}"
    print(f"\n=== {sys_} ({TAG})")
    for f, keys in ((f"{sim}/relion/fsc_poses_vs_truth.json", ("fsc05", "cc")),
                    (f"{sim}/dock_poses_m/dock.json", ("hand",)), (f"{sim}/dock_full_m/dock.json", ("hand",))):
        if os.path.isfile(f):
            j = json.load(open(f))
            extra = "  ".join(f"{h} {j[h]['score']:.3f}" for h in ("original", "mirrored") if h in j)
            print(f"  {f.split('/')[-2]}/{f.split('/')[-1]}: " + "  ".join(f"{k} {j[k]}" for k in keys if k in j)
                  + (f"  ({extra})" if extra else ""))
    pp = f"{sim}/relion/PostProcess/postprocess.star"
    if os.path.isfile(pp):
        res = [ln.split()[-1] for ln in open(pp) if "_rlnFinalResolution" in ln]
        print(f"  RELION ab initio + auto-refine: {res[0] if res else '?'} A")
    csv = f"out/{sys_}/check_pipeline_{TAG}.csv"
    if not os.path.isfile(csv):
        print(f"  (no {csv})")
        continue
    d = pd.read_csv(csv)
    print(f"  {'arm':<46}{'oracle':>8}{'n':>4}{'core':>8}{'full':>8}{'core<2':>8}{'full<5':>8}{'sel.':>7}")
    for name, run, orc in ARMS:
        x = d[d.run == run]
        if not len(x):
            print(f"  {name:<46}{orc:>8}   (missing {run})")
            continue
        sel = x.loc[x.m3_img.idxmin(), "core"] if "m3_img" in x else float("nan")
        print(f"  {name:<46}{orc:>8}{len(x):>4}{x.core.median():>8.2f}{x.rmsd.median():>8.2f}"
              f"{(x.core < 2).mean():>8.0%}{(x.rmsd < 5).mean():>8.0%}{sel:>7.2f}")

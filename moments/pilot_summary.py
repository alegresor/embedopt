"""Summary of experiments/noise_pilot.sh. Environment: SYS, TAG (e.g. snr16).

Prints the data checks (moments vs the model, noise covariance), the RELION and docking diagnostics, and for every arm
the per-pass medians (core / full RMSD, success rates), paired final-pass comparisons, and lowest-m3_img selection.
"""

import json
import os

import pandas as pd

SYS, TAG = os.environ["SYS"], os.environ["TAG"]
SIM = f"out/{SYS}/noise_{TAG}"
CSV = os.environ.get("CSV", f"out/{SYS}/check_pilot_{TAG}.csv")
T = ["", "_seq80", "_seq80_seq30", "_seq80_seq30_seq15"]


def chain(p1, p234=None):
    return [p1] + [(p234 or p1) + t for t in T[1:]]


ARMS = {
    "map, oracle map + oracle frame": chain("map_lr0.5"),
    "map, RELION known poses, docked frame": chain("map_relpose_lr0.5"),
    "map, RELION ab initio, docked frame": chain("map_relfull_lr0.5"),
    "moments, exact (imgx)": chain("moment2_imgx_w12_lr0.1", "moment3_imgx_w12_m3wblock_lr0.1"),
    f"moments, {TAG} N=1e5": chain(f"moment2_imgd_{TAG}n1e5_w12_lr0.1", f"moment3_imgd_{TAG}n1e5_w12_m3wblock_lr0.1"),
    f"moments, {TAG} N=1e6": chain(f"moment2_imgd_{TAG}n1e6_w12_lr0.1", f"moment3_imgd_{TAG}n1e6_w12_m3wblock_lr0.1"),
}


def show_json(path, keys=None):
    if os.path.isfile(path):
        d = json.load(open(path))
        d = {k: v for k, v in d.items() if keys is None or k in keys}
        print(f"  {os.path.basename(os.path.dirname(path))}/{os.path.basename(path)}: {json.dumps(d, default=str)[:600]}")


print(f"\n=== {SYS}, {TAG}: diagnostics")
show_json(f"{SIM}/sim.json", ["n", "L", "pixel", "snr", "noise_var", "defoci"])
show_json(f"{SIM}/relion/fsc_poses_vs_truth.json", ["fsc05", "fsc0143", "cc"])
for d in ("dock_selftest", "dock_poses", "dock_full"):
    show_json(f"{SIM}/{d}/dock.json", ["hand", "rmsd_to_truth_frame", "clean_threshold"])
    if os.path.isfile(f"{SIM}/{d}/dock.json"):
        j = json.load(open(f"{SIM}/{d}/dock.json"))
        print("      scores: " + ", ".join(f"{h} {j[h]['score']:.3f}" for h in ("original", "mirrored") if h in j))
pp = f"{SIM}/relion/PostProcess/postprocess.star"
if os.path.isfile(pp):
    for line in open(pp):
        if "_rlnFinalResolution" in line:
            print(f"  RELION ab-initio + auto-refine: FSC 0.143 resolution {line.split()[-1]} A")

d = pd.read_csv(CSV)
core, full = (d.pivot_table(index="seed", columns="run", values=v) for v in ("core", "rmsd"))
w = max(map(len, ARMS)) + 2
print(f"\nper-pass medians ({CSV})")
print(f"{'arm':<{w}}{'pass':>5}{'core':>8}{'full':>8}{'full<5':>8}{'core<2':>8}")
for name, runs in ARMS.items():
    for k, r in enumerate(runs, 1):
        if r in core:
            c, f = core[r], full[r]
            print(f"{name:<{w}}{k:>5}{c.median():>8.2f}{f.median():>8.2f}{(f < 5).mean():>8.0%}{(c < 2).mean():>8.0%}")
        else:
            print(f"{name:<{w}}{k:>5}   (missing {r})")
fin = {n: r[-1] for n, r in ARMS.items()}
refs = [list(ARMS)[0], "moments, exact (imgx)"]
print("\npaired final-pass core RMSD (negative = arm better)")
for n, r in fin.items():
    for ref in refs:
        if n != ref and r in core and fin[ref] in core:
            x = (core[r] - core[fin[ref]]).dropna()
            print(f"  {n:<{w}} vs {ref:<34} median {x.median():+.2f} A, better in {(x < 0).sum()}/{len(x)}")
if "m3_img" in d:
    print("\nselection of the lowest-m3_img seed (data: N=1e5 moments), final pass: selected / best / median core RMSD")
    for n, r in fin.items():
        x = d[d.run == r]
        if len(x):
            print(f"  {n:<{w}} {x.loc[x.m3_img.idxmin(), 'core']:6.2f} {x.core.min():6.2f} {x.core.median():6.2f}")

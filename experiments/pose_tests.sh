#!/bin/bash
# Learned viewing density (8K9Z_A, SNR 1/16, N = 1e5 particles with non-uniform viewing directions from
# moments.poses.VMFMixture.default(), CTF and noise: out/8K9Z_A/noise_snr16/moments_N100000_poses.npz).
# 10 seeds, full chains (moment2 w12, then 3 restarts moment3 w12 wblock with the m12 cap eps = 0.1):
#   oracle  : the true density (POSE=oracle)
#   fit     : nu fitted by the exact QP every 10 reward calls (P = 4, 14 parameters), in all passes
#   frozen  : pass 1 of `fit`; theta taken from its seed with the lowest final m12, then fixed for passes 2-4
#   uniform : uniform poses assumed (the current model)
# About 7-9 h (the moments under a density cost more per step than the uniform ones):
#   setsid nohup bash experiments/pose_tests.sh > out/pose_tests.log 2>&1 < /dev/null &
set -uo pipefail
SYS=8K9Z_A; SEEDS=101,102,103,104,105,106,107,108,109,110; NS=10; F=_seq80_seq30_seq15; EPS=1e-1
D=out/$SYS/noise_snr16/moments_N100000_poses.npz
log() { echo "=== $(date '+%F %T') $*"; }
count() { (ls out/$SYS/$1/*/seed_*/predictions/*_sample_0.cif 2>/dev/null || true) | wc -l; }
arm() {
    local run=$1 k; shift
    [ "$(count $run)" -ge "$NS" ] && { echo "skip $run"; return 0; }
    for k in 1 2 3; do
        rm -rf out/$SYS/$run
        make run SYSTEM=$SYS SEEDS=$SEEDS "$@" > out/$SYS/$run.log 2>&1
        [ "$(count $run)" -ge "$NS" ] && return 0
        log "$run: $(count $run)/$NS (attempt $k)"; sleep 30
    done
    return 1
}
COMMON="METHOD=moment LR=0.1 SPACE=image DATA=$D M12=white"
restarts() {  # restarts PASS1 DTAG make-args...
    local prev=$1 DT=$2 t="" M3=moment3_imgd${2}_w12_m3wblock_lr0.1; shift 2
    for S in 80 30 15; do
        t="${t}_seq$S"
        arm $M3$t $COMMON ORDER=3 M3=wblock M12_EPS=$EPS DTAG=$DT SIGMA0=$S TAG=$t \
            INIT="out/$SYS/$prev/*/seed_{seed}/predictions/*_sample_0.cif" "$@" || return 1
        prev=$M3$t
    done
}
chain() {  # chain DTAG make-args...
    local DT=$1 M2=moment2_imgd${1}_w12_lr0.1; shift
    arm $M2 $COMMON ORDER=2 DTAG=$DT "$@" || return 1
    restarts $M2 $DT "$@"
}
[ -f $D ] || { echo "missing $D (python -m moments.poses stream ...)"; exit 1; }
log "oracle"; chain _posesoracle POSE=oracle || log "oracle FAILED"
log "fit";    chain _posesfit POSE=fit POSE_P=4 POSE_EVERY=10 || log "fit FAILED"
log "frozen: theta from pass 1 of fit"
TH=out/$SYS/poses_theta_pass1.json
python - <<'PY'
import glob, json
best = None
for f in glob.glob("out/8K9Z_A/moment2_imgd_posesfit_w12_lr0.1/*/seed_*/**/moment_trace.json", recursive=True):
    tr = [r for r in json.load(open(f)) if r.get("pose_theta") is not None]
    if tr and (best is None or tr[-1]["m12"] < best[0]):
        best = (tr[-1]["m12"], tr[-1]["pose_theta"], f)
if best is None:
    raise SystemExit("no fitted theta found in the pass-1 traces")
json.dump(best[1], open("out/8K9Z_A/poses_theta_pass1.json", "w"))
print(f"  theta from {best[2]} (final m12 {best[0]:.3e})")
PY
[ -f $TH ] && { restarts moment2_imgd_posesfit_w12_lr0.1 _posesfrozen POSE=fixed POSE_P=4 POSE_THETA=$TH || log "frozen FAILED"; }
log "uniform"; chain _posesuniform POSE=uniform || log "uniform FAILED"

log "check"
RUNS="moment3_imgd_snr16n1e5eps1e-1_w12_m3wblock_lr0.1$F"
for a in oracle fit frozen uniform; do RUNS="$RUNS moment3_imgd_poses${a}_w12_m3wblock_lr0.1$F"; done
make check SYSTEM=$SYS FIG=check_poses CHECK_DATA=out/$SYS/noise_snr16/moments_N100000.npz RUNS="$RUNS" \
    > out/$SYS/check_poses.log 2>&1
python - <<'PY' 2>&1 | tee out/$SYS/poses_summary.log
import glob, json, sys, numpy as np, pandas as pd
sys.path.insert(0, ".")
from moments.pose_moments import SHDensity
from moments.poses import VMFMixture
F = "_seq80_seq30_seq15"; d = pd.read_csv("out/8K9Z_A/check_poses.csv")
run = lambda a: f"moment3_imgd_poses{a}_w12_m3wblock_lr0.1{F}"
arms = [("uniform poses, uniform model (reference)", f"moment3_imgd_snr16n1e5eps1e-1_w12_m3wblock_lr0.1{F}"),
        ("non-uniform, true density (oracle)", run("oracle")), ("non-uniform, uniform assumed", run("uniform")),
        ("non-uniform, density fitted every 10 calls", run("fit")), ("non-uniform, density frozen after pass 1", run("frozen"))]
core = d.pivot_table(index="seed", columns="run", values="core")
o = run("oracle")
print(f"8K9Z_A, final pass, 10 seeds, eps = 0.1\n{'arm':<44}{'median':>8}{'full':>7}{'core<2':>8}{'vs oracle, paired':>22}   per seed")
for lab, r in arms:
    if r not in core: print(f"{lab:<44}   (missing)"); continue
    x = d[d.run == r]; c = core[r]
    p = (c - core[o]).dropna() if (o in core and r != o) else None
    ps = f"{p.median():+.2f} A, better {int((p < 0).sum())}/{len(p)}" if p is not None else ""
    print(f"{lab:<44}{c.median():>8.2f}{x.rmsd.median():>7.2f}{(c < 2).mean():>8.0%}{ps:>22}   "
          + " ".join(f"{v:.2f}" for v in c.values))
dens = SHDensity(4); true = dens.project(VMFMixture.default())
for a in ("fit", "frozen"):
    if a == "frozen":
        th = [np.asarray(json.load(open("out/8K9Z_A/poses_theta_pass1.json")))]
    else:
        th = []
        for f in glob.glob(f"out/8K9Z_A/{run(a)}/*/seed_*/**/moment_trace.json", recursive=True):
            tr = [r for r in json.load(open(f)) if r.get("pose_theta") is not None]
            if tr: th.append(np.asarray(tr[-1]["pose_theta"]))
    if th:
        e = [np.linalg.norm(t - true) / np.linalg.norm(true) for t in th]
        print(f"density {a}: |theta - theta_true| / |theta_true| (degree-4 projection of the test density): "
              f"median {np.median(e):.2f} over {len(e)} seed(s)")
PY
log "all done"

#!/bin/bash
# In-plane shifts (8K9Z_A, SNR 1/16, N = 1e5, the same particles as the baseline; shifts of 1 A std per axis):
#   rc03   : images recentred with the true shifts + N(0, 0.3^2) A error (as after alignment / 2D classification)
#   envfit : uncorrected images, Gaussian shift envelope on the model moments, sigma fitted at every evaluation
#   env1   : the same with the true sigma (1 A), as a control
# Passes 2-4 (restarts at 80/30/15 A, moment3 w12 wblock) on 5 seeds, all from the existing pass 1 of the uncorrected
# shifted data (moment2_imgd_snr16n1e5sh1_w12_lr0.1). Compared with the uncorrected chain (sh1) and the unshifted
# baseline, seed by seed. About 1.5-2 h. Run from the repo root:
#   setsid nohup bash experiments/shift_tests.sh > out/shift_tests.log 2>&1 < /dev/null &
set -uo pipefail
SYS=8K9Z_A; SEEDS=101,102,103,104,105; NS=5; SIM=out/$SYS/noise_snr16; F=_seq80_seq30_seq15
P1=moment2_imgd_snr16n1e5sh1_w12_lr0.1
log() { echo "=== $(date '+%F %T') $*"; }
count() { (ls out/$SYS/$1/*/seed_*/predictions/*_sample_0.cif 2>/dev/null || true) | wc -l; }
arm() {
    local run=$1 k; shift
    [ "$(count $run)" -ge "$NS" ] && { echo "skip $run"; return 0; }
    for k in 1 2 3; do
        rm -rf out/$SYS/$run
        make run SYSTEM=$SYS SEEDS=$SEEDS "$@" > out/$SYS/$run.log 2>&1
        [ "$(count $run)" -ge "$NS" ] && return 0
        log "$run: $(count $run)/$NS (attempt $k)"; sleep 20
    done
    return 1
}
restarts() {  # restarts DATA DTAG extra-make-args...
    local D=$1 DT=$2 prev=$P1 t="" M3; shift 2
    M3=moment3_imgd${DT}_w12_m3wblock_lr0.1
    for S in 80 30 15; do
        t="${t}_seq$S"
        arm $M3$t METHOD=moment ORDER=3 LR=0.1 SPACE=image DATA=$D DTAG=$DT M12=white M3=wblock SIGMA0=$S TAG=$t \
            INIT="out/$SYS/$prev/*/seed_{seed}/predictions/*_sample_0.cif" "$@" || return 1
        prev=$M3$t
    done
}
[ "$(count $P1)" -ge "$NS" ] || { echo "missing pass 1 $P1 (run experiments/nuisance_tests.sh)"; exit 1; }
log "data"
D0=$SIM/moments_N100000.npz; D1=$SIM/moments_N100000_stream_sh1.npz; D2=$SIM/moments_N100000_stream_sh1rc03.npz
[ -f $D2 ] || python -m moments.image_data stream --sim $SIM --n 100000 --seed 2 --shift 1.0 --recenter 0.3 \
    --suffix _sh1rc03 > $SIM/moments_sh1rc03.log 2>&1
log "floors"
{ python -m moments.image_data floors --sim $SIM --files $D0 $D1 $D2 2>&1 | grep "vs data"
  python -m moments.image_data floors --sim $SIM --shift-sigma 1.0 --files $D1 2>&1 | grep "vs data"
  python -m moments.image_data floors --sim $SIM --shift-sigma fit --files $D1 $D0 2>&1 | grep "vs data"; } \
    | tee out/$SYS/shift_floors.log
log "rc03"; restarts $D2 _snr16n1e5sh1rc03 || log "rc03 FAILED"
log "envfit"; restarts $D1 _snr16n1e5sh1envfit SHIFT_SIGMA=fit || log "envfit FAILED"
log "env1"; restarts $D1 _snr16n1e5sh1env1 SHIFT_SIGMA=1.0 || log "env1 FAILED"
log "check"
RUNS="moment3_imgd_snr16n1e5_w12_m3wblock_lr0.1$F moment3_imgd_snr16n1e5sh1_w12_m3wblock_lr0.1$F"
for DT in sh1rc03 sh1envfit sh1env1; do RUNS="$RUNS moment3_imgd_snr16n1e5${DT}_w12_m3wblock_lr0.1$F"; done
make check SYSTEM=$SYS FIG=check_shift CHECK_DATA=$D0 RUNS="$RUNS" > out/$SYS/check_shift.log 2>&1
python - <<'PY' 2>&1 | tee out/$SYS/shift_summary.log
import glob, json, numpy as np, pandas as pd
d = pd.read_csv("out/8K9Z_A/check_shift.csv"); F = "_seq80_seq30_seq15"; seeds = [f"seed_{s}" for s in range(101, 106)]
d = d[d.seed.isin(seeds)]
run = lambda t: f"moment3_imgd_snr16n1e5{t}_w12_m3wblock_lr0.1{F}"
arms = [("unshifted baseline", run("")), ("1 A shifts, uncorrected", run("sh1")), ("recentred, 0.3 A residual", run("sh1rc03")),
        ("envelope, sigma fitted", run("sh1envfit")), ("envelope, sigma = 1 A (true)", run("sh1env1"))]
core = d.pivot_table(index="seed", columns="run", values="core")
print("8K9Z_A, final pass, seeds 101-105 (core RMSD, A)")
print(f"{'arm':<32}{'median':>8}{'mean':>7}{'core<2':>8}   per seed")
for lab, r in arms:
    if r not in core: print(f"{lab:<32}   (missing)"); continue
    c = core[r]
    print(f"{lab:<32}{c.median():>8.2f}{c.mean():>7.2f}{(c < 2).mean():>8.0%}   " + " ".join(f"{v:.2f}" for v in c.values))
tr = sorted(glob.glob(f"out/8K9Z_A/{run('sh1envfit')}/*/seed_*/**/moment_trace.json", recursive=True))
if tr:
    s = [json.load(open(f))[-1].get("shift_sigma") for f in tr]
    print("fitted sigma at the last step, per seed:", [round(v, 3) for v in s if v is not None], "(true 1.0 A)")
PY
log "all done"

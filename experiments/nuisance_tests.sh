#!/bin/bash
# Moment arm under realistic nuisances (8K9Z_A, SNR 1/16 unless noted, N = 1e5 streamed particles, the same seed as
# the existing baseline chain moment3_imgd_snr16n1e5_*, so all variants see the same poses, CTF groups and noise):
#   de200 : estimator defocus perturbed by N(0, 200^2) A per defocus group (images unchanged)
#   ne    : noise variance estimated from the image corners (images unchanged)
#   sh1   : Gaussian in-plane shifts of 1 A std, uncorrected
#   snr64 : SNR 1/64, N = 1e5, then N = 1e6
# Per variant: moments, floors (x* vs data), moment chain (moment2 w12 + 3 restarts moment3 w12 wblock), then one
# check and a summary against the baseline. Finished steps are skipped; restart with the same command.
#   setsid nohup bash experiments/nuisance_tests.sh > out/nuisance_tests.log 2>&1 < /dev/null &
set -uo pipefail
SYS=8K9Z_A; SEEDS=101,102,103,104,105,106,107,108,109,110; NS=10; RETRIES=3
SIM=out/$SYS/noise_snr16; SIM64=out/$SYS/noise_snr64
F=_seq80_seq30_seq15; LEVELS="80 30 15"
log() { echo "=== $(date '+%F %T') $*"; }
count() { (ls out/$SYS/$1/*/seed_*/predictions/*_sample_0.cif 2>/dev/null || true) | wc -l; }
arm() {
    local run=$1 k; shift
    [ "$(count $run)" -eq "$NS" ] && { echo "skip $run (complete)"; return 0; }
    for k in $(seq 1 $RETRIES); do
        rm -rf out/$SYS/$run
        make run SYSTEM=$SYS SEEDS=$SEEDS "$@" > out/$SYS/$run.log 2>&1
        [ "$(count $run)" -eq "$NS" ] && return 0
        log "$run: $(count $run)/$NS predictions (attempt $k/$RETRIES)"; sleep 30
    done
    return 1
}
chain() {  # chain DATA DTAG
    local D=$1 DT=$2 M2=moment2_imgd${2}_w12_lr0.1 M3=moment3_imgd${2}_w12_m3wblock_lr0.1 prev t=""
    arm $M2 METHOD=moment ORDER=2 LR=0.1 SPACE=image DATA=$D DTAG=$DT M12=white || return 1
    prev=$M2
    for S in $LEVELS; do
        t="${t}_seq$S"
        arm $M3$t METHOD=moment ORDER=3 LR=0.1 SPACE=image DATA=$D DTAG=$DT M12=white M3=wblock SIGMA0=$S TAG=$t \
            INIT="out/$SYS/$prev/*/seed_{seed}/predictions/*_sample_0.cif" || return 1
        prev=$M3$t
    done
}
variant() {  # variant NAME SIMDIR DTAG N STREAM-OPTIONS...
    local name=$1 sim=$2 DT=$3 n=$4; shift 4
    local D=$sim/moments_N${n}_stream_$name.npz
    log "$name: moments"
    [ -f $D ] || python -m moments.image_data stream --sim $sim --n $n --seed 2 --suffix _$name "$@" \
        > $sim/moments_$name.log 2>&1 || { log "$name: moments FAILED"; return 1; }
    grep -E "estimator defocus|estimated from the corners" $sim/moments_$name.log
    python -m moments.image_data floors --sim $sim --files $D 2>&1 | grep "x\* vs data" | tee -a out/$SYS/nuisance_floors.log
    log "$name: chain"
    chain $D $DT || { log "$name: chain FAILED"; return 1; }
}

log "nuisance tests on $SYS"
python -m moments.image_data floors --sim $SIM --files $SIM/moments_N100000.npz 2>&1 | grep "x\* vs data" \
    | sed 's/^/baseline: /' | tee out/$SYS/nuisance_floors.log
variant de200 $SIM _snr16n1e5de200 100000 --defocus-error 200
variant ne    $SIM _snr16n1e5ne    100000 --noise-est
variant sh1   $SIM _snr16n1e5sh1   100000 --shift 1.0
[ -f $SIM64/sim.json ] || python -m moments.simulate --system $SYS --snr 0.015625 --n 0 --out $SIM64 > $SIM64.log 2>&1
variant snr64 $SIM64 _snr64n1e5    100000
variant snr64 $SIM64 _snr64n1e6    1000000

log "check"
RUNS="base moment3_imgx_w12_m3wblock_lr0.1$F moment3_imgd_snr16n1e5_w12_m3wblock_lr0.1$F"
for DT in _snr16n1e5de200 _snr16n1e5ne _snr16n1e5sh1 _snr64n1e5 _snr64n1e6; do RUNS="$RUNS moment3_imgd${DT}_w12_m3wblock_lr0.1$F"; done
make check SYSTEM=$SYS FIG=check_nuisance CHECK_DATA=$SIM/moments_N100000.npz RUNS="$RUNS" > out/$SYS/check_nuisance.log 2>&1
python - <<'PY' 2>&1 | tee out/$SYS/nuisance_summary.log
import pandas as pd
d = pd.read_csv("out/8K9Z_A/check_nuisance.csv"); F = "_seq80_seq30_seq15"
base = f"moment3_imgd_snr16n1e5_w12_m3wblock_lr0.1{F}"
arms = [("exact image moments", f"moment3_imgx_w12_m3wblock_lr0.1{F}"), ("baseline: SNR 1/16, N=1e5", base)] + \
       [(lab, f"moment3_imgd_{t}_w12_m3wblock_lr0.1{F}") for lab, t in
        (("defocus error 200 A", "snr16n1e5de200"), ("estimated noise variance", "snr16n1e5ne"),
         ("1 A shifts, uncorrected", "snr16n1e5sh1"), ("SNR 1/64, N=1e5", "snr64n1e5"), ("SNR 1/64, N=1e6", "snr64n1e6"))]
core = d.pivot_table(index="seed", columns="run", values="core")
print(f"{'variant (final pass, 8K9Z_A)':<30}{'core':>7}{'full':>7}{'core<2':>8}{'paired vs baseline':>22}")
for lab, r in arms:
    x = d[d.run == r]
    if not len(x): print(f"{lab:<30}   (missing)"); continue
    p = (core[r] - core[base]).dropna() if base in core and r != base else None
    ps = f"{p.median():+.2f} A, better {int((p < 0).sum())}/{len(p)}" if p is not None else ""
    print(f"{lab:<30}{x.core.median():>7.2f}{x.rmsd.median():>7.2f}{(x.core < 2).mean():>8.0%}{ps:>22}")
PY
cat out/$SYS/nuisance_floors.log
log "all done"

#!/bin/bash
# Shift sigma estimated from the data, then fixed (8K9Z_A, SNR 1/16, N = 1e5, 1 A shifts; 10 seeds; full chains):
#   envest : sigma from per-particle shift estimates (true shifts + 0.3 A error per axis, as from picking or 2D
#            classification): the pooled std of the estimates; envelope with that sigma in all four passes
#   envref : pass 1 uncorrected (the existing sh1 pass 1); sigma fitted once from the lowest-m3 structure of pass 1
#            (moments.image_data fitsigma), pass 2 with it; refitted from the lowest-m3 structure of pass 2; passes
#            3-4 with that. Only the data and the current predictions are used.
#   rcfull : images recentred with the estimated shifts (0.3 A residual), all four passes
# Compared seed by seed with the uncorrected chain (sh1) and the unshifted baseline. About 4.5-5 h:
#   setsid nohup bash experiments/shift_tests2.sh > out/shift_tests2.log 2>&1 < /dev/null &
set -uo pipefail
SYS=8K9Z_A; SEEDS=101,102,103,104,105,106,107,108,109,110; NS=10; SIM=out/$SYS/noise_snr16; F=_seq80_seq30_seq15
D0=$SIM/moments_N100000.npz; D1=$SIM/moments_N100000_stream_sh1.npz; D2=$SIM/moments_N100000_stream_sh1rc03.npz
P1SH=moment2_imgd_snr16n1e5sh1_w12_lr0.1
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
p3() { echo "moment3_imgd${1}_w12_m3wblock_lr0.1$2"; }   # p3 DTAG TAG
best() {  # best RUN DATA -> structure file of the lowest-m3_img seed (data only)
    local run=$1 D=$2 seed
    [ -f out/$SYS/sel_$run.csv ] || make check SYSTEM=$SYS FIG=sel_$run CHECK_DATA=$D RUNS="$run" > out/$SYS/sel_$run.log 2>&1
    seed=$(python -c "import pandas as pd; d=pd.read_csv('out/$SYS/sel_$run.csv'); print(d.loc[d.m3_img.idxmin(),'seed'])")
    ls out/$SYS/$run/*/$seed/predictions/*_sample_0.cif
}
init() { echo "INIT=out/$SYS/$1/*/seed_{seed}/predictions/*_sample_0.cif"; }
[ -f $D1 ] && [ -f $D2 ] || { echo "missing $D1 or $D2 (run experiments/nuisance_tests.sh and shift_tests.sh)"; exit 1; }
[ "$(count $P1SH)" -ge "$NS" ] || { echo "missing $P1SH"; exit 1; }
COMMON="METHOD=moment LR=0.1 SPACE=image M12=white"

log "envest: sigma from shift estimates"
SE=$(python -m moments.image_data shiftsigma --sim $SIM --n 100000 --seed 2 --shift 1.0 --est-error 0.3 2>/dev/null | tail -1)
echo "  sigma (shift estimates) = $SE A" | tee out/$SYS/shift_sigmas.log
DT=_snr16n1e5sh1envest; M2=moment2_imgd${DT}_w12_lr0.1
arm $M2 $COMMON ORDER=2 DATA=$D1 DTAG=$DT SHIFT_SIGMA=$SE && {
    prev=$M2; t=""
    for S in 80 30 15; do t="${t}_seq$S"; arm $(p3 $DT $t) $COMMON ORDER=3 M3=wblock DATA=$D1 DTAG=$DT SHIFT_SIGMA=$SE \
        SIGMA0=$S TAG=$t "$(init $prev)" || break; prev=$(p3 $DT $t); done; } || log "envest FAILED"

log "envref: sigma refitted from the current best structure"
DT=_snr16n1e5sh1envref
B1=$(best $P1SH $D1); S1=$(python -m moments.image_data fitsigma --sim $SIM --files $D1 --model $B1 2>/dev/null | tail -1)
echo "  envref: sigma from pass-1 best ($B1) = $S1 A" | tee -a out/$SYS/shift_sigmas.log
arm $(p3 $DT _seq80) $COMMON ORDER=3 M3=wblock DATA=$D1 DTAG=$DT SHIFT_SIGMA=$S1 SIGMA0=80 TAG=_seq80 "$(init $P1SH)" && {
    B2=$(best $(p3 $DT _seq80) $D1); S2=$(python -m moments.image_data fitsigma --sim $SIM --files $D1 --model $B2 2>/dev/null | tail -1)
    echo "  envref: sigma from pass-2 best ($B2) = $S2 A" | tee -a out/$SYS/shift_sigmas.log
    prev=$(p3 $DT _seq80); t=_seq80
    for S in 30 15; do t="${t}_seq$S"; arm $(p3 $DT $t) $COMMON ORDER=3 M3=wblock DATA=$D1 DTAG=$DT SHIFT_SIGMA=$S2 \
        SIGMA0=$S TAG=$t "$(init $prev)" || break; prev=$(p3 $DT $t); done; } || log "envref FAILED"

log "rcfull: recentred data, full chain"
DT=_snr16n1e5sh1rcfull; M2=moment2_imgd${DT}_w12_lr0.1
arm $M2 $COMMON ORDER=2 DATA=$D2 DTAG=$DT && {
    prev=$M2; t=""
    for S in 80 30 15; do t="${t}_seq$S"; arm $(p3 $DT $t) $COMMON ORDER=3 M3=wblock DATA=$D2 DTAG=$DT \
        SIGMA0=$S TAG=$t "$(init $prev)" || break; prev=$(p3 $DT $t); done; } || log "rcfull FAILED"

log "check"
RUNS=""
for DT in "" sh1 sh1envest sh1envref sh1rcfull; do RUNS="$RUNS $(p3 _snr16n1e5$DT $F)"; done
make check SYSTEM=$SYS FIG=check_shift2 CHECK_DATA=$D0 RUNS="$RUNS" > out/$SYS/check_shift2.log 2>&1
python - <<'PY' 2>&1 | tee out/$SYS/shift2_summary.log
import pandas as pd
d = pd.read_csv("out/8K9Z_A/check_shift2.csv"); F = "_seq80_seq30_seq15"
run = lambda t: f"moment3_imgd_snr16n1e5{t}_w12_m3wblock_lr0.1{F}"
arms = [("unshifted baseline", run("")), ("1 A shifts, uncorrected", run("sh1")),
        ("envelope, sigma from shift estimates", run("sh1envest")), ("envelope, sigma refitted per pass", run("sh1envref")),
        ("recentred (0.3 A residual), full chain", run("sh1rcfull"))]
core = d.pivot_table(index="seed", columns="run", values="core"); u = run("sh1")
print(f"8K9Z_A, final pass, 10 seeds\n{'arm':<40}{'median':>8}{'core<2':>8}{'vs uncorrected':>22}{'sel.':>7}")
for lab, r in arms:
    x = d[d.run == r]
    if not len(x): print(f"{lab:<40}   (missing)"); continue
    p = (core[r] - core[u]).dropna() if r != u else None
    ps = f"{p.median():+.2f} A, {int((p < 0).sum())}/{len(p)}" if p is not None else ""
    print(f"{lab:<40}{x.core.median():>8.2f}{(x.core < 2).mean():>8.0%}{ps:>22}{x.loc[x.m3_img.idxmin(), 'core']:>7.2f}")
PY
cat out/$SYS/shift_sigmas.log
log "all done"

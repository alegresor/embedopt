#!/bin/bash
# Colored noise and astigmatic CTFs (8K9Z_A, SNR 1/16, N = 1e5 streamed particles, the same poses and defocus groups as
# the baseline chain moment3_imgd_snr16n1e5_*). Three data passes, each writing several estimates (moments.image_data
# stream2d), floors of all estimates, then moment chains (moment2 w12 + 3 restarts moment3 w12 wblock, 10 seeds):
#   colest   : colored noise, spectrum estimated from the image corners             (main)
#   ast2d    : 800 A astigmatism, 2D least-squares estimator, white noise            (main)
#   both     : both together, 2D estimator, estimated spectrum                      (main)
#   colwhite : colored noise, white-noise assumption (corner variance only)          (control)
#   astrad   : 800 A astigmatism ignored (radial CTF at the mean defocus)            (control)
# Main arms first; a summary is written after the main arms and again at the end. Finished steps are skipped.
#   setsid nohup bash experiments/ctf_noise_tests.sh > out/ctf_noise_tests.log 2>&1 < /dev/null &
set -uo pipefail
SYS=8K9Z_A; SEEDS=101,102,103,104,105,106,107,108,109,110; NS=10; RETRIES=3
SIM=out/$SYS/noise_snr16; N=100000; F=_seq80_seq30_seq15
log() { echo "=== $(date '+%F %T') $*"; }
count() { (ls out/$SYS/$1/*/seed_*/predictions/*_sample_0.cif 2>/dev/null || true) | wc -l; }
arm() {
    local run=$1 k; shift
    [ "$(count $run)" -ge "$NS" ] && { echo "skip $run"; return 0; }
    for k in $(seq 1 $RETRIES); do
        rm -rf out/$SYS/$run
        make run SYSTEM=$SYS SEEDS=$SEEDS "$@" > out/$SYS/$run.log 2>&1
        [ "$(count $run)" -ge "$NS" ] && return 0
        log "$run: $(count $run)/$NS (attempt $k/$RETRIES)"; sleep 30
    done
    return 1
}
chain() {  # chain DATA DTAG
    local D=$1 DT=$2 M2=moment2_imgd${2}_w12_lr0.1 M3=moment3_imgd${2}_w12_m3wblock_lr0.1 prev t=""
    [ -f $D ] || { log "missing $D"; return 1; }
    arm $M2 METHOD=moment ORDER=2 LR=0.1 SPACE=image DATA=$D DTAG=$DT M12=white || return 1
    prev=$M2
    for S in 80 30 15; do
        t="${t}_seq$S"
        arm $M3$t METHOD=moment ORDER=3 LR=0.1 SPACE=image DATA=$D DTAG=$DT M12=white M3=wblock SIGMA0=$S TAG=$t \
            INIT="out/$SYS/$prev/*/seed_{seed}/predictions/*_sample_0.cif" || return 1
        prev=$M3$t
    done
}
pass() {  # pass SUFFIX expected-file stream2d-options...
    local suf=$1 want=$2; shift 2
    [ -f $want ] && { echo "skip data pass $suf"; return 0; }
    python -m moments.image_data stream2d --sim $SIM --n $N --seed 2 --suffix $suf "$@" > $SIM/stream2d$suf.log 2>&1 \
        || { log "data pass $suf FAILED (see $SIM/stream2d$suf.log)"; return 1; }
    grep -E "noise spectrum|conjugate|wrote" $SIM/stream2d$suf.log
}
summary() {
    local RUNS="moment3_imgd_snr16n1e5_w12_m3wblock_lr0.1$F"
    for DT in colest ast2d both colwhite astrad; do RUNS="$RUNS moment3_imgd_snr16n1e5${DT}_w12_m3wblock_lr0.1$F"; done
    make check SYSTEM=$SYS FIG=check_ctfnoise CHECK_DATA=$SIM/moments_N100000.npz RUNS="$RUNS" > out/$SYS/check_ctfnoise.log 2>&1
    python - <<'PY' 2>&1 | tee out/$SYS/ctfnoise_summary.log
import pandas as pd
d = pd.read_csv("out/8K9Z_A/check_ctfnoise.csv"); F = "_seq80_seq30_seq15"
run = lambda t: f"moment3_imgd_snr16n1e5{t}_w12_m3wblock_lr0.1{F}"
arms = [("baseline (white noise, radial CTF)", run("")), ("colored noise, estimated spectrum", run("colest")),
        ("astigmatism, 2D estimator", run("ast2d")), ("both, 2D + estimated spectrum", run("both")),
        ("control: colored noise as white", run("colwhite")), ("control: astigmatism ignored", run("astrad"))]
core = d.pivot_table(index="seed", columns="run", values="core"); b = run("")
print(f"8K9Z_A, final pass, 10 seeds\n{'arm':<38}{'median':>8}{'full':>7}{'core<2':>8}{'paired vs baseline':>22}{'sel.':>7}")
for lab, r in arms:
    x = d[d.run == r]
    if not len(x): print(f"{lab:<38}   (missing)"); continue
    p = (core[r] - core[b]).dropna() if r != b else None
    ps = f"{p.median():+.2f} A, better {int((p < 0).sum())}/{len(p)}" if p is not None else ""
    print(f"{lab:<38}{x.core.median():>8.2f}{x.rmsd.median():>7.2f}{(x.core < 2).mean():>8.0%}{ps:>22}"
          f"{x.loc[x.m3_img.idxmin(), 'core']:>7.2f}")
PY
}

f() { echo "$SIM/moments_N${N}_stream_$1.npz"; }
log "data: colored noise"
pass _col  $(f col_radial_white)  --noise colored --estimators radial --psds true,est,white
log "data: astigmatism (800 A), white noise"
pass _ast  $(f ast_radial_true)   --astig 800 --estimators 2d,2d0,radial --psds true
log "data: astigmatism + colored noise"
pass _both $(f both_2d_est)       --astig 800 --noise colored --estimators 2d --psds est
log "floors"
FILES="$SIM/moments_N100000.npz"
for x in col_radial_true col_radial_est col_radial_white ast_2d_true ast_2d0_true ast_radial_true both_2d_est; do
    [ -f $(f $x) ] && FILES="$FILES $(f $x)"; done
python -m moments.image_data floors --sim $SIM --files $FILES 2>&1 | grep "vs data" | tee out/$SYS/ctfnoise_floors.log

log "chain colest";   chain $(f col_radial_est)   _snr16n1e5colest   || log "colest FAILED"
log "chain ast2d";    chain $(f ast_2d_true)      _snr16n1e5ast2d    || log "ast2d FAILED"
log "chain both";     chain $(f both_2d_est)      _snr16n1e5both     || log "both FAILED"
log "summary (main arms)"; summary
log "chain colwhite"; chain $(f col_radial_white) _snr16n1e5colwhite || log "colwhite FAILED"
log "chain astrad";   chain $(f ast_radial_true)  _snr16n1e5astrad   || log "astrad FAILED"
log "summary (all arms)"; summary
cat out/$SYS/ctfnoise_floors.log
log "all done"

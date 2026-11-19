#!/bin/bash
# Two checks of the colored-noise-as-white result (8K9Z_A, 10 seeds, full chains; about 2.5 h):
#   colwhiterep : replicate of the colwhite chain (identical data; differences = run-to-run noise)
#   offset      : the clean baseline data (white noise, correct subtraction) plus the structure-independent offset
#                 that the white-noise assumption leaves in the colored data, Delta = col_radial_white - col_radial_true
#                 (same images, same accumulator; only the noise subtraction differs). If this also improves on the
#                 baseline, the offset itself helps (a loss-weighting effect), independent of colored noise.
#   setsid nohup bash experiments/colwhite_checks.sh > out/colwhite_checks.log 2>&1 < /dev/null &
set -uo pipefail
SYS=8K9Z_A; SEEDS=101,102,103,104,105,106,107,108,109,110; NS=10; SIM=out/$SYS/noise_snr16; F=_seq80_seq30_seq15
D0=$SIM/moments_N100000.npz; DW=$SIM/moments_N100000_stream_col_radial_white.npz
DT=$SIM/moments_N100000_stream_col_radial_true.npz; DO=$SIM/moments_N100000_offset.npz
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
chain() {
    local D=$1 DT_=$2 M2=moment2_imgd${2}_w12_lr0.1 M3=moment3_imgd${2}_w12_m3wblock_lr0.1 prev t=""
    arm $M2 METHOD=moment ORDER=2 LR=0.1 SPACE=image DATA=$D DTAG=$DT_ M12=white || return 1
    prev=$M2
    for S in 80 30 15; do
        t="${t}_seq$S"
        arm $M3$t METHOD=moment ORDER=3 LR=0.1 SPACE=image DATA=$D DTAG=$DT_ M12=white M3=wblock SIGMA0=$S TAG=$t \
            INIT="out/$SYS/$prev/*/seed_{seed}/predictions/*_sample_0.cif" || return 1
        prev=$M3$t
    done
}
for f in $D0 $DW $DT; do [ -f $f ] || { echo "missing $f"; exit 1; }; done
log "offset data"
python - "$D0" "$DW" "$DT" "$DO" <<'PY'
import sys, numpy as np
d0, dw, dt = (dict(np.load(f)) for f in sys.argv[1:4])
out = dict(d0)
for k in ("a", "c2", "c3"):
    delta = dw[k] - dt[k]
    out[k] = d0[k] + delta
    print(f"  {k}: |offset| / |baseline data| = {np.linalg.norm(delta) / np.linalg.norm(d0[k]):.3e}")
np.savez(sys.argv[4], **out)
PY
python -m moments.image_data floors --sim $SIM --files $D0 $DW $DO 2>&1 | grep "vs data" | tee out/$SYS/colwhite_floors.log
log "chain colwhiterep"; chain $DW _snr16n1e5colwhiterep || log "colwhiterep FAILED"
log "chain offset";      chain $DO _snr16n1e5offset      || log "offset FAILED"
log "check"
RUNS=""
for t in "" colwhite colwhiterep offset colest; do RUNS="$RUNS moment3_imgd_snr16n1e5${t}_w12_m3wblock_lr0.1$F"; done
make check SYSTEM=$SYS FIG=check_colwhite CHECK_DATA=$D0 RUNS="$RUNS" > out/$SYS/check_colwhite.log 2>&1
python - <<'PY' 2>&1 | tee out/$SYS/colwhite_summary.log
import pandas as pd
d = pd.read_csv("out/8K9Z_A/check_colwhite.csv"); F = "_seq80_seq30_seq15"
run = lambda t: f"moment3_imgd_snr16n1e5{t}_w12_m3wblock_lr0.1{F}"
arms = [("baseline", run("")), ("colored as white", run("colwhite")), ("colored as white, replicate", run("colwhiterep")),
        ("baseline data + white-noise offset", run("offset")), ("colored, estimated spectrum", run("colest"))]
core = d.pivot_table(index="seed", columns="run", values="core"); b = run("")
print(f"8K9Z_A, final pass, 10 seeds\n{'arm':<36}{'median':>8}{'core<2':>8}{'paired vs baseline':>22}   per seed")
for lab, r in arms:
    if r not in core: print(f"{lab:<36}   (missing)"); continue
    c = core[r]; p = (c - core[b]).dropna() if r != b else None
    ps = f"{p.median():+.2f} A, better {int((p < 0).sum())}/{len(p)}" if p is not None else ""
    print(f"{lab:<36}{c.median():>8.2f}{(c < 2).mean():>8.0%}{ps:>22}   " + " ".join(f"{v:.2f}" for v in c.values))
r1, r2 = run("colwhite"), run("colwhiterep")
if r1 in core and r2 in core:
    print(f"replicate agreement: median |colwhite - replicate| per seed = {(core[r1] - core[r2]).abs().median():.2f} A")
PY
log "all done"
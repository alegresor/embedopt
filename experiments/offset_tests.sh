#!/bin/bash
# Which part of the white-noise offset helps, and how much of it (8K9Z_A, 10 seeds, full chains; about 5 h).
# Delta = col_radial_white - col_radial_true (structure-independent; same images, only the noise subtraction differs).
# Data = clean baseline data + offset, with
#   c2only : Delta in c2 only            c3only : Delta in c3 only
#   x2     : 2 Delta (both)              x05    : 0.5 Delta (both)
# The full offset (x1, both) is the existing arm moment3_imgd_snr16n1e5offset_*.
#   setsid nohup bash experiments/offset_tests.sh > out/offset_tests.log 2>&1 < /dev/null &
set -uo pipefail
SYS=8K9Z_A; SEEDS=101,102,103,104,105,106,107,108,109,110; NS=10; SIM=out/$SYS/noise_snr16; F=_seq80_seq30_seq15
D0=$SIM/moments_N100000.npz; DW=$SIM/moments_N100000_stream_col_radial_white.npz
DT=$SIM/moments_N100000_stream_col_radial_true.npz
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
python - "$D0" "$DW" "$DT" "$SIM" <<'PY'
import sys, numpy as np
d0, dw, dt = (dict(np.load(f)) for f in sys.argv[1:4]); sim = sys.argv[4]
variants = {"c2only": dict(c2=1.0, c3=0.0), "c3only": dict(c2=0.0, c3=1.0), "x2": dict(c2=2.0, c3=2.0),
            "x05": dict(c2=0.5, c3=0.5)}
for name, w in variants.items():
    out = dict(d0)
    for k in ("c2", "c3"):
        out[k] = d0[k] + w[k] * (dw[k] - dt[k])
    np.savez(f"{sim}/moments_N100000_offset_{name}.npz", **out)
    print(f"  {name}: c2 x{w['c2']}, c3 x{w['c3']}")
PY
python -m moments.image_data floors --sim $SIM --files $SIM/moments_N100000_offset.npz \
    $(for v in c2only c3only x2 x05; do echo $SIM/moments_N100000_offset_$v.npz; done) 2>&1 | grep "vs data" \
    | tee out/$SYS/offset_floors.log
for v in c2only c3only x2 x05; do
    log "chain $v"; chain $SIM/moments_N100000_offset_$v.npz _snr16n1e5off$v || log "$v FAILED"
done
log "check"
RUNS=""
for t in "" offset offc2only offc3only offx2 offx05; do RUNS="$RUNS moment3_imgd_snr16n1e5${t}_w12_m3wblock_lr0.1$F"; done
make check SYSTEM=$SYS FIG=check_offset CHECK_DATA=$D0 RUNS="$RUNS" > out/$SYS/check_offset.log 2>&1
python - <<'PY' 2>&1 | tee out/$SYS/offset_summary.log
import pandas as pd
d = pd.read_csv("out/8K9Z_A/check_offset.csv"); F = "_seq80_seq30_seq15"
run = lambda t: f"moment3_imgd_snr16n1e5{t}_w12_m3wblock_lr0.1{F}"
arms = [("baseline (no offset)", run("")), ("offset x1, c2 + c3", run("offset")), ("offset x1, c2 only", run("offc2only")),
        ("offset x1, c3 only", run("offc3only")), ("offset x0.5, c2 + c3", run("offx05")), ("offset x2, c2 + c3", run("offx2"))]
core = d.pivot_table(index="seed", columns="run", values="core"); b = run("")
rg = d.pivot_table(index="seed", columns="run", values="rg") if "rg" in d else None
print(f"8K9Z_A, final pass, 10 seeds\n{'arm':<26}{'median':>8}{'full':>7}{'core<2':>8}{'paired vs baseline':>22}"
      f"{'Rg':>7}   per seed")
for lab, r in arms:
    if r not in core: print(f"{lab:<26}   (missing)"); continue
    x = d[d.run == r]; c = core[r]; p = (c - core[b]).dropna() if r != b else None
    ps = f"{p.median():+.2f} A, better {int((p < 0).sum())}/{len(p)}" if p is not None else ""
    g = f"{rg[r].median():7.2f}" if rg is not None else ""
    print(f"{lab:<26}{c.median():>8.2f}{x.rmsd.median():>7.2f}{(c < 2).mean():>8.0%}{ps:>22}{g}   "
          + " ".join(f"{v:.2f}" for v in c.values))
PY
log "all done"

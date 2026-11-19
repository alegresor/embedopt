#!/bin/bash
# Explicit reweighting of m12 against m3 (8K9Z_A, clean baseline data, 10 seeds; about 3-3.5 h).
#   loss = (m12 + eps) * m3^lambda.  Pass 1 (m12 alone) is unchanged by eps, so all arms restart from the existing
#   baseline pass 1 (moment2_imgd_snr16n1e5_w12_lr0.1) and run passes 2-4:
#     eps1e-3, eps1e-2, eps1e-1 : lambda = 1          lam2 : eps = 0, lambda = 2
#   Compared with the baseline chain and with the c2-only offset arm (1.44 A), seed by seed.
#   setsid nohup bash experiments/eps_tests.sh > out/eps_tests.log 2>&1 < /dev/null &
set -uo pipefail
SYS=8K9Z_A; SEEDS=101,102,103,104,105,106,107,108,109,110; NS=10; SIM=out/$SYS/noise_snr16; F=_seq80_seq30_seq15
D0=$SIM/moments_N100000.npz; P1=moment2_imgd_snr16n1e5_w12_lr0.1
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
restarts() {  # restarts DTAG make-args...
    local DT=$1 prev=$P1 t="" M3; shift
    M3=moment3_imgd${DT}_w12_m3wblock_lr0.1
    for S in 80 30 15; do
        t="${t}_seq$S"
        arm $M3$t METHOD=moment ORDER=3 LR=0.1 SPACE=image DATA=$D0 DTAG=$DT M12=white M3=wblock SIGMA0=$S TAG=$t \
            INIT="out/$SYS/$prev/*/seed_{seed}/predictions/*_sample_0.cif" "$@" || return 1
        prev=$M3$t
    done
}
[ "$(count $P1)" -ge "$NS" ] || { echo "missing $P1"; exit 1; }
for e in 1e-2 1e-1 1e-3; do log "eps $e"; restarts _snr16n1e5eps$e M12_EPS=$e || log "eps $e FAILED"; done
log "lambda 2"; restarts _snr16n1e5lam2 LAMBDA3=2 || log "lambda 2 FAILED"
log "check"
RUNS=""
for t in "" offc2only eps1e-3 eps1e-2 eps1e-1 lam2; do RUNS="$RUNS moment3_imgd_snr16n1e5${t}_w12_m3wblock_lr0.1$F"; done
make check SYSTEM=$SYS FIG=check_eps CHECK_DATA=$D0 RUNS="$RUNS" > out/$SYS/check_eps.log 2>&1
python - <<'PY' 2>&1 | tee out/$SYS/eps_summary.log
import pandas as pd
d = pd.read_csv("out/8K9Z_A/check_eps.csv"); F = "_seq80_seq30_seq15"
run = lambda t: f"moment3_imgd_snr16n1e5{t}_w12_m3wblock_lr0.1{F}"
arms = [("baseline (eps 0, lambda 1)", run("")), ("c2-only offset (earlier)", run("offc2only")),
        ("eps 1e-3", run("eps1e-3")), ("eps 1e-2", run("eps1e-2")), ("eps 1e-1", run("eps1e-1")), ("lambda 2", run("lam2"))]
core = d.pivot_table(index="seed", columns="run", values="core"); b = run("")
print(f"8K9Z_A, final pass, 10 seeds\n{'arm':<28}{'median':>8}{'full':>7}{'core<2':>8}{'paired vs baseline':>22}{'sel.':>7}   per seed")
for lab, r in arms:
    if r not in core: print(f"{lab:<28}   (missing)"); continue
    x = d[d.run == r]; c = core[r]; p = (c - core[b]).dropna() if r != b else None
    ps = f"{p.median():+.2f} A, better {int((p < 0).sum())}/{len(p)}" if p is not None else ""
    print(f"{lab:<28}{c.median():>8.2f}{x.rmsd.median():>7.2f}{(c < 2).mean():>8.0%}{ps:>22}"
          f"{x.loc[x.m3_img.idxmin(), 'core']:>7.2f}   " + " ".join(f"{v:.2f}" for v in c.values))
PY
log "all done"

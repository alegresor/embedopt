#!/bin/bash
# Is the m12 cap (loss = (m12 + eps) m3, eps = 0.1) general, and is it about estimation noise? About 1.5-2 h.
#   8K9Z_A exact image moments (no estimation noise)       : passes 2-4 from moment2_imgx_w12_lr0.1
#   8H1I_A, 8AHU_A SNR 1/16, N = 1e5 (as in the pipeline)  : passes 2-4 from moment2_imgd_snr16n1e5_w12_lr0.1
# Each compared seed by seed with the same chain at eps = 0.
#   setsid nohup bash experiments/eps_general.sh > out/eps_general.log 2>&1 < /dev/null &
set -uo pipefail
EPS=${EPS:-1e-1}; F=_seq80_seq30_seq15
declare -A SEEDS=([8AHU_A]=101,102,103,104,105 [8K9Z_A]=101,102,103,104,105,106,107,108,109,110
                  [8H1I_A]=101,102,103,104,105,106,107,108,109,110)
log() { echo "=== $(date '+%F %T') $*"; }
count() { (ls out/$1/$2/*/seed_*/predictions/*_sample_0.cif 2>/dev/null || true) | wc -l; }
arm() {
    local sys=$1 run=$2 n k; shift 2
    n=$(echo ${SEEDS[$sys]//,/ } | wc -w)
    [ "$(count $sys $run)" -ge "$n" ] && { echo "skip $sys/$run"; return 0; }
    for k in 1 2 3; do
        rm -rf out/$sys/$run
        make run SYSTEM=$sys SEEDS=${SEEDS[$sys]} "$@" > out/$sys/$run.log 2>&1
        [ "$(count $sys $run)" -ge "$n" ] && return 0
        log "$sys/$run: $(count $sys $run)/$n (attempt $k)"; sleep 30
    done
    return 1
}
restarts() {  # restarts SYS PASS1 M3PREFIX make-args...
    local sys=$1 prev=$2 M3=$3 t=""; shift 3
    [ "$(count $sys $prev)" -ge 1 ] || { log "missing $sys/$prev"; return 1; }
    for S in 80 30 15; do
        t="${t}_seq$S"
        arm $sys $M3$t METHOD=moment ORDER=3 LR=0.1 SPACE=image M12=white M3=wblock SIGMA0=$S TAG=$t M12_EPS=$EPS \
            INIT="out/$sys/$prev/*/seed_{seed}/predictions/*_sample_0.cif" "$@" || return 1
        prev=$M3$t
    done
}
log "8K9Z_A exact moments"
restarts 8K9Z_A moment2_imgx_w12_lr0.1 moment3_imgx_eps${EPS}_w12_m3wblock_lr0.1 DATA=exact DTAG=_eps$EPS || log "8K9Z_A FAILED"
for SYS in 8H1I_A 8AHU_A; do
    log "$SYS SNR 1/16"
    restarts $SYS moment2_imgd_snr16n1e5_w12_lr0.1 moment3_imgd_snr16n1e5eps${EPS}_w12_m3wblock_lr0.1 \
        DATA=out/$SYS/noise_snr16/moments_N100000.npz DTAG=_snr16n1e5eps$EPS || log "$SYS FAILED"
done
log "checks"
make check SYSTEM=8K9Z_A FIG=check_epsgen RUNS="moment3_imgx_w12_m3wblock_lr0.1$F moment3_imgx_eps${EPS}_w12_m3wblock_lr0.1$F" \
    > out/8K9Z_A/check_epsgen.log 2>&1
for SYS in 8H1I_A 8AHU_A; do
    make check SYSTEM=$SYS FIG=check_epsgen CHECK_DATA=out/$SYS/noise_snr16/moments_N100000.npz \
        RUNS="moment3_imgd_snr16n1e5_w12_m3wblock_lr0.1$F moment3_imgd_snr16n1e5eps${EPS}_w12_m3wblock_lr0.1$F" \
        > out/$SYS/check_epsgen.log 2>&1
done
EPS=$EPS python - <<'PY' 2>&1 | tee out/eps_general_summary.log
import os, pandas as pd
eps, F = os.environ["EPS"], "_seq80_seq30_seq15"
cases = [("8K9Z_A exact moments", "8K9Z_A", f"moment3_imgx_w12_m3wblock_lr0.1{F}", f"moment3_imgx_eps{eps}_w12_m3wblock_lr0.1{F}"),
         ("8H1I_A SNR 1/16", "8H1I_A", f"moment3_imgd_snr16n1e5_w12_m3wblock_lr0.1{F}", f"moment3_imgd_snr16n1e5eps{eps}_w12_m3wblock_lr0.1{F}"),
         ("8AHU_A SNR 1/16", "8AHU_A", f"moment3_imgd_snr16n1e5_w12_m3wblock_lr0.1{F}", f"moment3_imgd_snr16n1e5eps{eps}_w12_m3wblock_lr0.1{F}")]
print(f"loss = (m12 + {eps}) m3 against eps = 0, final pass\n{'case':<22}{'eps 0: med  <2A  sel':>24}{'eps: med  <2A  sel':>24}{'paired':>22}")
for lab, sys_, r0, r1 in cases:
    f = f"out/{sys_}/check_epsgen.csv"
    if not os.path.isfile(f): print(f"{lab:<22}   (no {f})"); continue
    d = pd.read_csv(f); core = d.pivot_table(index="seed", columns="run", values="core")
    if r0 not in core or r1 not in core: print(f"{lab:<22}   (missing runs)"); continue
    sel = "m3_img" if "m3_img" in d else "m3"
    s = lambda r: (lambda x: f"{x.core.median():6.2f} {(x.core < 2).mean():5.0%} {x.loc[x[sel].idxmin(), 'core']:6.2f}")(d[d.run == r])
    p = (core[r1] - core[r0]).dropna()
    print(f"{lab:<22}{s(r0):>24}{s(r1):>24}{p.median():+10.2f} A, {int((p < 0).sum())}/{len(p)}")
    print(f"{'':<22}per seed eps 0: " + " ".join(f"{v:.2f}" for v in core[r0].values))
    print(f"{'':<22}per seed eps  : " + " ".join(f"{v:.2f}" for v in core[r1].values))
PY
log "all done"

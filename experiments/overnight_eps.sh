#!/bin/bash
# Overnight (about 3.5-4 h):
# Part 1 (about 1.5 h): the map pass of the two-stage pipeline from the eps-improved moment results.
#   For 8K9Z_A, 8H1I_A, 8AHU_A (SNR 1/16, N = 1e5): restart every seed of moment3_imgd_snr16n1e5eps1e-1_*_seq15 at 30 A,
#   map reward on the known-pose and the ab-initio RELION maps, map-based frame (MAP_FRAME=fit; the docked moment model
#   from the pipeline only initializes the frame). Compared with the same map pass from the eps = 0 moment results.
# Part 2 (about 2 h): more seeds against 8H1I_A's wrong domain arrangements. 20 new seeds (111-130), the same chain
#   (moment2 w12, then 3 restarts moment3 w12 wblock with eps = 0.1); with the existing 10 that gives 30, and the
#   question is whether lowest-m3 selection over 30 seeds still picks a correct one, and how often seeds land correctly.
#   setsid nohup bash experiments/overnight_eps.sh > out/overnight_eps.log 2>&1 < /dev/null &
set -uo pipefail
F=_seq80_seq30_seq15; EPS=1e-1
declare -A SEEDS=([8AHU_A]=101,102,103,104,105 [8K9Z_A]=101,102,103,104,105,106,107,108,109,110
                  [8H1I_A]=101,102,103,104,105,106,107,108,109,110)
NEW=111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130
log() { echo "=== $(date '+%F %T') $*"; }
count() { (ls out/$1/$2/*/seed_*/predictions/*_sample_0.cif 2>/dev/null || true) | wc -l; }
arm() {  # arm SYS RUN SEEDLIST make-args...
    local sys=$1 run=$2 sd=$3 n k; shift 3
    n=$(echo ${sd//,/ } | wc -w)
    [ "$(count $sys $run)" -ge "$n" ] && { echo "skip $sys/$run"; return 0; }
    for k in 1 2 3; do
        rm -rf out/$sys/$run
        make run SYSTEM=$sys SEEDS=$sd "$@" > out/$sys/$run.log 2>&1
        [ "$(count $sys $run)" -ge "$n" ] && return 0
        log "$sys/$run: $(count $sys $run)/$n (attempt $k)"; sleep 30
    done
    return 1
}
MOMEPS=moment3_imgd_snr16n1e5eps${EPS}_w12_m3wblock_lr0.1$F

log "part 1: map pass from the eps moment results"
export MAP_FRAME=fit
for SYS in 8K9Z_A 8H1I_A 8AHU_A; do
    SIM=out/$SYS/noise_snr16
    [ "$(count $SYS $MOMEPS)" -ge 1 ] || { log "$SYS: missing $MOMEPS"; continue; }
    for A in relpose_m4:dock_poses_m relfull_m4:dock_full_m; do
        IFS=: read T2 D <<< "$A"
        [ -f $SIM/$D/map_crop.mrc ] || { log "$SYS: missing $SIM/$D/map_crop.mrc"; continue; }
        log "$SYS map pass $T2"
        arm $SYS map_${T2}_lr0.5_from_eps ${SEEDS[$SYS]} METHOD=map LR=0.5 MAP=$SIM/$D/map_crop.mrc \
            MAPREF=$SIM/$D/docked.pdb MTAG=_$T2 SIGMA0=30 TAG=_from_eps \
            INIT="out/$SYS/$MOMEPS/*/seed_{seed}/predictions/*_sample_0.cif" || log "$SYS $T2 FAILED"
    done
done
unset MAP_FRAME
for SYS in 8K9Z_A 8H1I_A 8AHU_A; do
    RUNS="base map_lr0.5$F moment3_imgd_snr16n1e5_w12_m3wblock_lr0.1$F $MOMEPS"
    RUNS="$RUNS map_relpose_m3_lr0.5_from_moments map_relfull_m3_lr0.5_from_moments map_relpose_m4_lr0.5_from_eps map_relfull_m4_lr0.5_from_eps"
    make check SYSTEM=$SYS FIG=check_mapeps CHECK_DATA=out/$SYS/noise_snr16/moments_N100000.npz RUNS="$RUNS" \
        > out/$SYS/check_mapeps.log 2>&1
done
python - <<'PY' 2>&1 | tee out/mapeps_summary.log
import os, pandas as pd
F = "_seq80_seq30_seq15"
arms = [("unguided prior", "base"), ("map, oracle (per-step alignment to x*)", f"map_lr0.5{F}"),
        ("moments, eps 0", f"moment3_imgd_snr16n1e5_w12_m3wblock_lr0.1{F}"),
        ("moments, eps 0.1", f"moment3_imgd_snr16n1e5eps1e-1_w12_m3wblock_lr0.1{F}"),
        ("moments eps 0 -> known-pose map", "map_relpose_m3_lr0.5_from_moments"),
        ("moments eps 0 -> ab-initio map", "map_relfull_m3_lr0.5_from_moments"),
        ("moments eps 0.1 -> known-pose map", "map_relpose_m4_lr0.5_from_eps"),
        ("moments eps 0.1 -> ab-initio map", "map_relfull_m4_lr0.5_from_eps")]
for s in ("8H1I_A", "8AHU_A", "8K9Z_A"):
    f = f"out/{s}/check_mapeps.csv"
    if not os.path.isfile(f): print(f"\n{s}: no {f}"); continue
    d = pd.read_csv(f)
    print(f"\n=== {s} (final pass)\n{'arm':<42}{'n':>4}{'core':>7}{'full':>7}{'core<2':>8}{'sel.':>7}")
    for lab, r in arms:
        x = d[d.run == r]
        if not len(x): print(f"{lab:<42}   (missing)"); continue
        print(f"{lab:<42}{len(x):>4}{x.core.median():>7.2f}{x.rmsd.median():>7.2f}{(x.core < 2).mean():>8.0%}"
              f"{x.loc[x.m3_img.idxmin(), 'core']:>7.2f}")
PY

log "part 2: 20 more seeds on 8H1I_A"
SYS=8H1I_A; D=out/$SYS/noise_snr16/moments_N100000.npz; DT=_snr16n1e5x
M2=moment2_imgd${DT}_w12_lr0.1; M3=moment3_imgd${DT}eps${EPS}_w12_m3wblock_lr0.1
arm $SYS $M2 $NEW METHOD=moment ORDER=2 LR=0.1 SPACE=image DATA=$D DTAG=$DT M12=white && {
    prev=$M2; t=""
    for S in 80 30 15; do
        t="${t}_seq$S"
        arm $SYS $M3$t $NEW METHOD=moment ORDER=3 LR=0.1 SPACE=image DATA=$D DTAG=${DT}eps$EPS M12=white M3=wblock \
            M12_EPS=$EPS SIGMA0=$S TAG=$t INIT="out/$SYS/$prev/*/seed_{seed}/predictions/*_sample_0.cif" || break
        prev=$M3$t
    done; } || log "part 2 FAILED"
make check SYSTEM=$SYS FIG=check_seeds30 CHECK_DATA=$D RUNS="$M2 $M3$F moment2_imgd_snr16n1e5_w12_lr0.1 $MOMEPS" \
    > out/$SYS/check_seeds30.log 2>&1
python - <<'PY' 2>&1 | tee out/seeds30_summary.log
import pandas as pd
F = "_seq80_seq30_seq15"
d = pd.read_csv("out/8H1I_A/check_seeds30.csv")
fin = d[d.run.isin([f"moment3_imgd_snr16n1e5xeps1e-1_w12_m3wblock_lr0.1{F}", f"moment3_imgd_snr16n1e5eps1e-1_w12_m3wblock_lr0.1{F}"])]
p1 = d[d.run.isin(["moment2_imgd_snr16n1e5x_w12_lr0.1", "moment2_imgd_snr16n1e5_w12_lr0.1"])]
print(f"8H1I_A, eps 0.1 chain, {len(fin)} seeds (101-130)")
print(f"  final pass: core median {fin.core.median():.2f} A, core < 2 A in {(fin.core < 2).mean():.0%} "
      f"({int((fin.core < 2).sum())}/{len(fin)}); correct arrangement after pass 1 (core < 5 A): "
      f"{int((p1.core < 5).sum())}/{len(p1)}")
for k in (10, 20, 30):
    sub = fin.sort_values("seed").head(k)
    print(f"  lowest-m3 selection over the first {k:>2} seeds: core {sub.loc[sub.m3_img.idxmin(), 'core']:.2f} A "
          f"(best {sub.core.min():.2f} A)")
s = fin.sort_values("m3_img")[["seed", "core", "m3_img"]].head(8)
print("  lowest m3_img seeds:\n" + s.to_string(index=False))
PY
log "all done"
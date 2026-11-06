#!/bin/bash
# Image-space moment chains: the 4-pass chain of experiments/chain.sh with the reward compared in image space.
#   STAGE=exact : data = moments of x* through the same maps (oracle; tests the redefined image-space m3)  -> *_imgx_*
#   STAGE=data  : data = moments estimated from N = 1e6 rendered images (out/<SYS>/image_moments_N1000000.npz) -> *_imgd_*
# Chains (pass 1 cold, 100 steps; passes 2-4 restart from the previous pass at 80, 30, 15 A):
#   moment2<T> : m12 in every pass (control);   moment3<T> : m12 in pass 1 (shared), then m12 * m3
# The map and invariant-space chains of chain.sh are reused for comparison if present (not rerun).
# Runs that already have all predictions are skipped. Run from the repo root.
#   STAGE=exact SYSTEMS="8AHU_A" bash experiments/image_chain.sh
set -euo pipefail
STAGE=${STAGE:-exact}
case $STAGE in
    exact) DATA=exact;  IT=_imgx ;;
    data)  DATA=images; IT=_imgd ;;
    *) echo "STAGE must be exact or data"; exit 1 ;;
esac
SYSTEMS=${SYSTEMS:-"8AHU_A 8K9Z_A 8H1I_A"}
declare -A SEEDS=([8AHU_A]=101,102,103,104,105 [8K9Z_A]=101,102,103,104,105,106,107,108,109,110
                  [8H1I_A]=101,102,103,104,105,106,107,108,109,110)
LEVELS="80 30 15"
BENCH=examples/synthetic_map_benchmark

count() { (ls out/$1/$2/*/seed_*/predictions/*_sample_0.cif 2>/dev/null || true) | wc -l; }
arm() {  # arm SYS RUN make-args...: run unless complete
    local sys=$1 run=$2; shift 2
    local n=$(echo ${SEEDS[$sys]//,/ } | wc -w)
    if [ "$(count $sys $run)" -eq "$n" ]; then echo "skip $sys/$run (complete)"; return; fi
    rm -rf out/$sys/$run
    make run SYSTEM=$sys SEEDS=${SEEDS[$sys]} "$@" 2>&1 | tee out/$sys/$run.log
    [ "$(count $sys $run)" -eq "$n" ] || { echo "$sys/$run: only $(count $sys $run)/$n predictions; see out/$sys/$run/*/ERR"; exit 1; }
}

for SYS in $SYSTEMS; do
    mkdir -p out/$SYS
    [ -f $BENCH/stage3a_cfgs/${SYS}_benchmark.json ] || make cfg SYSTEM=$SYS
    if [ $STAGE = data ] && [ ! -f out/$SYS/image_moments_N1000000.npz ]; then
        echo "missing out/$SYS/image_moments_N1000000.npz: run make imagechecks IC_SYSTEMS=$SYS"; exit 1
    fi
    IMG="SPACE=image DATA=$DATA"

    arm $SYS moment2${IT}_lr0.1 METHOD=moment ORDER=2 LR=0.1 $IMG
    for CHAIN in "2:moment2${IT}_lr0.1" "3:moment3${IT}_lr0.1"; do
        IFS=: read ORDER PREFIX <<< "$CHAIN"
        PREV=moment2${IT}_lr0.1 TAG=""
        for S in $LEVELS; do
            TAG="${TAG}_seq$S"
            arm $SYS $PREFIX$TAG METHOD=moment ORDER=$ORDER LR=0.1 $IMG SIGMA0=$S TAG=$TAG \
                INIT="out/$SYS/$PREV/*/seed_{seed}/predictions/*_sample_0.cif"
            PREV=$PREFIX$TAG
        done
    done

    RUNS="base"
    for P in map_lr0.5 moment2_lr0.1 moment2${IT}_lr0.1; do
        RUNS="$RUNS $P ${P}_seq80 ${P}_seq80_seq30 ${P}_seq80_seq30_seq15"
    done
    for P in moment3_lr0.1 moment3${IT}_lr0.1; do
        RUNS="$RUNS ${P}_seq80 ${P}_seq80_seq30 ${P}_seq80_seq30_seq15"
    done
    make check SYSTEM=$SYS FIG=check_chain$IT CHECK_DATA=$DATA RUNS="$RUNS" 2>&1 | tee out/$SYS/check_chain$IT.log
    SYS=$SYS TAG=$IT SELECT=m3_img python -m moments.chain_summary 2>&1 | tee -a out/$SYS/check_chain$IT.log
done
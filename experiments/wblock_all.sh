#!/bin/bash
# Whitened m12 + block-normalized bispectrum m3 (wblock) on all three systems, with exact moments (imgx) and with
# moments estimated from N = 1e6 images (imgd; out/<SYS>/image_moments_N1000000.npz), lowest-8-shell grid, |q| <= 16:
#   pass 1      moment2_img<x|d>_w12              : m12 white                          cold, 100 steps
#   passes 2-4  moment3_img<x|d>_w12_m3wblock     : m12 white * m3 wblock, restarts from the previous pass at 80/30/15 A
# Then, per system and stage: check (all image-space loss variants, scored against the same data) and the chain
# summary against base, the map chain, the invariant chains and the earlier image-space chains (moment{2,3}_img<x|d>).
# Completed runs are skipped (e.g. 8K9Z_A exact from experiments/whitening.sh). Run from the repo root:
#   bash experiments/wblock_all.sh            [STAGES="exact data"] [SYSTEMS="8AHU_A 8K9Z_A 8H1I_A"]
set -euo pipefail
STAGES=${STAGES:-"exact data"}
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

for STAGE in $STAGES; do
    case $STAGE in exact) DATA=exact; IT=_imgx ;; data) DATA=images; IT=_imgd ;; *) echo "bad stage $STAGE"; exit 1 ;; esac
    for SYS in $SYSTEMS; do
        mkdir -p out/$SYS
        [ -f $BENCH/stage3a_cfgs/${SYS}_benchmark.json ] || make cfg SYSTEM=$SYS
        if [ $STAGE = data ] && [ ! -f out/$SYS/image_moments_N1000000.npz ]; then
            echo "missing out/$SYS/image_moments_N1000000.npz: run make imagechecks IC_SYSTEMS=$SYS"; exit 1
        fi
        P1=moment2${IT}_w12_lr0.1
        arm $SYS $P1 METHOD=moment ORDER=2 LR=0.1 SPACE=image DATA=$DATA M12=white
        PREV=$P1 TAG=""
        for S in $LEVELS; do
            TAG="${TAG}_seq$S"
            RUN=moment3${IT}_w12_m3wblock_lr0.1$TAG
            arm $SYS $RUN METHOD=moment ORDER=3 LR=0.1 SPACE=image DATA=$DATA M12=white M3=wblock SIGMA0=$S TAG=$TAG \
                INIT="out/$SYS/$PREV/*/seed_{seed}/predictions/*_sample_0.cif"
            PREV=$RUN
        done

        RUNS="base $P1"
        for P in map_lr0.5 moment2_lr0.1 moment2${IT}_lr0.1; do RUNS="$RUNS $P ${P}_seq80 ${P}_seq80_seq30 ${P}_seq80_seq30_seq15"; done
        for P in moment3_lr0.1 moment3${IT}_lr0.1 moment3${IT}_w12_m3wblock_lr0.1; do
            RUNS="$RUNS ${P}_seq80 ${P}_seq80_seq30 ${P}_seq80_seq30_seq15"
        done
        FIG=check_wblock$IT
        make check SYSTEM=$SYS FIG=$FIG CHECK_DATA=$DATA RUNS="$RUNS" 2>&1 | tee out/$SYS/$FIG.log
        for SEL in m3_img m3b_img; do
            SYS=$SYS CSV=out/$SYS/$FIG.csv TAG=${IT}_w12 TAGS3="${IT}_w12_m3wblock" EXTRA=$IT SELECT=$SEL \
                python -m moments.chain_summary 2>&1 | tee -a out/$SYS/$FIG.log
        done
    done
done

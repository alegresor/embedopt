#!/bin/bash
# Fair multi-pass comparison: every method gets the same 4-pass chain on every system.
#   pass 1: cold start (100 steps)
#   pass 2: restart from pass 1 at 80 A (53 steps), pass 3: from pass 2 at 30 A (44), pass 4: from pass 3 at 15 A (38)
# Chains (same reward in every pass unless noted):
#   map     : map_lr0.5        -> map_lr0.5_seq80        -> ..._seq30        -> ..._seq15
#   moment2 : moment2_lr0.1    -> moment2_lr0.1_seq80    -> ..._seq30        -> ..._seq15   (m12 only; control)
#   moment3 : moment2_lr0.1    -> moment3_lr0.1_seq80    -> ..._seq30        -> ..._seq15   (pass 1 m12, then m12 * m3)
# Plus the unguided base. Runs that already have all predictions are reused. Run from the repo root.
#   SYSTEMS="8H1I_A" bash experiments/chain.sh     # a subset of systems
set -euo pipefail
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
    [ -f $BENCH/synthetic_maps/5.0A/$SYS.map ] || make maps

    # pass 1 (cold start)
    arm $SYS base          METHOD=base
    arm $SYS map_lr0.5     METHOD=map LR=0.5
    arm $SYS moment2_lr0.1 METHOD=moment ORDER=2 LR=0.1

    # passes 2-4 for each chain
    for CHAIN in "map:0:0.5:map_lr0.5:map_lr0.5" "moment:2:0.1:moment2_lr0.1:moment2_lr0.1" "moment:3:0.1:moment2_lr0.1:moment3_lr0.1"; do
        IFS=: read METHOD ORDER LR PREV PREFIX <<< "$CHAIN"
        TAG=""
        for S in $LEVELS; do
            TAG="${TAG}_seq$S"
            arm $SYS $PREFIX$TAG METHOD=$METHOD ORDER=$ORDER LR=$LR SIGMA0=$S TAG=$TAG \
                INIT="out/$SYS/$PREV/*/seed_{seed}/predictions/*_sample_0.cif"
            PREV=$PREFIX$TAG
        done
    done

    T="_seq80_seq30_seq15"
    make check SYSTEM=$SYS FIG=check_chain RUNS="base map_lr0.5 map_lr0.5_seq80 map_lr0.5_seq80_seq30 map_lr0.5$T moment2_lr0.1 \
        moment2_lr0.1_seq80 moment2_lr0.1_seq80_seq30 moment2_lr0.1$T \
        moment3_lr0.1_seq80 moment3_lr0.1_seq80_seq30 moment3_lr0.1$T" 2>&1 | tee out/$SYS/check_chain.log
    SYS=$SYS python -m moments.chain_summary 2>&1 | tee -a out/$SYS/check_chain.log
done

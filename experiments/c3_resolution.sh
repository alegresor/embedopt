#!/bin/bash
# Does the image-space m3 guide worse than the invariant m3 because of its third-order grid? (8K9Z_A, exact data)
#   moment3_imgx_c3s2q24 : passes 2-4 of the image-space m3 chain with 8 shells strided over all 16 (out to 5 A,
#                          as the invariant bispectrum) and |q| <= 24                                    (the test)
#   moment3_imgx_rep     : passes 2-4 of the existing chain (lowest 8 shells, |q| <= 16), rerun as-is  (noise floor)
# Both restart from the existing pass 1 (moment2_imgx_lr0.1, order 2: unaffected by the third-order grid).
# Then: check scored with the default and with the new third-order grid, and chain summaries for both.
#   bash experiments/c3_resolution.sh            (from the repo root; completed runs are skipped)
set -euo pipefail
SYS=${SYS:-8K9Z_A}
SEEDS=${SEEDS:-101,102,103,104,105,106,107,108,109,110}
NEW="S3STRIDE=2 Q3=24"
LEVELS="80 30 15"

count() { (ls out/$SYS/$1/*/seed_*/predictions/*_sample_0.cif 2>/dev/null || true) | wc -l; }
N=$(echo ${SEEDS//,/ } | wc -w)
arm() {  # arm RUN make-args...
    local run=$1; shift
    if [ "$(count $run)" -eq "$N" ]; then echo "skip $SYS/$run (complete)"; return; fi
    rm -rf out/$SYS/$run
    make run SYSTEM=$SYS SEEDS=$SEEDS "$@" 2>&1 | tee out/$SYS/$run.log
    [ "$(count $run)" -eq "$N" ] || { echo "$SYS/$run: only $(count $run)/$N predictions; see out/$SYS/$run/*/ERR"; exit 1; }
}
[ "$(count moment2_imgx_lr0.1)" -eq "$N" ] || { echo "missing pass 1: out/$SYS/moment2_imgx_lr0.1 (run experiments/image_chain.sh)"; exit 1; }

for V in "_c3s2q24:$NEW" "_rep:VTAG=_rep"; do
    IFS=: read VT ARGS <<< "$V"
    PREV=moment2_imgx_lr0.1 TAG=""
    for S in $LEVELS; do
        TAG="${TAG}_seq$S"
        RUN=moment3_imgx${VT}_lr0.1$TAG
        arm $RUN METHOD=moment ORDER=3 LR=0.1 SPACE=image DATA=exact $ARGS SIGMA0=$S TAG=$TAG \
            INIT="out/$SYS/$PREV/*/seed_{seed}/predictions/*_sample_0.cif"
        PREV=$RUN
    done
done

RUNS="base"
for P in map_lr0.5 moment2_lr0.1 moment2_imgx_lr0.1; do RUNS="$RUNS $P ${P}_seq80 ${P}_seq80_seq30 ${P}_seq80_seq30_seq15"; done
for P in moment3_lr0.1 moment3_imgx_lr0.1 moment3_imgx_rep_lr0.1 moment3_imgx_c3s2q24_lr0.1; do
    RUNS="$RUNS ${P}_seq80 ${P}_seq80_seq30 ${P}_seq80_seq30_seq15"
done
TAGS3="_imgx _imgx_rep _imgx_c3s2q24"
for SC in "c3:" "c3_s2q24:$NEW"; do  # m3_img scored on the old grid, then on the new one
    IFS=: read FIG ARGS <<< "$SC"
    make check SYSTEM=$SYS FIG=check_$FIG CHECK_DATA=exact $ARGS RUNS="$RUNS" 2>&1 | tee out/$SYS/check_$FIG.log
    SYS=$SYS CSV=out/$SYS/check_$FIG.csv TAG=_imgx TAGS3="$TAGS3" SELECT=m3_img python -m moments.chain_summary \
        2>&1 | tee -a out/$SYS/check_$FIG.log
done

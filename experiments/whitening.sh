#!/bin/bash
# Reweighted image-space losses on 8K9Z_A (exact data), all from one cold pass 1 with the whitened m12:
#   pass 1   moment2_imgx_w12              : m12 white (= invariant m12 recovered from c2)      [always]
#   block    moment3_imgx_w12              : m12 white + m3 block (current c3 loss)              passes 2-4
#   perq     moment3_imgx_w12_m3perq       : m12 white + m3 perq                                 passes 2-4
#   white    moment3_imgx_w12_m3white      : m12 white + m3 white (bispectrum metric)            passes 2-4
#   white5   moment3_imgx_c3s2q24_w12_m3white : as white, on the 5 A third-order grid           passes 2-4
#   wblock   moment3_imgx_w12_m3wblock     : m12 white + m3 wblock (block-normalized bispectrum)  passes 2-4
#   wblock5  moment3_imgx_c3s2q24_w12_m3wblock : as wblock, on the 5 A third-order grid           passes 2-4
# VARIANTS selects the passes-2-4 chains (default: block wblock wblock5). perq and white barely see Kam's ambiguity
# (Kam scramble ~1e-6 in the unit tests), so they are expected to act like m12 alone; add them to VARIANTS if wanted. Completed runs are skipped. Run from the repo root:
#   VARIANTS="block" bash experiments/whitening.sh
set -euo pipefail
SYS=${SYS:-8K9Z_A}
SEEDS=${SEEDS:-101,102,103,104,105,106,107,108,109,110}
VARIANTS=${VARIANTS:-"block wblock wblock5"}
LEVELS="80 30 15"
declare -A ARGS=([block]="M12=white" [perq]="M12=white M3=perq" [white]="M12=white M3=white"
                 [white5]="M12=white M3=white S3STRIDE=2 Q3=24" [wblock]="M12=white M3=wblock"
                 [wblock5]="M12=white M3=wblock S3STRIDE=2 Q3=24")
declare -A NAME=([block]=_imgx_w12 [perq]=_imgx_w12_m3perq [white]=_imgx_w12_m3white [white5]=_imgx_c3s2q24_w12_m3white
                 [wblock]=_imgx_w12_m3wblock [wblock5]=_imgx_c3s2q24_w12_m3wblock)

count() { (ls out/$SYS/$1/*/seed_*/predictions/*_sample_0.cif 2>/dev/null || true) | wc -l; }
N=$(echo ${SEEDS//,/ } | wc -w)
arm() {  # arm RUN make-args...
    local run=$1; shift
    if [ "$(count $run)" -eq "$N" ]; then echo "skip $SYS/$run (complete)"; return; fi
    rm -rf out/$SYS/$run
    make run SYSTEM=$SYS SEEDS=$SEEDS "$@" 2>&1 | tee out/$SYS/$run.log
    [ "$(count $run)" -eq "$N" ] || { echo "$SYS/$run: only $(count $run)/$N predictions; see out/$SYS/$run/*/ERR"; exit 1; }
}

mkdir -p out/$SYS
P1=moment2_imgx_w12_lr0.1
arm $P1 METHOD=moment ORDER=2 LR=0.1 SPACE=image DATA=exact M12=white
TAGS3=""
for V in $VARIANTS; do
    PREV=$P1 TAG=""
    for S in $LEVELS; do
        TAG="${TAG}_seq$S"
        RUN=moment3${NAME[$V]}_lr0.1$TAG
        arm $RUN METHOD=moment ORDER=3 LR=0.1 SPACE=image DATA=exact ${ARGS[$V]} SIGMA0=$S TAG=$TAG \
            INIT="out/$SYS/$PREV/*/seed_{seed}/predictions/*_sample_0.cif"
        PREV=$RUN
    done
    TAGS3="$TAGS3 ${NAME[$V]}"
done

RUNS="base $P1"
for P in map_lr0.5 moment2_lr0.1 moment2_imgx_lr0.1; do RUNS="$RUNS $P ${P}_seq80 ${P}_seq80_seq30 ${P}_seq80_seq30_seq15"; done
for P in moment3_lr0.1 moment3_imgx_lr0.1 $(for t in $TAGS3; do echo moment3${t}_lr0.1; done); do
    RUNS="$RUNS ${P}_seq80 ${P}_seq80_seq30 ${P}_seq80_seq30_seq15"
done
make check SYSTEM=$SYS FIG=check_whitening CHECK_DATA=exact RUNS="$RUNS" 2>&1 | tee out/$SYS/check_whitening.log
SYS=$SYS CSV=out/$SYS/check_whitening.csv TAG=_imgx_w12 TAGS3="$TAGS3" EXTRA=_imgx SELECT=m3_img \
    python -m moments.chain_summary 2>&1 | tee -a out/$SYS/check_whitening.log

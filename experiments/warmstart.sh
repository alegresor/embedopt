#!/bin/bash
# Partial-diffusion diagnostic (~25 min, 8K9Z_A): is the third-order reward informative once inside its basin?
# Start from a near-native structure (the earlier map_lr0.5 seed-101 prediction, core RMSD 0.31 A), noise it to
# SIGMA0 and run only the remaining diffusion steps under each reward, from the original trunk embedding:
#   moment2_lr0.0_pd<S> : no guidance           -> how far the prior alone pulls the structure away (control)
#   moment2_lr0.1_pd<S> : second-order guidance -> does m12 alone drift (local Kam degeneracy)?
#   moment3_lr0.1_pd<S> : third-order guidance  -> the test: does m3 hold / improve the structure?
# Two starting noise levels: 3 A (local stability) and 10 A (basin capture). Run from the repo root.
set -euo pipefail
SYS=8K9Z_A
SEEDS=101,102,103,104,105
INIT=$(find out/$SYS/map_lr0.5 -path "*seed_101/predictions/*_sample_0.cif" | head -1)
test -n "$INIT" || { echo "starting structure not found under out/$SYS/map_lr0.5"; exit 1; }
echo "Starting structure: $INIT"

RUNS="map_lr0.5"
for S in 3 10; do
    for A in "2 0.0" "2 0.1" "3 0.1"; do
        set -- $A
        make run SYSTEM=$SYS METHOD=moment ORDER=$1 LR=$2 SEEDS=$SEEDS INIT=$INIT SIGMA0=$S TAG=_pd$S \
            2>&1 | tee out/$SYS/moment$1_lr$2_pd$S.log
        RUNS="$RUNS moment$1_lr$2_pd$S"
    done
done
make check SYSTEM=$SYS FIG=check_pd RUNS="$RUNS" 2>&1 | tee out/$SYS/check_pd.log

#!/bin/bash
# Warm-start diagnostic (~25 min, 8K9Z_A): does the third-order reward keep / improve a near-native structure?
#  1. Rerun map_lr0.5 seed 101 (previously core RMSD 0.31 A) and save its final optimized embedding.
#  2. From that embedding, with 5 fresh-noise seeds:
#       moment2_lr0.0_warm : no guidance  -> what the warm embedding alone produces (control)
#       moment2_lr0.1_warm : second-order guidance -> does m12 alone drift away (Kam degeneracy)?
#       moment3_lr0.1_warm : third-order guidance  -> the test
#  3. Focused check figure out/8K9Z_A/check_warm.png and table.
# Run from the repo root.
set -euo pipefail
SYS=8K9Z_A
SEEDS=101,102,103,104,105

make run SYSTEM=$SYS METHOD=map LR=0.5 SEEDS=101 SAVE_EMB=1 TAG=_save 2>&1 | tee out/$SYS/map_lr0.5_save.log
EMB=out/$SYS/map_lr0.5_save/${SYS}_map_lr0.5_save/seed_101/opted_vectors/embeddings_step_99.pt
test -f $EMB || { echo "embedding not found: $EMB"; ls -R out/$SYS/map_lr0.5_save | head -30; exit 1; }

make run SYSTEM=$SYS METHOD=moment ORDER=2 LR=0.0 SEEDS=$SEEDS WARM=$EMB TAG=_warm 2>&1 | tee out/$SYS/moment2_lr0.0_warm.log
make run SYSTEM=$SYS METHOD=moment ORDER=2 LR=0.1 SEEDS=$SEEDS WARM=$EMB TAG=_warm 2>&1 | tee out/$SYS/moment2_lr0.1_warm.log
make run SYSTEM=$SYS METHOD=moment ORDER=3 LR=0.1 SEEDS=$SEEDS WARM=$EMB TAG=_warm 2>&1 | tee out/$SYS/moment3_lr0.1_warm.log

make check SYSTEM=$SYS FIG=check_warm \
     RUNS="base map_lr0.5 map_lr0.5_save moment2_lr0.0_warm moment2_lr0.1_warm moment3_lr0.1_warm" 2>&1 | tee out/$SYS/check_warm.log

#!/bin/bash
# Stage 3a experiment (~90 min): {base, moment, map} x LR in {0.1, 0.5}; 5 seeds for 8AHU_A (easy sanity check),
# 10 seeds for 8K9Z_A (hard target, where success rates matter).
# Outputs: out/<SYS>/{base, moment_lr<LR>, map_lr<LR>}/, per-run logs, out/<SYS>/summary.csv and out/<SYS>/check.png
set -euo pipefail

declare -A SEEDS=(
    [8AHU_A]=101,102,103,104,105
    [8K9Z_A]=101,102,103,104,105,106,107,108,109,110
)
LRS="0.1 0.5"

for SYS in "${!SEEDS[@]}"; do make cfg SYSTEM=$SYS; done
make maps   # 5 A synthetic maps for all systems (needed by METHOD=map)

for SYS in 8AHU_A 8K9Z_A; do
    S=${SEEDS[$SYS]}
    mkdir -p out/$SYS
    make run SYSTEM=$SYS SEEDS=$S METHOD=base 2>&1 | tee out/$SYS/base.log
    for LR in $LRS; do
        make run SYSTEM=$SYS SEEDS=$S METHOD=moment LR=$LR 2>&1 | tee out/$SYS/moment_lr$LR.log
        make run SYSTEM=$SYS SEEDS=$S METHOD=map    LR=$LR 2>&1 | tee out/$SYS/map_lr$LR.log
    done
    make check SYSTEM=$SYS 2>&1 | tee out/$SYS/check.log
done
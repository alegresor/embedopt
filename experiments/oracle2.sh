#!/bin/bash
# Oracle second-moment experiment (the completed run): {base, moment2, map} x LR in {0.1, 0.5};
# 5 seeds for 8AHU_A, 10 for 8K9Z_A.  Outputs under out/<SYS>/.  Run from the repository root.
set -euo pipefail

declare -A SEEDS=([8AHU_A]=101,102,103,104,105 [8K9Z_A]=101,102,103,104,105,106,107,108,109,110)
LRS="0.1 0.5"

for SYS in 8AHU_A 8K9Z_A; do make cfg SYSTEM=$SYS; done
make maps

for SYS in 8AHU_A 8K9Z_A; do
    S=${SEEDS[$SYS]}
    mkdir -p out/$SYS
    make run SYSTEM=$SYS SEEDS=$S METHOD=base 2>&1 | tee out/$SYS/base.log
    for LR in $LRS; do
        make run SYSTEM=$SYS SEEDS=$S METHOD=moment ORDER=2 LR=$LR 2>&1 | tee out/$SYS/moment2_lr$LR.log
        make run SYSTEM=$SYS SEEDS=$S METHOD=map LR=$LR 2>&1 | tee out/$SYS/map_lr$LR.log
    done
    make check SYSTEM=$SYS 2>&1 | tee out/$SYS/check.log
done

#!/bin/bash
# Debye-Waller-weighted oracle moments (~50 min): moment2dw and moment3dw at LR 0.1 on both systems, then a full check
# and a focused comparison figure. Reuses base / map / moment2 / moment3 runs in out/<SYS>/. Run from the repo root.
set -euo pipefail

declare -A SEEDS=([8AHU_A]=101,102,103,104,105 [8K9Z_A]=101,102,103,104,105,106,107,108,109,110)

for SYS in 8AHU_A 8K9Z_A; do
    for ORDER in 3 2; do
        make run SYSTEM=$SYS SEEDS=${SEEDS[$SYS]} METHOD=moment ORDER=$ORDER DW=1 LR=0.1 2>&1 | tee out/$SYS/moment${ORDER}dw_lr0.1.log
    done
done
for SYS in 8AHU_A 8K9Z_A; do
    make check SYSTEM=$SYS 2>&1 | tee out/$SYS/check.log
    make check SYSTEM=$SYS FIG=check_dw RUNS="base map_lr0.5 moment2_lr0.1 moment3_lr0.1 moment2dw_lr0.1 moment3dw_lr0.1" \
        2>&1 | tee out/$SYS/check_dw.log
done

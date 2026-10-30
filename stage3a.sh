#!/bin/bash
# Full Stage 3a pipeline for both systems: configs, synthetic maps, guided / base / LR-sweep / map-guided runs, checks.
set -euo pipefail

SYSTEMS="8AHU_A 8K9Z_A"
LRS="0.5 1.0"

for SYS in $SYSTEMS; do make cfg SYSTEM=$SYS; done
make maps   # renders the 5 A synthetic maps for all systems (needed by stage_map only)

for SYS in $SYSTEMS; do
    mkdir -p out/$SYS
    make stage3a      SYSTEM=$SYS 2>&1 | tee out/$SYS/stage3a.log
    make stage3a_base SYSTEM=$SYS 2>&1 | tee out/$SYS/stage3a_base.log
    for LR in $LRS; do
        make stage3a_lr SYSTEM=$SYS LR=$LR 2>&1 | tee out/$SYS/stage3a_lr$LR.log
    done
    make stage_map    SYSTEM=$SYS 2>&1 | tee out/$SYS/stage_map.log
    make stage3acheck SYSTEM=$SYS 2>&1 | tee out/$SYS/stage3acheck.log
done
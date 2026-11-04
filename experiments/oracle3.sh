#!/bin/bash
# Oracle third-moment experiment (~1.5 h): moment3 (second-order loss x bispectrum loss) x LR in {0.1, 0.5}
# at 5 A on both systems, plus 4 A on 8K9Z_A. Reuses base / map / moment2 runs already in out/<SYS>/.
# Run from the repository root after experiments/oracle2.sh.
set -euo pipefail

declare -A SEEDS=([8AHU_A]=101,102,103,104,105 [8K9Z_A]=101,102,103,104,105,106,107,108,109,110)
LRS="0.1 0.5"

# Smoke test first (~2 min): checks the bispectrum code path on the GPU before committing the full run
make run SYSTEM=8AHU_A SEEDS=101 METHOD=moment ORDER=3 LR=0.1 N_STEP=10 2>&1 | tee out/8AHU_A/moment3_smoke.log
rm -rf out/8AHU_A/moment3_lr0.1

for SYS in 8AHU_A 8K9Z_A; do
    S=${SEEDS[$SYS]}
    for LR in $LRS; do
        make run SYSTEM=$SYS SEEDS=$S METHOD=moment ORDER=3 LR=$LR 2>&1 | tee out/$SYS/moment3_lr$LR.log
    done
done
for LR in $LRS; do  # higher resolution on the hard target
    make run SYSTEM=8K9Z_A SEEDS=${SEEDS[8K9Z_A]} METHOD=moment ORDER=3 LR=$LR RES=4.0 2>&1 | tee out/8K9Z_A/moment3_lr${LR}_res4.0.log
done
for SYS in 8AHU_A 8K9Z_A; do make check SYSTEM=$SYS 2>&1 | tee out/$SYS/check.log; done

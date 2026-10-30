
# 8AHU_A,8K9Z_A
make cfg SYSTEM=8K9Z_A
mkdir -p out/8K9Z_A
make stage3a      SYSTEM=8K9Z_A 2>&1 | tee out/8K9Z_A/stage3a.log
make stage3a_base SYSTEM=8K9Z_A 2>&1 | tee out/8K9Z_A/stage3a_base.log
make stage3acheck SYSTEM=8K9Z_A 2>&1 | tee out/8K9Z_A/stage3acheck.log

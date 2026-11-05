#!/bin/bash
# Three-pass sequential moment matching on 8K9Z_A (~25 min). Coarse-to-fine across passes:
#   pass 1: cold start, 100 steps, m12                          (existing: moment2_lr0.1)
#   pass 2: restart from pass 1 at 80 A (53 steps)              (existing: moment{2,3}_lr0.1_seq80)
#   pass 3: restart from pass 2 at 30 A (44 steps)              (new:      moment{2,3}_lr0.1_seq80_seq30)
# Two chains with identical restarts, differing only in the reward of passes 2 and 3:
#   third-order chain: m12 * m3        control chain: m12 only
# Run from the repo root after experiments/sequential.sh (needs the seq80 arms).
set -euo pipefail
SYS=8K9Z_A
SEEDS=101,102,103,104,105,106,107,108,109,110
NSEED=$(echo ${SEEDS//,/ } | wc -w)
count() { (ls out/$SYS/$1/*/seed_*/predictions/*_sample_0.cif 2>/dev/null || true) | wc -l; }

for ORDER in 2 3; do
    PREV=moment${ORDER}_lr0.1_seq80
    RUN=${PREV}_seq30
    [ "$(count $PREV)" -eq "$NSEED" ] || { echo "$PREV: expected $NSEED pass-2 structures, found $(count $PREV)"; exit 1; }
    if [ "$(count $RUN)" -eq "$NSEED" ]; then echo "skip $RUN (complete)"; continue; fi
    rm -rf out/$SYS/$RUN
    make run SYSTEM=$SYS METHOD=moment ORDER=$ORDER LR=0.1 SEEDS=$SEEDS SIGMA0=30 TAG=_seq80_seq30 \
        INIT="out/$SYS/$PREV/*/seed_{seed}/predictions/*_sample_0.cif" 2>&1 | tee out/$SYS/$RUN.log
done

make check SYSTEM=$SYS FIG=check_3pass RUNS="base map_lr0.5 moment2_lr0.1 \
    moment2_lr0.1_seq30 moment3_lr0.1_seq30 moment2_lr0.1_seq80 moment3_lr0.1_seq80 \
    moment2_lr0.1_seq80_seq30 moment3_lr0.1_seq80_seq30" 2>&1 | tee out/$SYS/check_3pass.log

# Per-seed core RMSD (A) along both chains, and the third-order effect at each pass
python - <<'PY' 2>&1 | tee -a out/8K9Z_A/check_3pass.log
import pandas as pd
d = pd.read_csv("out/8K9Z_A/check_3pass.csv")
p = d.pivot_table(index="seed", columns="run", values="core")
chains = {"control (m12)": ["moment2_lr0.1", "moment2_lr0.1_seq80", "moment2_lr0.1_seq80_seq30"],
          "third order (m12 m3)": ["moment2_lr0.1", "moment3_lr0.1_seq80", "moment3_lr0.1_seq80_seq30"]}
for name, c in chains.items():
    print(f"\n{name}: core RMSD per seed, pass 1 -> pass 2 (80 A) -> pass 3 (30 A)")
    t = p[c].copy(); t.columns = ["pass1", "pass2", "pass3"]
    print(t.round(2).to_string())
    for a, b in [("pass1", "pass2"), ("pass2", "pass3"), ("pass1", "pass3")]:
        dlt = t[b] - t[a]
        print(f"  {a} -> {b}: median {dlt.median():+.2f} A, improved in {(dlt < 0).sum()}/{len(t)}; "
              f"{b} median core {t[b].median():.2f} A, core<2 in {(t[b] < 2).sum()}/{len(t)}")
print("\nthird order vs control (negative = third order better):")
for lab, a, b in [("pass 2 (80 A)", "moment3_lr0.1_seq80", "moment2_lr0.1_seq80"),
                  ("pass 3 (80 -> 30 A)", "moment3_lr0.1_seq80_seq30", "moment2_lr0.1_seq80_seq30"),
                  ("3 passes vs 2 passes at 30 A", "moment3_lr0.1_seq80_seq30", "moment3_lr0.1_seq30")]:
    dlt = p[a] - p[b]
    print(f"  {lab:<30} median {dlt.median():+.2f} A, first better in {(dlt < 0).sum()}/{len(p)} seeds")
PY

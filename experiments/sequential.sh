#!/bin/bash
# Sequential (two-pass) moment matching on 8K9Z_A.
# Pass 1 = the existing moment2_lr0.1 runs (second-order guidance, cold start, ~6 A from x*).
# Pass 2 = partial diffusion from each seed's pass-1 structure, re-noised to SIGMA0, with the same seed:
#   moment2_lr0.1_seq<S> : second-order guidance again (control: re-noising + refitting alone)
#   moment3_lr0.1_seq<S> : second x third-order guidance (the test)
# Restart levels from $SIGMAS (default "10 30"), e.g.  SIGMAS="50 80" bash experiments/sequential.sh
# Arms that already have all predictions are skipped. The check covers every _seq arm present.
set -euo pipefail
SYS=8K9Z_A
SEEDS=101,102,103,104,105,106,107,108,109,110
SIGMAS=${SIGMAS:-"10 30"}
PREV=out/$SYS/moment2_lr0.1
INIT="$PREV/*/seed_{seed}/predictions/*_sample_0.cif"
NSEED=$(echo ${SEEDS//,/ } | wc -w)
for s in ${SEEDS//,/ }; do  # every seed needs its pass-1 structure
    n=$( (ls $PREV/*/seed_$s/predictions/*_sample_0.cif 2>/dev/null || true) | wc -l)
    [ "$n" -eq 1 ] || { echo "seed $s: expected 1 pass-1 structure under $PREV, found $n"; exit 1; }
done

for S in $SIGMAS; do
    for ORDER in 2 3; do
        RUN=moment${ORDER}_lr0.1_seq$S
        DONE=$( (ls out/$SYS/$RUN/*/seed_*/predictions/*_sample_0.cif 2>/dev/null || true) | wc -l)
        if [ "$DONE" -eq "$NSEED" ]; then echo "skip $RUN (complete)"; continue; fi
        rm -rf out/$SYS/$RUN
        make run SYSTEM=$SYS METHOD=moment ORDER=$ORDER LR=0.1 SEEDS=$SEEDS INIT="$INIT" SIGMA0=$S TAG=_seq$S \
            2>&1 | tee out/$SYS/$RUN.log
    done
done

SEQ=$( (ls -d out/$SYS/moment[23]_lr0.1_seq*/ 2>/dev/null || true) | xargs -r -n1 basename | grep -v "_seq[0-9]*_seq" | sort -t q -k2 -n)
make check SYSTEM=$SYS FIG=check_seq RUNS="base map_lr0.5 moment2_lr0.1 moment3_lr0.1 $SEQ" 2>&1 | tee out/$SYS/check_seq.log

# Paired per-seed comparison (core RMSD, A): each arm vs pass 1, and moment3 vs moment2 at the same restart level
python - <<'PY' 2>&1 | tee -a out/8K9Z_A/check_seq.log
import re
import pandas as pd
d = pd.read_csv("out/8K9Z_A/check_seq.csv")
p = d[d.run.str.startswith("moment")].pivot_table(index="seed", columns="run", values="core")
lv = sorted({int(m.group(1)) for c in p for m in [re.search(r"_seq(\d+)$", c)] if m})
cols = ["moment2_lr0.1"] + [f"moment{o}_lr0.1_seq{s}" for s in lv for o in (2, 3) if f"moment{o}_lr0.1_seq{s}" in p]
print("\nPaired core RMSD (A) per seed; pass 1 = moment2_lr0.1\n" + p[cols].round(2).to_string())
print("\nchange vs pass 1:")
for c in cols[1:]:
    print(f"  {c:<22} median {(p[c] - p[cols[0]]).median():+.2f} A, improved in {(p[c] < p[cols[0]]).sum()}/{len(p)} seeds, "
          f"median core {p[c].median():.2f} A, core<2 in {(p[c] < 2).sum()}/{len(p)}")
print("\nmoment3 vs moment2 at the same restart level (negative = third order better):")
for s in lv:
    a, b = f"moment3_lr0.1_seq{s}", f"moment2_lr0.1_seq{s}"
    if a in p and b in p:
        diff = p[a] - p[b]
        print(f"  seq{s:<3} median {diff.median():+.2f} A, moment3 better in {(diff < 0).sum()}/{len(p)} seeds")
PY

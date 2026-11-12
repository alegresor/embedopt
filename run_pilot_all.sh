#!/bin/bash
# Stage A (data, moments, checks), an automatic gate on the checks, then Stage B (RELION, docking, chains, check).
set -euo pipefail
export SYS=8K9Z_A SNR=0.0625
SIM=out/$SYS/noise_snr16
mkdir -p $SIM
echo "=== Stage A started $(date)"
[ -f $SIM/particles.star ] || python -m moments.simulate --system $SYS --snr $SNR --n 100000 --out $SIM 2>&1 | tee $SIM/simulate.log
[ -f $SIM/moments_N100000.npz ] || python -m moments.image_data stack --sim $SIM 2>&1 | tee $SIM/moments_stack.log
[ -f $SIM/moments_N1000000_stream.npz ] || python -m moments.image_data stream --sim $SIM --n 1000000 --seed 1 2>&1 | tee $SIM/moments_stream.log
[ -f $SIM/image_data_checks.json ] || python -m moments.image_data checks --sim $SIM 2>&1 | tee $SIM/image_data_checks.log
echo "=== Stage A finished $(date); gate:"
python - "$SIM/image_data_checks.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
scales = {k: v["scale"] for k, v in r.items() if isinstance(v, dict) and "scale" in v}
cov = r["noise_cov_mc_err"]["median"]
ok = all(0.98 < s < 1.02 for s in scales.values()) and cov < 0.05
print(f"  calibration scales {scales}, noise-covariance median rel. diff {cov:.3f} -> {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
PY
echo "=== Stage B started $(date)"
bash experiments/noise_pilot.sh
echo "=== all done $(date)"

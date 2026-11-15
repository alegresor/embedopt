#!/bin/bash
# RELION maps from a simulated stack (moments.simulate), run from the repo root:
#   bash experiments/relion_maps.sh out/8K9Z_A/noise_snr16
# 0. invert : contrast-inverted copy of the stack (RELION's convention; ASPIRE writes positive density)
#                                                                       -> particles_inv.star + particles_*_inv.mrcs
# 1. poses  : relion_reconstruct with the true poses and CTFs             -> relion/recon_poses.mrc (+ FSC vs truth)
# 2. full   : VDAM initial model (no poses), 3D auto-refine (split halves), mask, postprocess
#                                                                       -> relion/Refine3D/run_class001.mrc, PostProcess/
# Finished steps are skipped; an interrupted initial model or auto-refine continues from its last complete
# iteration. Env: NP (MPI ranks for auto-refine, default 3), J (threads, 4), GPU (ids, ""), ITER_INI (200).
set -euo pipefail
SIM=$1
NP=${NP:-3}; J=${J:-4}; GPU=${GPU:-""}; ITER_INI=${ITER_INI:-200}
read D NRES <<< $(python -c "
import json, biotite.structure.io as bio
m = json.load(open('$SIM/sim.json')); a = bio.load_structure('$SIM/xstar_mapframe.pdb')
print(int(min(0.9 * m['L'] * m['pixel'], 2.3 * m['particle_radius'])), int((a.atom_name == 'CA').sum()))")
echo "RELION on $SIM: particle diameter $D A, $NRES residues"

if [ ! -f $SIM/particles_inv.star ]; then
    python - "$SIM" <<'PY'
import glob, re, sys, mrcfile
d = sys.argv[1]
for f in sorted(glob.glob(f"{d}/particles_*.mrcs")):
    if f.endswith("_inv.mrcs"):
        continue
    with mrcfile.open(f) as m, mrcfile.new(f.replace(".mrcs", "_inv.mrcs"), overwrite=True) as o:
        o.set_data(-m.data); o.voxel_size = m.voxel_size
s = open(f"{d}/particles.star").read()
open(f"{d}/particles_inv.star.tmp", "w").write(re.sub(r"(particles_\d+_\d+)\.mrcs", r"\1_inv.mrcs", s))
PY
    mv $SIM/particles_inv.star.tmp $SIM/particles_inv.star
    echo "  wrote the contrast-inverted stack"
fi
mkdir -p $SIM/relion/InitialModel $SIM/relion/Refine3D $SIM/relion/PostProcess
cd $SIM
latest() { ls $1 2>/dev/null | sort | tail -1; }

if [ ! -f relion/recon_poses.mrc ]; then
    relion_reconstruct --i particles_inv.star --o relion/recon_poses.mrc --ctf 2>&1 | tee relion/recon_poses.log
fi

FINAL_INI=relion/InitialModel/run_it$(printf %03d $ITER_INI)_class001.mrc
if [ ! -f $FINAL_INI ] && [ ! -f relion/Refine3D/run_class001.mrc ]; then
    OPT=$(latest "relion/InitialModel/run_it*_optimiser.star")
    if [ -n "$OPT" ]; then
        echo "  continuing the initial model from $OPT"
        relion_refine --continue $OPT --o relion/InitialModel/run --iter $ITER_INI --j $J --gpu "$GPU" \
            2>&1 | tee -a relion/InitialModel/run.log
    else
        relion_refine --o relion/InitialModel/run --iter $ITER_INI --grad --denovo_3dref --i particles_inv.star --ctf \
            --K 1 --sym C1 --flatten_solvent --zero_mask --dont_combine_weights_via_disc --pool 30 --pad 1 \
            --particle_diameter $D --oversampling 1 --healpix_order 1 --offset_range 6 --offset_step 2 --auto_sampling \
            --tau2_fudge 4 --j $J --gpu "$GPU" 2>&1 | tee relion/InitialModel/run.log
    fi
fi
INI=$(latest "relion/InitialModel/run_it*_class001.mrc")

if [ ! -f relion/Refine3D/run_class001.mrc ]; then
    OPT=$(latest "relion/Refine3D/run_it*_optimiser.star")
    if [ -n "$OPT" ]; then
        echo "  continuing auto-refine from $OPT"
        mpirun -n $NP relion_refine_mpi --continue $OPT --o relion/Refine3D/run --dont_combine_weights_via_disc \
            --pool 30 --pad 2 --j $J --gpu "$GPU" 2>&1 | tee -a relion/Refine3D/run.log
    else
        mpirun -n $NP relion_refine_mpi --o relion/Refine3D/run --auto_refine --split_random_halves \
            --i particles_inv.star --ref $INI --ini_high 30 --dont_combine_weights_via_disc --pool 30 --pad 2 --ctf \
            --particle_diameter $D --flatten_solvent --zero_mask --oversampling 1 --healpix_order 2 \
            --auto_local_healpix_order 4 --offset_range 5 --offset_step 2 --sym C1 --low_resol_join_halves 40 \
            --norm --scale --j $J --gpu "$GPU" 2>&1 | tee relion/Refine3D/run.log
    fi
fi
test -f relion/Refine3D/run_class001.mrc

if [ ! -f relion/PostProcess/postprocess.star ]; then
    cd - > /dev/null
    T=$(python -m moments.dock threshold --map $SIM/relion/Refine3D/run_class001.mrc --nres $NRES --factor 1.0)
    cd $SIM
    relion_mask_create --i relion/Refine3D/run_class001.mrc --o relion/PostProcess/mask.mrc --lowpass 15 \
        --ini_threshold $T --extend_inimask 3 --width_soft_edge 6 --j $J 2>&1 | tee relion/PostProcess/mask.log
    relion_postprocess --mask relion/PostProcess/mask.mrc --i relion/Refine3D/run_half1_class001_unfil.mrc \
        --o relion/PostProcess/postprocess --auto_bfac --autob_lowres 10 2>&1 | tee relion/PostProcess/postprocess.log
fi
cd - > /dev/null
python -m moments.dock fsc --map $SIM/relion/recon_poses.mrc --ref $SIM/truth.mrc --out $SIM/relion/fsc_poses_vs_truth.json
grep -h "_rlnFinalResolution" $SIM/relion/PostProcess/postprocess.star 2>/dev/null | head -1 || true

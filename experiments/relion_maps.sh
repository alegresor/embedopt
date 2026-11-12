#!/bin/bash
# RELION maps from a simulated stack (moments.simulate), run from the repo root:
#   bash experiments/relion_maps.sh out/8K9Z_A/noise_snr16
# 1. poses : relion_reconstruct with the true poses and CTFs           -> relion/recon_poses.mrc (+ FSC vs truth.mrc)
# 2. full  : VDAM initial model (no poses), 3D auto-refine (split halves), mask, postprocess
#                                                                      -> relion/Refine3D/run_class001.mrc, PostProcess/
# Steps whose outputs exist are skipped. Env: NP (MPI ranks for auto-refine, default 3), J (threads, 4), GPU (ids, "").
set -euo pipefail
SIM=$1
NP=${NP:-3}; J=${J:-4}; GPU=${GPU:-""}
read D NRES <<< $(python -c "
import json, biotite.structure.io as bio
m = json.load(open('$SIM/sim.json')); a = bio.load_structure('$SIM/xstar_mapframe.pdb')
print(int(min(0.9 * m['L'] * m['pixel'], 2.3 * m['particle_radius'])), int((a.atom_name == 'CA').sum()))")
echo "RELION on $SIM: particle diameter $D A, $NRES residues"
mkdir -p $SIM/relion/InitialModel $SIM/relion/Refine3D $SIM/relion/PostProcess
cd $SIM

if [ ! -f relion/recon_poses.mrc ]; then
    relion_reconstruct --i particles_inv.star --o relion/recon_poses.mrc --ctf --j $J 2>&1 | tee relion/recon_poses.log
fi

if ! ls relion/InitialModel/run_it*_class001.mrc > /dev/null 2>&1; then
    relion_refine --o relion/InitialModel/run --iter 200 --grad --denovo_3dref --i particles_inv.star --ctf --K 1 --sym C1 \
        --flatten_solvent --zero_mask --dont_combine_weights_via_disc --pool 30 --pad 1 --particle_diameter $D \
        --oversampling 1 --healpix_order 1 --offset_range 6 --offset_step 2 --auto_sampling --tau2_fudge 4 \
        --j $J --gpu "$GPU" 2>&1 | tee relion/InitialModel/run.log
fi
INI=$(ls relion/InitialModel/run_it*_class001.mrc | sort | tail -1)

if [ ! -f relion/Refine3D/run_class001.mrc ]; then
    mpirun -n $NP relion_refine_mpi --o relion/Refine3D/run --auto_refine --split_random_halves --i particles_inv.star \
        --ref $INI --ini_high 30 --dont_combine_weights_via_disc --pool 30 --pad 2 --ctf --particle_diameter $D \
        --flatten_solvent --zero_mask --oversampling 1 --healpix_order 2 --auto_local_healpix_order 4 \
        --offset_range 5 --offset_step 2 --sym C1 --low_resol_join_halves 40 --norm --scale \
        --j $J --gpu "$GPU" 2>&1 | tee relion/Refine3D/run.log
fi

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
grep -h "FINAL RESOLUTION\|_rlnFinalResolution" $SIM/relion/PostProcess/postprocess.log $SIM/relion/PostProcess/postprocess.star 2>/dev/null | head -2 || true

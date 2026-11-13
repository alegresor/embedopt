#!/bin/bash
# Pilot: noisy, CTF-modulated images of x* (one system, one SNR) feeding both the moment reward and the map reward.
#   SYS=8K9Z_A SNR=0.0625 bash experiments/noise_pilot.sh            (run from the repo root; finished steps are skipped)
# Steps
#   1. simulate     moments.simulate: 1e5 particles (192 px, 1.5 A, CTF 0.8-2.0 um, white noise)  -> $SIM/particles.star
#   2. moments      moments.image_data: stack (N = 1e5) and streamed (N = 1e6) estimates, then the checks
#   3. RELION       experiments/relion_maps.sh: known-pose reconstruction, and VDAM + auto-refine + postprocess
#   4. docking      moments.dock: prior (base seed 101) docked into each map (both hands), map cleaned; self-test with x*
#   5. chains       4-pass chains (cold pass + restarts at 80/30/15 A), 10 seeds:
#                     map_relpose  : map reward on the known-pose RELION map, frame from the docked prior   (LR 0.5)
#                     map_relfull  : map reward on the ab-initio RELION map, frame from the docked prior    (LR 0.5)
#                     moment*_imgd_<tag>n1e5 / n1e6 : w12 + wblock moments from the same images         (LR 0.1)
#                   compared with base, the oracle map chain (map_lr0.5*) and the exact-moment chain (imgx w12 wblock)
#   6. check + summary against the N = 1e5 moments           -> out/$SYS/check_pilot_<tag>.{log,csv,png}
set -euo pipefail
SYS=${SYS:-8K9Z_A}
SNR=${SNR:-0.0625}
N=${N:-100000}
NSTREAM=${NSTREAM:-1000000}
SEEDS=${SEEDS:-101,102,103,104,105,106,107,108,109,110}
TAG=$(python -c "print('snr%d' % round(1 / $SNR) if $SNR > 0 else 'clean')")
SIM=out/$SYS/noise_$TAG
LEVELS="80 30 15"
mkdir -p $SIM

count() { (ls out/$SYS/$1/*/seed_*/predictions/*_sample_0.cif 2>/dev/null || true) | wc -l; }
NS=$(echo ${SEEDS//,/ } | wc -w)
arm() {  # arm RUN make-args...
    local run=$1; shift
    if [ "$(count $run)" -eq "$NS" ]; then echo "skip $SYS/$run (complete)"; return; fi
    rm -rf out/$SYS/$run
    make run SYSTEM=$SYS SEEDS=$SEEDS "$@" 2>&1 | tee out/$SYS/$run.log
    [ "$(count $run)" -eq "$NS" ] || { echo "$SYS/$run: only $(count $run)/$NS predictions; see out/$SYS/$run/*/ERR"; exit 1; }
}
chain() {  # chain PREFIX(run name before _seq) make-args... : pass 1 + 3 restarts with the same reward
    local p1=$1; shift
    arm $p1 "$@"
    local prev=$p1 tag=""
    for S in $LEVELS; do
        tag="${tag}_seq$S"
        arm $p1$tag "$@" SIGMA0=$S TAG=$tag INIT="out/$SYS/$prev/*/seed_{seed}/predictions/*_sample_0.cif"
        prev=$p1$tag
    done
}

# 1-2. data and moments
[ -f $SIM/particles.star ] || python -m moments.simulate --system $SYS --snr $SNR --n $N --out $SIM 2>&1 | tee $SIM/simulate.log
[ -f $SIM/moments_N$N.npz ] || python -m moments.image_data stack --sim $SIM 2>&1 | tee $SIM/moments_stack.log
[ -f $SIM/moments_N${NSTREAM}_stream.npz ] || python -m moments.image_data stream --sim $SIM --n $NSTREAM --seed 1 \
    2>&1 | tee $SIM/moments_stream.log
[ -f $SIM/image_data_checks.json ] || python -m moments.image_data checks --sim $SIM 2>&1 | tee $SIM/image_data_checks.log

# 3. RELION
bash experiments/relion_maps.sh $SIM 2>&1 | tee $SIM/relion.log

# 4. docking (self-test with x* into the true map, then the prior into each RELION map)
PRIOR=$(ls out/$SYS/base/*/seed_101/predictions/*_sample_0.cif | head -1)
[ -f $SIM/dock_selftest/dock.json ] || python -m moments.dock dock --map $SIM/truth.mrc \
    --model examples/synthetic_map_benchmark/pdb_clean/$SYS.pdb --truth $SIM/xstar_mapframe.pdb --out $SIM/dock_selftest \
    --no-clean 2>&1 | tee $SIM/dock_selftest.log
[ -f $SIM/dock_poses/dock.json ] || python -m moments.dock dock --map $SIM/relion/recon_poses.mrc --model $PRIOR \
    --out $SIM/dock_poses 2>&1 | tee $SIM/dock_poses.log
[ -f $SIM/dock_full/dock.json ] || python -m moments.dock dock --map $SIM/relion/Refine3D/run_class001.mrc --model $PRIOR \
    --out $SIM/dock_full 2>&1 | tee $SIM/dock_full.log

# 5. chains
chain map_relpose_lr0.5 METHOD=map LR=0.5 MAP=$SIM/dock_poses/map_crop.mrc MAPREF=$SIM/dock_poses/docked.pdb MTAG=_relpose
chain map_relfull_lr0.5 METHOD=map LR=0.5 MAP=$SIM/dock_full/map_crop.mrc MAPREF=$SIM/dock_full/docked.pdb MTAG=_relfull
for NN in $N ${NSTREAM}_stream; do
    DT=_${TAG}n$(python -c "n=int('$NN'.split('_')[0]); print('1e%d' % len(str(n)[1:]))")
    D=$SIM/moments_N$NN.npz
    arm moment2_imgd${DT}_w12_lr0.1 METHOD=moment ORDER=2 LR=0.1 SPACE=image DATA=$D DTAG=$DT M12=white
    prev=moment2_imgd${DT}_w12_lr0.1 t=""
    for S in $LEVELS; do
        t="${t}_seq$S"
        arm moment3_imgd${DT}_w12_m3wblock_lr0.1$t METHOD=moment ORDER=3 LR=0.1 SPACE=image DATA=$D DTAG=$DT M12=white \
            M3=wblock SIGMA0=$S TAG=$t INIT="out/$SYS/$prev/*/seed_{seed}/predictions/*_sample_0.cif"
        prev=moment3_imgd${DT}_w12_m3wblock_lr0.1$t
    done
done

# 6. check + summary
F=_seq80_seq30_seq15
RUNS="base map_lr0.5 map_lr0.5$F moment2_imgx_w12_lr0.1 moment3_imgx_w12_m3wblock_lr0.1$F"
for P in map_relpose_lr0.5 map_relfull_lr0.5; do RUNS="$RUNS $P $P$F"; done
for DT in _${TAG}n1e5 _${TAG}n1e6; do RUNS="$RUNS moment2_imgd${DT}_w12_lr0.1 moment3_imgd${DT}_w12_m3wblock_lr0.1$F"; done
make check SYSTEM=$SYS FIG=check_pilot_$TAG CHECK_DATA=$SIM/moments_N$N.npz RUNS="$RUNS" 2>&1 | tee out/$SYS/check_pilot_$TAG.log
SYS=$SYS TAG=$TAG python -m moments.pilot_summary 2>&1 | tee -a out/$SYS/check_pilot_$TAG.log

#!/bin/bash
# Full non-oracle pipeline, per system: noisy CTF images -> moments -> moment chain -> RELION map -> docking of the
# m3-selected moment model -> one map pass with the map-based frame. Run from the repo root (preferably detached):
#   setsid nohup bash experiments/full_pipeline.sh > out/full_pipeline.log 2>&1 < /dev/null &
# Env: SYSTEMS (default "8H1I_A 8AHU_A 8K9Z_A"; 8K9Z_A is already done and only gets the final check), SNR (0.0625),
#      N (images, 1e5), RETRIES (attempts per GPU step, 3), KEEP_STACK (1: keep the non-inverted stack, default 1).
# Every step is skipped when its output exists, so the script can simply be restarted after an interruption.
# Steps per system (SIM = out/<SYS>/noise_snr<1/SNR>):
#   1. simulate   1e5 particles, 192 px, 1.5 A, CTF 0.8-2.0 um, white noise          (moments.simulate)
#   2. moments    CTF-corrected, debiased moments of the stack -> SIM/moments_N<N>.npz, and the checks
#   3. RELION     known-pose map, and VDAM + auto-refine + postprocess (contrast-inverted copy)  (relion_maps.sh)
#   4. chain      moment2 w12 (cold) + 3 restarts moment3 w12 wblock (80/30/15 A), image-space data, LR 0.1
#   5. select     lowest m3_img seed of the final pass (data only)
#   6. dock       selected model into both RELION maps (both hands), crop                   (moments.dock)
#   7. map pass   restart from every seed's moment result at 30 A, map reward on each RELION map, MAP_FRAME=fit
#   8. check      all arms against the image data -> out/<SYS>/check_pipeline_<tag>.{log,csv,png}
# then moments.pipeline_summary over all systems -> out/pipeline_summary.log
set -uo pipefail
SYSTEMS=${SYSTEMS:-"8H1I_A 8AHU_A 8K9Z_A"}
SNR=${SNR:-0.0625}
N=${N:-100000}
RETRIES=${RETRIES:-3}
KEEP_STACK=${KEEP_STACK:-1}
TAG=$(python -c "print('snr%d' % round(1 / $SNR))")
NT=$(python -c "n=int($N); print('1e%d' % len(str(n)[1:]))")
LEVELS="80 30 15"
F=_seq80_seq30_seq15
declare -A SEEDS=([8AHU_A]=101,102,103,104,105 [8K9Z_A]=101,102,103,104,105,106,107,108,109,110
                  [8H1I_A]=101,102,103,104,105,106,107,108,109,110)

log() { echo "=== $(date '+%F %T') $*"; }
retry() {  # retry CMD...: up to RETRIES attempts (GPU steps can be killed by the display watchdog)
    local k
    for k in $(seq 1 $RETRIES); do
        "$@" && return 0
        log "attempt $k/$RETRIES failed: $*"; sleep 30
    done
    return 1
}
count() { (ls out/$1/$2/*/seed_*/predictions/*_sample_0.cif 2>/dev/null || true) | wc -l; }
arm() {  # arm SYS RUN make-args...: run unless complete, with retries
    local sys=$1 run=$2 n; shift 2
    n=$(echo ${SEEDS[$sys]//,/ } | wc -w)
    [ "$(count $sys $run)" -eq "$n" ] && { echo "skip $sys/$run (complete)"; return 0; }
    local k
    for k in $(seq 1 $RETRIES); do
        rm -rf out/$sys/$run
        make run SYSTEM=$sys SEEDS=${SEEDS[$sys]} "$@" > out/$sys/$run.log 2>&1
        [ "$(count $sys $run)" -eq "$n" ] && return 0
        log "$sys/$run: $(count $sys $run)/$n predictions (attempt $k/$RETRIES); see out/$sys/$run.log"; sleep 30
    done
    return 1
}

system() {
    local SYS=$1 SIM=out/$1/noise_$TAG
    local MOM2=moment2_imgd_${TAG}n${NT}_w12_lr0.1 MOM3=moment3_imgd_${TAG}n${NT}_w12_m3wblock_lr0.1
    local DATA=$SIM/moments_N$N.npz
    mkdir -p $SIM
    log "$SYS: 1-2 data"
    [ -f $SIM/particles.star ] || python -m moments.simulate --system $SYS --snr $SNR --n $N --out $SIM \
        > $SIM/simulate.log 2>&1 || return 1
    [ -f $DATA ] || python -m moments.image_data stack --sim $SIM > $SIM/moments_stack.log 2>&1 || return 1
    [ -f $SIM/image_data_checks.json ] || python -m moments.image_data checks --sim $SIM \
        > $SIM/image_data_checks.log 2>&1 || return 1
    grep -v INFO $SIM/image_data_checks.log | tail -8

    log "$SYS: 3 RELION"
    retry bash experiments/relion_maps.sh $SIM > $SIM/relion.log 2>&1 || return 1
    tail -2 $SIM/relion.log
    [ "$KEEP_STACK" = 1 ] || rm -f $(ls $SIM/particles_*.mrcs | grep -v _inv.mrcs)

    log "$SYS: 4 moment chain"
    arm $SYS $MOM2 METHOD=moment ORDER=2 LR=0.1 SPACE=image DATA=$DATA DTAG=_${TAG}n$NT M12=white || return 1
    local prev=$MOM2 t=""
    for S in $LEVELS; do
        t="${t}_seq$S"
        arm $SYS $MOM3$t METHOD=moment ORDER=3 LR=0.1 SPACE=image DATA=$DATA DTAG=_${TAG}n$NT M12=white M3=wblock \
            SIGMA0=$S TAG=$t INIT="out/$SYS/$prev/*/seed_{seed}/predictions/*_sample_0.cif" || return 1
        prev=$MOM3$t
    done

    log "$SYS: 5 select"
    [ -f out/$SYS/check_momsel_$TAG.csv ] || make check SYSTEM=$SYS FIG=check_momsel_$TAG CHECK_DATA=$DATA \
        RUNS="$MOM3$F" > out/$SYS/check_momsel_$TAG.log 2>&1 || return 1
    local SEED MODEL
    SEED=$(python -c "import pandas as pd; d=pd.read_csv('out/$SYS/check_momsel_$TAG.csv'); print(d.loc[d.m3_img.idxmin(),'seed'])")
    MODEL=$(ls out/$SYS/$MOM3$F/*/$SEED/predictions/*_sample_0.cif)
    echo "  selected $SEED: $MODEL"

    log "$SYS: 6 dock"
    [ -f $SIM/dock_poses_m/dock.json ] || retry python -m moments.dock dock --map $SIM/relion/recon_poses.mrc \
        --model $MODEL --out $SIM/dock_poses_m > $SIM/dock_poses_m.log 2>&1 || return 1
    [ -f $SIM/dock_full_m/dock.json ] || retry python -m moments.dock dock --map $SIM/relion/Refine3D/run_class001.mrc \
        --model $MODEL --out $SIM/dock_full_m > $SIM/dock_full_m.log 2>&1 || return 1
    grep -h "correlation\|hand " $SIM/dock_poses_m.log $SIM/dock_full_m.log 2>/dev/null | tail -6

    log "$SYS: 7 map pass from the moment results"
    local A T2 D ok=0
    export MAP_FRAME=fit   # rigid-body frame refined against the map (moments/map_entry.py)
    for A in relpose_m3:dock_poses_m relfull_m3:dock_full_m; do
        IFS=: read T2 D <<< "$A"
        arm $SYS map_${T2}_lr0.5_from_moments METHOD=map LR=0.5 MAP=$SIM/$D/map_crop.mrc \
            MAPREF=$SIM/$D/docked.pdb MTAG=_$T2 SIGMA0=30 TAG=_from_moments \
            INIT="out/$SYS/$MOM3$F/*/seed_{seed}/predictions/*_sample_0.cif" || ok=1
    done
    unset MAP_FRAME
    [ $ok = 0 ] || return 1

    log "$SYS: 8 check"
    local RUNS="base map_lr0.5 map_lr0.5$F moment3_lr0.1$F moment3_imgx_w12_m3wblock_lr0.1$F $MOM2 $MOM3$F"
    RUNS="$RUNS map_relpose_m3_lr0.5_from_moments map_relfull_m3_lr0.5_from_moments"
    make check SYSTEM=$SYS FIG=check_pipeline_$TAG CHECK_DATA=$DATA RUNS="$RUNS" > out/$SYS/check_pipeline_$TAG.log 2>&1 \
        || return 1
    sed -n '/^run /,/^Rows/p' out/$SYS/check_pipeline_$TAG.log
}

log "full pipeline: systems $SYSTEMS, SNR $SNR, N $N"
for SYS in $SYSTEMS; do
    if system $SYS; then log "$SYS done"; else log "$SYS FAILED (continuing with the next system)"; fi
done
SYSTEMS="$SYSTEMS" TAG=$TAG python -m moments.pipeline_summary 2>&1 | tee out/pipeline_summary.log
log "all done"

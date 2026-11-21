#!/bin/bash
# Viewing-density follow-ups with the frame fix (about 11-12 h): the density is expressed in a frame of the molecule,
# so x0hat is rotated into that frame before its moments are computed (oracle: x*; fixed theta: the structure it was
# fitted from; fit: the previous fit's structure). The earlier oracle / fit / frozen arms lacked this.
# All chains: 10 seeds, eps = 0.1 unless noted, image data with non-uniform viewing directions.
# A. 8K9Z_A: oracle and fitted-every-10-calls chains again, with the frame fix (oracle2, fit2).
# B. 8K9Z_A, alternation (EM-like): theta from the uniform-assumed chain's lowest-m3 final structure, passes 2-4 with
#    that theta fixed in that structure's frame (from the uniform pass 1); a second round from round 1's best structure.
# C. 8K9Z_A, the m12 balance: passes 2-4 of the oracle2 chain with eps = 0.
# D. 8H1I_A: non-uniform data (15 min), uniform-assumed chain, one alternation round, oracle chain.
#   setsid nohup bash experiments/pose_weekend.sh > out/pose_weekend.log 2>&1 < /dev/null &
set -uo pipefail
F=_seq80_seq30_seq15; EPS=1e-1
declare -A SEEDS=([8K9Z_A]=101,102,103,104,105,106,107,108,109,110 [8H1I_A]=101,102,103,104,105,106,107,108,109,110)
log() { echo "=== $(date '+%F %T') $*"; }
count() { (ls out/$1/$2/*/seed_*/predictions/*_sample_0.cif 2>/dev/null || true) | wc -l; }
arm() {  # arm SYS RUN make-args...
    local sys=$1 run=$2 n k; shift 2
    n=$(echo ${SEEDS[$sys]//,/ } | wc -w)
    [ "$(count $sys $run)" -ge "$n" ] && { echo "skip $sys/$run"; return 0; }
    for k in 1 2 3; do
        rm -rf out/$sys/$run
        make run SYSTEM=$sys SEEDS=${SEEDS[$sys]} "$@" > out/$sys/$run.log 2>&1
        [ "$(count $sys $run)" -ge "$n" ] && return 0
        log "$sys/$run: $(count $sys $run)/$n (attempt $k)"; sleep 30
    done
    return 1
}
data() { echo "out/$1/noise_snr16/moments_N100000_poses.npz"; }
restarts() {  # restarts SYS PASS1 DTAG make-args...
    local sys=$1 prev=$2 DT=$3 t="" M3=moment3_imgd${3}_w12_m3wblock_lr0.1; shift 3
    for S in 80 30 15; do
        t="${t}_seq$S"
        arm $sys $M3$t METHOD=moment LR=0.1 SPACE=image DATA=$(data $sys) M12=white ORDER=3 M3=wblock DTAG=$DT \
            SIGMA0=$S TAG=$t INIT="out/$sys/$prev/*/seed_{seed}/predictions/*_sample_0.cif" "$@" || return 1
        prev=$M3$t
    done
}
chain() {  # chain SYS DTAG make-args...
    local sys=$1 DT=$2 M2=moment2_imgd${2}_w12_lr0.1; shift 2
    arm $sys $M2 METHOD=moment LR=0.1 SPACE=image DATA=$(data $sys) M12=white ORDER=2 DTAG=$DT "$@" || return 1
    restarts $sys $M2 $DT M12_EPS=$EPS "$@"
}
best() {  # best SYS RUN -> structure file of the lowest-m3_img seed of RUN (scored against the non-uniform data)
    local sys=$1 run=$2 seed
    [ -f out/$sys/sel_$run.csv ] || make check SYSTEM=$sys FIG=sel_$run CHECK_DATA=$(data $sys) RUNS="$run" \
        > out/$sys/sel_$run.log 2>&1
    seed=$(python -c "import pandas as pd; d=pd.read_csv('out/$sys/sel_$run.csv'); print(d.loc[d.m3_img.idxmin(),'seed'])")
    ls out/$sys/$run/*/$seed/predictions/*_sample_0.cif
}
em_round() {  # em_round SYS FROM_FINAL_RUN PASS1 DTAG THETA_JSON
    local sys=$1 src=$2 p1=$3 DT=$4 th=$5 m
    m=$(best $sys $src) || return 1
    python -m moments.pose_floors --sim out/$sys/noise_snr16 --model $m --out $th 2>&1 | grep -v INFO | tail -1 \
        | tee -a out/pose_weekend_thetas.log
    [ -f $th ] || return 1
    restarts $sys $p1 $DT M12_EPS=$EPS POSE=fixed POSE_P=4 POSE_THETA=$th POSE_REF=$m
}

log "A. 8K9Z_A oracle chain with the frame fix";  chain 8K9Z_A _posesoracle2 POSE=oracle || log "A oracle FAILED"
log "A. 8K9Z_A fitted chain with the frame fix";  chain 8K9Z_A _posesfit2 POSE=fit POSE_P=4 POSE_EVERY=10 || log "A fit FAILED"

log "B. 8K9Z_A alternation, round 1"
em_round 8K9Z_A moment3_imgd_posesuniform_w12_m3wblock_lr0.1$F moment2_imgd_posesuniform_w12_lr0.1 _posesem1 \
    out/8K9Z_A/theta_em1.json || log "round 1 FAILED"
log "B. 8K9Z_A alternation, round 2"
em_round 8K9Z_A moment3_imgd_posesem1_w12_m3wblock_lr0.1$F moment2_imgd_posesuniform_w12_lr0.1 _posesem2 \
    out/8K9Z_A/theta_em2.json || log "round 2 FAILED"

log "C. 8K9Z_A oracle passes 2-4 with eps = 0"
restarts 8K9Z_A moment2_imgd_posesoracle2_w12_lr0.1 _posesoracleeps0 POSE=oracle M12_EPS=0 || log "C FAILED"

log "D. 8H1I_A non-uniform data"
[ -f $(data 8H1I_A) ] || python -m moments.poses stream --sim out/8H1I_A/noise_snr16 --n 100000 \
    > out/8H1I_A/poses_stream.log 2>&1 || log "8H1I_A stream FAILED"
log "D. 8H1I_A uniform-assumed chain";  chain 8H1I_A _posesuniform POSE=uniform || log "D uniform FAILED"
log "D. 8H1I_A alternation, round 1"
em_round 8H1I_A moment3_imgd_posesuniform_w12_m3wblock_lr0.1$F moment2_imgd_posesuniform_w12_lr0.1 _posesem1 \
    out/8H1I_A/theta_em1.json || log "D round 1 FAILED"
log "D. 8H1I_A oracle chain";  chain 8H1I_A _posesoracle POSE=oracle || log "D oracle FAILED"

log "checks and summary"
R8="moment3_imgd_snr16n1e5eps1e-1_w12_m3wblock_lr0.1$F"
for a in oracle oracle2 uniform fit fit2 frozen em1 em2 oracleeps0; do R8="$R8 moment3_imgd_poses${a}_w12_m3wblock_lr0.1$F"; done
make check SYSTEM=8K9Z_A FIG=check_poses_weekend CHECK_DATA=$(data 8K9Z_A) RUNS="$R8" > out/8K9Z_A/check_poses_weekend.log 2>&1
R1="moment3_imgd_snr16n1e5eps1e-1_w12_m3wblock_lr0.1$F"
for a in uniform em1 oracle; do R1="$R1 moment3_imgd_poses${a}_w12_m3wblock_lr0.1$F"; done
make check SYSTEM=8H1I_A FIG=check_poses_weekend CHECK_DATA=$(data 8H1I_A) RUNS="$R1" > out/8H1I_A/check_poses_weekend.log 2>&1
python - <<'PY' 2>&1 | tee out/pose_weekend_summary.log
import os, pandas as pd
F = "_seq80_seq30_seq15"
run = lambda a: f"moment3_imgd_poses{a}_w12_m3wblock_lr0.1{F}"
ref = f"moment3_imgd_snr16n1e5eps1e-1_w12_m3wblock_lr0.1{F}"
arms = {"8K9Z_A": [("uniform poses, uniform model (reference)", ref), ("oracle (frame fix)", run("oracle2")),
                   ("oracle (frame fix), eps = 0", run("oracleeps0")), ("uniform assumed", run("uniform")),
                   ("fitted every 10 calls (frame fix)", run("fit2")), ("alternation round 1", run("em1")),
                   ("alternation round 2", run("em2")), ("old oracle (no frame fix)", run("oracle")),
                   ("old fitted (no frame fix)", run("fit"))],
        "8H1I_A": [("uniform poses, uniform model (reference)", ref), ("oracle", run("oracle")),
                   ("uniform assumed", run("uniform")), ("alternation round 1", run("em1"))]}
for s, lst in arms.items():
    f = f"out/{s}/check_poses_weekend.csv"
    if not os.path.isfile(f): print(f"\n{s}: no {f}"); continue
    d = pd.read_csv(f); core = d.pivot_table(index="seed", columns="run", values="core")
    o = run("oracle2") if s == "8K9Z_A" else run("oracle")
    print(f"\n=== {s}, final pass, 10 seeds\n{'arm':<44}{'median':>8}{'full':>7}{'core<2':>8}{'vs oracle':>20}{'sel.':>7}")
    for lab, r in lst:
        x = d[d.run == r]
        if not len(x): print(f"{lab:<44}   (missing)"); continue
        p = (core[r] - core[o]).dropna() if (o in core and r != o) else None
        ps = f"{p.median():+.2f} A, {int((p < 0).sum())}/{len(p)}" if p is not None else ""
        print(f"{lab:<44}{x.core.median():>8.2f}{x.rmsd.median():>7.2f}{(x.core < 2).mean():>8.0%}{ps:>20}"
              f"{x.loc[x.m3_img.idxmin(), 'core']:>7.2f}")
if os.path.isfile("out/pose_weekend_thetas.log"):
    print("\nfitted densities (alternation):"); print(open("out/pose_weekend_thetas.log").read())
PY
log "all done"

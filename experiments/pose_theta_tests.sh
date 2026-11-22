#!/bin/bash
# Viewing-density accuracy (about 9-10 h). 1. Sweep of fitting variants (moments.pose_sweep: number of structures k,
# loss terms, ridge, degree P), density error in x*'s frame, from the uniform-assumed chains of 8K9Z_A and 8H1I_A and
# from 8K9Z_A's alternation round 2. 2. The variant with the lowest mean error over the two systems (V), and the best
# single-structure variant (V1) for comparison, each as an alternation round (passes 2-4 from the uniform pass 1 with
# theta fixed in the best structure's frame); then a second round of V from its own best structure.
# (Choosing V by the error against the true density uses x*: a hyperparameter study, not a blind pipeline.)
#   setsid nohup bash experiments/pose_theta_tests.sh > out/pose_theta_tests.log 2>&1 < /dev/null &
set -uo pipefail
F=_seq80_seq30_seq15; EPS=1e-1
declare -A SEEDS=([8K9Z_A]=101,102,103,104,105,106,107,108,109,110 [8H1I_A]=101,102,103,104,105,106,107,108,109,110)
log() { echo "=== $(date '+%F %T') $*"; }
count() { (ls out/$1/$2/*/seed_*/predictions/*_sample_0.cif 2>/dev/null || true) | wc -l; }
data() { echo "out/$1/noise_snr16/moments_N100000_poses.npz"; }
arm() {
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
restarts() {  # restarts SYS PASS1 DTAG make-args...
    local sys=$1 prev=$2 DT=$3 t="" M3=moment3_imgd${3}_w12_m3wblock_lr0.1; shift 3
    for S in 80 30 15; do
        t="${t}_seq$S"
        arm $sys $M3$t METHOD=moment LR=0.1 SPACE=image DATA=$(data $sys) M12=white ORDER=3 M3=wblock DTAG=$DT \
            SIGMA0=$S TAG=$t M12_EPS=$EPS INIT="out/$sys/$prev/*/seed_{seed}/predictions/*_sample_0.cif" "$@" || return 1
        prev=$M3$t
    done
}
sel() {  # sel SYS RUN: make sure out/SYS/sel_RUN.csv exists (moments.check against the non-uniform data)
    [ -f out/$1/sel_$2.csv ] || make check SYSTEM=$1 FIG=sel_$2 CHECK_DATA=$(data $1) RUNS="$2" > out/$1/sel_$2.log 2>&1
}
sweep() {  # sweep SYS RUN [WRITE]: -> out/SYS/pose_sweep_RUN/ (skipped if done and nothing to write)
    sel $1 $2
    [ -z "${3:-}" ] && [ -f out/$1/pose_sweep_$2/sweep.json ] && { echo "skip sweep $1 $2"; return 0; }
    python -m moments.pose_sweep --sim out/$1/noise_snr16 --run $2 ${3:+--write $3} 2>&1 | grep -v INFO \
        | tee out/$1/pose_sweep_$2.log | grep -E "top|wrote|projected"
}
round() {  # round SYS FROM_RUN VARIANT DTAG: theta + ref from FROM_RUN's best structures, passes 2-4 fixed
    local sys=$1 src=$2 v=$3 DT=$4 P
    sweep $sys $src $v > /dev/null
    local d=out/$sys/pose_sweep_$src
    [ -f $d/theta_$v.json ] || { log "$sys: no theta for $v"; return 1; }
    P=${v##*_P}
    restarts $sys moment2_imgd_posesuniform_w12_lr0.1 $DT POSE=fixed POSE_P=$P POSE_THETA=$d/theta_$v.json \
        POSE_REF=$d/ref_$v.cif
}
UNI=moment3_imgd_posesuniform_w12_m3wblock_lr0.1$F
log "1. sweeps"
sweep 8K9Z_A $UNI
sweep 8H1I_A $UNI
sweep 8K9Z_A moment3_imgd_posesem2_w12_m3wblock_lr0.1$F
read V V1 <<< $(python - <<'PY'
import json
r = {}
for s in ("8K9Z_A", "8H1I_A"):
    for x in json.load(open(f"out/{s}/pose_sweep_moment3_imgd_posesuniform_w12_m3wblock_lr0.1_seq80_seq30_seq15/sweep.json"))["rows"]:
        r.setdefault(x["variant"], []).append(x["err"])
m = {k: sum(v) / len(v) for k, v in r.items() if len(v) == 2}
V = min(m, key=m.get); V1 = min((k for k in m if k.startswith("k1_")), key=m.get)
print(V, V1)
PY
)
echo "  chosen variant V = $V, best single-structure variant V1 = $V1" | tee out/pose_theta_choice.log
for SYS in 8H1I_A 8K9Z_A; do
    log "2. $SYS round 1 with $V";  round $SYS $UNI $V _posesv1 || log "$SYS V round 1 FAILED"
    log "2. $SYS round 2 with $V";  round $SYS moment3_imgd_posesv1_w12_m3wblock_lr0.1$F $V _posesv2 || log "$SYS V round 2 FAILED"
    if [ "$V1" != "$V" ]; then
        log "2. $SYS round 1 with $V1"; round $SYS $UNI $V1 _posesk1 || log "$SYS V1 FAILED"
    fi
done
log "checks and summary"
for SYS in 8K9Z_A 8H1I_A; do
    R="moment3_imgd_snr16n1e5eps1e-1_w12_m3wblock_lr0.1$F"
    for a in oracle oracle2 uniform em1 em2 v1 v2 k1; do R="$R moment3_imgd_poses${a}_w12_m3wblock_lr0.1$F"; done
    make check SYSTEM=$SYS FIG=check_poses_theta CHECK_DATA=$(data $SYS) RUNS="$R" > out/$SYS/check_poses_theta.log 2>&1
done
V=$V V1=$V1 python - <<'PY' 2>&1 | tee out/pose_theta_summary.log
import json, os, pandas as pd
F = "_seq80_seq30_seq15"; V, V1 = os.environ["V"], os.environ["V1"]
run = lambda a: f"moment3_imgd_poses{a}_w12_m3wblock_lr0.1{F}"
print(f"chosen variant V = {V}; best single-structure variant V1 = {V1}\n")
for s in ("8K9Z_A", "8H1I_A"):
    for src in ("posesuniform", "posesem2", "posesv1"):
        f = f"out/{s}/pose_sweep_moment3_imgd_{src}_w12_m3wblock_lr0.1{F}/sweep.json"
        if not os.path.isfile(f): continue
        j = json.load(open(f)); rows = sorted(j["rows"], key=lambda r: r["err"])
        base = next((r["err"] for r in j["rows"] if r["variant"] == "k1_both_r0_P4"), float("nan"))
        print(f"{s}, density fitted from {src} (seeds {j['seeds'][:5]}): baseline k1_both_r0_P4 error {base:.3f}; "
              f"projection of the truth P4 {j['projection_err']['4']:.3f}")
        for r in rows[:6]: print(f"    {r['variant']:<22} {r['err']:.3f}")
        for key in ("k", "terms", "ridge", "P"):
            g = pd.DataFrame(j["rows"]).groupby(key)["err"].min()
            print(f"    best by {key}: " + ", ".join(f"{i}: {v:.3f}" for i, v in g.items()))
    f = f"out/{s}/check_poses_theta.csv"
    if not os.path.isfile(f): continue
    d = pd.read_csv(f); core = d.pivot_table(index="seed", columns="run", values="core")
    arms = [("uniform poses, uniform model (reference)", f"moment3_imgd_snr16n1e5eps1e-1_w12_m3wblock_lr0.1{F}"),
            ("oracle" if s == "8H1I_A" else "oracle (aligned each step)", run("oracle" if s == "8H1I_A" else "oracle2")),
            ("uniform assumed", run("uniform")), ("alternation round 1 (k1, both, P4)", run("em1")),
            ("alternation round 2 (k1, both, P4)", run("em2")), (f"round 1 with {V}", run("v1")),
            (f"round 2 with {V}", run("v2")), (f"round 1 with {V1}", run("k1"))]
    u = run("uniform")
    print(f"\n=== {s}, final pass, 10 seeds\n{'arm':<44}{'median':>8}{'full':>7}{'core<2':>8}{'vs uniform assumed':>22}{'sel.':>7}")
    for lab, r in arms:
        x = d[d.run == r]
        if not len(x): continue
        p = (core[r] - core[u]).dropna() if (u in core and r != u) else None
        ps = f"{p.median():+.2f} A, {int((p < 0).sum())}/{len(p)}" if p is not None else ""
        print(f"{lab:<44}{x.core.median():>8.2f}{x.rmsd.median():>7.2f}{(x.core < 2).mean():>8.0%}{ps:>22}"
              f"{x.loc[x.m3_img.idxmin(), 'core']:>7.2f}")
    print()
PY
log "all done"

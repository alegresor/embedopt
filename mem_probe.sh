#!/usr/bin/env bash
# Peak GPU memory (nvidia-smi, MiB) of the moment and map passes, per system -> mem_probe.csv
set -u
echo "system,tokens,pass,peak_mib" > mem_probe.csv
declare -A TOK=([8AHU_A]=250 [8K9Z_A]=283 [8H1I_A]=405 [7N6N]=617)   # fix the three benchmark counts if needed
for SYS in 8AHU_A 8K9Z_A 8H1I_A 7N6N; do
  for PASS in moment map; do
    case $PASS in
      moment) ARGS="METHOD=moment ORDER=3 SPACE=image DATA=exact LR=0.1 TAG=_mem" ;;
      map)    ARGS="METHOD=map LR=0.5 TAG=_mem" ;;
    esac
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 200 > /tmp/mem.log &
    MON=$!
    make run SYSTEM=$SYS $ARGS SEEDS=101 N_STEP=20 > logs_mem_${SYS}_${PASS}.txt 2>&1
    kill $MON
    PEAK=$(sort -n /tmp/mem.log | tail -1)
    echo "$SYS,${TOK[$SYS]},$PASS,$PEAK" | tee -a mem_probe.csv
  done
done
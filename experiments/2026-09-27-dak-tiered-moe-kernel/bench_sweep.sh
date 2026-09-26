#!/usr/bin/env bash
# Benchmark sweep of tiered_moe on one GPU of the standby node, NUMA-bound.
#   bench_sweep.sh <gpu> <binary> [grid]
set -uo pipefail
gpu=$1; bin=$2; grid=${3:-core}
node="$(.venv/bin/python agent_space/experiments/2026-07-29-marlin-smem-monopoly/detect_numa.py "${gpu}")"
run() { CUDA_VISIBLE_DEVICES=${gpu} numactl --cpunodebind="${node}" --membind="${node}" "${bin}" bench "$@"; }
echo "GPU ${gpu} Grace node ${node} on $(hostname)"
if [[ ${grid} == core ]]; then
  for g in w13 w2; do
    for cc in 4 8 16; do
      for hc in "9 2" "8 1" "11 3" "6 2"; do run ${g} ${hc} ${cc} 4 0; done
    done
    for hc in "9 0" "0 2"; do run ${g} ${hc} 8 4 0; done
  done
fi
if [[ ${grid} == quick ]]; then
  for g in w13 w2; do
    run ${g} 9 0 0 4 0
    for cc in 8 16 24; do run ${g} 9 2 ${cc} 4 0; done
  done
fi

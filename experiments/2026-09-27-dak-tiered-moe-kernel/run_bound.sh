#!/usr/bin/env bash
# Run a benchmark binary on one GPU of the standby node, bound to that GPU's
# Grace NUMA node (pinned cold experts must live there, or C2C reads run ~5x slow).
#   run_bound.sh <gpu> <binary> <args...>
set -uo pipefail
gpu=$1; shift
repo=/e/project1/profound/alint77/vllm
node="$("${repo}"/.venv/bin/python "${repo}"/agent_space/experiments/2026-07-29-marlin-smem-monopoly/detect_numa.py "${gpu}")"
CUDA_VISIBLE_DEVICES=${gpu} exec numactl --cpunodebind="${node}" --membind="${node}" "$@"

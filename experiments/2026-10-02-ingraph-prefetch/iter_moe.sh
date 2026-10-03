#!/usr/bin/env bash
# One prefill-MoE kernel iteration on a held node: bench_iter.py for each
# VLLM_TIERED_PREFILL_DEFINES variant given (quoted), GPU-local NUMA binding.
#   ./onnode.sh <D>/iter_moe.sh "" "TP_PRODUCER_WIDE=1" ...
cd /e/project1/profound/alint77/vllm
D=agent_space/experiments/2026-10-02-ingraph-prefetch
export VLLM_CACHE_ROOT=/e/fscratch/profound/${USER}/caches/prefill-kernel-iter TRITON_CACHE_DIR=/e/fscratch/profound/${USER}/cache/triton
for d in "$@"; do
  VLLM_TIERED_PREFILL_DEFINES="$d" numactl --cpunodebind=0 --membind=0 .venv/bin/python $D/bench_iter.py 2>&1 | grep -v -i warn
done

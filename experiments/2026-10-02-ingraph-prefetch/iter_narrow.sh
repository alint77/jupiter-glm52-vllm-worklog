#!/usr/bin/env bash
# Per define set: ptxas check, kernel tests built with those defines, bench.
cd /e/project1/profound/alint77/vllm
D=agent_space/experiments/2026-10-02-ingraph-prefetch
export VLLM_CACHE_ROOT=/e/fscratch/profound/${USER}/caches/prefill-kernel-iter
for d in "$@"; do
  $D/ptx_check.sh "$d"
  VLLM_TIERED_PREFILL_DEFINES="$d" .venv/bin/python -m pytest -q tests/kernels/moe/test_tiered_prefill_moe.py 2>&1 | tail -1
done
bash $D/iter_moe.sh "$@"

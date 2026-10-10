#!/usr/bin/env bash
# On a hold: bench_dcp.py over 4 GPUs.  run_bench.sh <log> <script> [args]
cd /e/project1/profound/alint77/vllm/agent_space/experiments/2026-10-11-dcp-oneshot-bw
export VLLM_CACHE_ROOT=/e/fscratch/profound/${USER}/caches/kdev TMPDIR=/e/fscratch/profound/${USER}/caches/tmp-kdev
log=$1; shift
/e/project1/profound/alint77/vllm/.venv/bin/torchrun --nproc-per-node 4 "$@" > $log 2>&1

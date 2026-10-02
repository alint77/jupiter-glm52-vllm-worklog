#!/usr/bin/env bash
# Run a Python script on a held node with the env, GPU-local NUMA binding.
cd /e/project1/profound/alint77/vllm
export TRITON_CACHE_DIR=/e/fscratch/profound/${USER}/cache/triton
numactl --cpunodebind=0 --membind=0 .venv/bin/python "$@" 2>&1 | grep -v -i warn

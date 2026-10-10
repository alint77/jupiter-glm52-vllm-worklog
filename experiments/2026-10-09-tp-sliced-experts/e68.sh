#!/usr/bin/env bash
# vLLM's sliced decode tests against the served kernel source
cd /e/project1/profound/alint77/vllm
export VLLM_CACHE_ROOT=/e/fscratch/profound/${USER}/caches/kdev-served TMPDIR=/e/fscratch/profound/${USER}/caches/tmp-kdev
numactl --cpunodebind=0 --membind=0 .venv/bin/python -m pytest -q tests/kernels/moe/test_tiered_decode_sliced.py 2>&1 | tail -15

#!/usr/bin/env bash
# sliced kernel test with routed_scale (in-tree kernel = v27 + routed scale)
cd /e/project1/profound/alint77/vllm
export VLLM_CACHE_ROOT=/e/fscratch/profound/${USER}/caches/kdev TMPDIR=/e/fscratch/profound/${USER}/caches/tmp-kdev
CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=0 --membind=0 timeout 1500 .venv/bin/python -m pytest -q tests/kernels/moe/test_tiered_decode_sliced.py 2>&1 | tail -5

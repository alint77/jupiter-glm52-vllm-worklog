#!/usr/bin/env bash
cd /e/project1/profound/alint77/vllm
export TRITON_CACHE_DIR=/e/fscratch/profound/${USER}/cache/triton
OUT=/e/fscratch/profound/${USER}/ingraph-prefetch/marlin
numactl --cpunodebind=0 --membind=0 .venv/bin/python benchmarks/kernels/benchmark_moe_wna16_marlin_prefill.py \
  --backend triton --tokens 512 1024 2048 4096 2>&1 | grep -v -i warn | tee "${OUT}/bench_triton.txt"
echo "=== triton done $(date +%T)"

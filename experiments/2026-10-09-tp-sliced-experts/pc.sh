#!/usr/bin/env bash
# Build + run a standalone probe on GPU 0:  pc.sh <probe.cu> [args]
cd /e/project1/profound/alint77/vllm/agent_space/experiments/2026-10-09-tp-sliced-experts
src=$1; shift
b=/e/fscratch/profound/${USER}/caches/kdev/probes; mkdir -p $b
exe=$b/$(basename $src .cu)
nvcc -O3 -gencode=arch=compute_90a,code=sm_90a -std=c++17 -lineinfo -o $exe kernels/$src 2>&1 | grep -v "^$" | head -20
cuobjdump -sass $exe > $exe.sass
CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=0 --membind=0 $exe "$@"

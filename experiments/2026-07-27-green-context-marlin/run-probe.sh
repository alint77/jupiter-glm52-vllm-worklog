#!/usr/bin/env bash
# Build + run the green-context feasibility probe on a GH200 (login node or Booster).
# Track A, plan §4.1-4.2. No Slurm required for the mechanism test; the login-node
# GH200 is sufficient. Use job.sh for an nsys/ncu-instrumented Booster run.
set -euo pipefail
cd "$(dirname -- "$0")"
exp_dir="$(pwd)"
source "$(dirname -- "$exp_dir")/../jupiter-env.sh"   # vllm/agent_space/jupiter-env.sh

bin="${exp_dir}/green_context_probe"
echo "=== nvcc build (sm_90a) ==="
nvcc -O2 -arch=sm_90a -o "$bin" green_context_probe.cu -lcuda
echo "built: $bin"

echo "=== run on $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) ==="
CUDA_DEVICE_MAX_CONNECTIONS=8 "$bin" "${1:-20000000}" 2>&1 | tee "${exp_dir}/logs/run-$(date +%Y%m%d-%H%M%S).log"

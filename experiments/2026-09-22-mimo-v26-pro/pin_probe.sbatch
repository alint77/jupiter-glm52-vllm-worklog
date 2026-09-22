#!/usr/bin/env bash
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:10:00
#SBATCH --job-name=mimo26-pin
#SBATCH --output=agent_space/experiments/2026-09-22-mimo-v26-pro/pin-%j.out
#SBATCH --error=agent_space/experiments/2026-09-22-mimo-v26-pro/pin-%j.err

# One MoE layer, one GPU. Measures runtime_expert_bytes for the tiered planner.
# gpu:4 because Booster has no partial-node allocation; the probe uses GPU 0.
set -euo pipefail
cd /e/project1/profound/alint77/vllm
source agent_space/jupiter-env.sh
export CUDA_VISIBLE_DEVICES=0
export VLLM_CACHE_ROOT=/e/fscratch/profound/${USER}/caches/marlin/vllm-cache-mimo26
export TRITON_CACHE_DIR=/e/fscratch/profound/${USER}/caches/triton
export TORCHINDUCTOR_CACHE_DIR=/e/fscratch/profound/${USER}/caches/inductor
mkdir -p "${VLLM_CACHE_ROOT}"
exec numactl --cpunodebind=0 --membind=0 .venv/bin/python agent_space/experiments/2026-09-22-mimo-v26-pro/pin_probe.py

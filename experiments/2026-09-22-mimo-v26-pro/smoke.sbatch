#!/usr/bin/env bash
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --job-name=mimo26-smoke
#SBATCH --output=agent_space/experiments/2026-09-22-mimo-v26-pro/smoke-%j.out
#SBATCH --error=agent_space/experiments/2026-09-22-mimo-v26-pro/smoke-%j.err

# First contact with MiMo-V2.6-Pro-RL: no tiered MoE, stock CPU offload.
#
# Answers, in one launch, what the tiered port would otherwise have to guess:
#   * do the mxfp4/fp8 expert weights load through mimo_v2.py at all
#   * are the visual.*/audio.* towers skipped cleanly (load_weights drops
#     anything not in params_dict, so they should be)
#   * is max_model_len reachable, and above all
#   * are the 60 sliding-window layers promoted to full attention for cache
#     allocation, which is a 7x KV swing at 250K
#
# Context is deliberately 64K, not the target 250K: bytes/token extrapolates
# exactly and a smaller cache cannot OOM the first attempt.
set -euo pipefail
cd /e/project1/profound/alint77/vllm
source agent_space/jupiter-env.sh
export VLLM_USE_V2_MODEL_RUNNER=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_CACHE_ROOT=/e/fscratch/profound/${USER}/caches/marlin/vllm-cache-mimo26
export TRITON_CACHE_DIR=/e/fscratch/profound/${USER}/caches/triton
export TORCHINDUCTOR_CACHE_DIR=/e/fscratch/profound/${USER}/caches/inductor
mkdir -p "${VLLM_CACHE_ROOT}"
exec .venv/bin/python agent_space/experiments/2026-09-22-mimo-v26-pro/smoke.py

#!/usr/bin/env bash
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:20:00
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
#
# 20 minutes, not the default hour, to fit a backfill window -- sinfo showed 20
# idle nodes while this sat at reason=Priority. Sized from a measured
# reference: GLM-5.3-W4A16 is 451 GB and reached "experts placed" at 3:00,
# fully ready at 7:47 with torch.compile. MiMo is 573 GB (1.27x) and runs
# eager, so ~4 min to weights and ~8 to the KV numbers. Those numbers are
# logged during engine init, before generation, so even a truncated run
# still answers the sliding-window question.
set -euo pipefail
cd /e/project1/profound/alint77/vllm
source agent_space/jupiter-env.sh
# VLLM_USE_V2_MODEL_RUNNER is deliberately NOT set. cpu_offload_gb only works
# under the V1 runner: gpu_model_runner.py:933 calls
# set_offloader(create_offloader(...)), and the V2 runner
# (v1/worker/gpu/model_runner.py) never does, so get_offloader() stays a
# NoopOffloader whose wrap_modules returns modules untouched. Under V2 the flag
# is accepted and silently ignored -- jobs 1943437 and 1944037 OOMed at 94.48
# and 94.47 GiB resident for cpu_offload_gb of 60 and 105 respectively, a 0.01
# GiB difference across a 45 GiB change in budget.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# cpu_offload_gb is PER GPU and pins host memory, so 105 meant 420 GiB pinned
# across four workers -- job 1944307 died silently right after the Marlin MoE
# backend came up, no traceback, no core. Only ~50 GiB per rank is needed
# (~145 GB/rank against 95 GiB of HBM), so 60 leaves margin on both sides.
export SMOKE_OFFLOAD_GB=60
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_CACHE_ROOT=/e/fscratch/profound/${USER}/caches/marlin/vllm-cache-mimo26
export TRITON_CACHE_DIR=/e/fscratch/profound/${USER}/caches/triton
export TORCHINDUCTOR_CACHE_DIR=/e/fscratch/profound/${USER}/caches/inductor
mkdir -p "${VLLM_CACHE_ROOT}"
exec .venv/bin/python agent_space/experiments/2026-09-22-mimo-v26-pro/smoke.py

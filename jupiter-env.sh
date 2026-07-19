#!/usr/bin/env bash

module load Stages/2026
module load GCC/14.3.0 CUDA/13 CMake/3.31.8 NCCL/default-CUDA-13
module load ccache/4.11.3 Ninja/1.13.0

export TRITON_PTXAS_PATH="$(command -v ptxas)"

VLLM_REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "${VLLM_REPO_DIR}/.venv/bin/activate"

export GLM52_W4A16_MODEL="$(dirname -- "${VLLM_REPO_DIR}")/models/GLM-5.2-W4A16-55c92ae"
export CCACHE_DIR=/e/scratch/profound/naeimitabiei1/vllm-ccache
export CCACHE_NOHASHDIR=true
export PRE_COMMIT_HOME="${PRE_COMMIT_HOME:-/e/scratch/profound/naeimitabiei1/pre-commit}"
export XDG_CACHE_HOME=/e/scratch/profound/naeimitabiei1/cache
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/e/scratch/profound/naeimitabiei1/vllm-cache}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-/e/scratch/profound/naeimitabiei1/flashinfer}"
export TRTLLM_DG_CACHE_DIR="${TRTLLM_DG_CACHE_DIR:-/e/scratch/profound/naeimitabiei1/trtllm-deepgemm}"
export MAX_JOBS="${MAX_JOBS:-4}"
export VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE=1
export VLLM_ENABLE_INDUCTOR_COORDINATE_DESCENT_TUNING=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

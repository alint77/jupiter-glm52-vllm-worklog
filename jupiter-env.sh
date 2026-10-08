#!/usr/bin/env bash

module load Stages/2026
module load GCC/14.3.0 CUDA/13 CMake/3.31.8 NCCL/default-CUDA-13
module load ccache/4.11.3 Ninja/1.13.0 git/2.50.1

export TRITON_PTXAS_PATH="$(command -v ptxas)"

VLLM_REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export VLLM_VENV_DIR="${VLLM_VENV_DIR:-${VLLM_REPO_DIR}/.venv}"
source "${VLLM_VENV_DIR}/bin/activate"

export GLM52_W4A16_MODEL="$(dirname -- "${VLLM_REPO_DIR}")/models/GLM-5.2-W4A16-55c92ae"
export CCACHE_DIR=/e/fscratch/profound/${USER:-$(id -un)}/vllm-ccache
export CCACHE_NOHASHDIR=true
export PRE_COMMIT_HOME="${PRE_COMMIT_HOME:-/e/fscratch/profound/${USER:-$(id -un)}/pre-commit}"
export XDG_CACHE_HOME=/e/fscratch/profound/${USER:-$(id -un)}/cache
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/e/fscratch/profound/${USER:-$(id -un)}/vllm-cache}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-/e/fscratch/profound/${USER:-$(id -un)}/flashinfer}"
export TRTLLM_DG_CACHE_DIR="${TRTLLM_DG_CACHE_DIR:-/e/fscratch/profound/${USER:-$(id -un)}/trtllm-deepgemm}"
export MAX_JOBS="${MAX_JOBS:-4}"
export VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE=1
export VLLM_ENABLE_INDUCTOR_COORDINATE_DESCENT_TUNING=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

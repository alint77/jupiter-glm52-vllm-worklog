#!/usr/bin/env bash
# Run a Python script on the held node against a worktree (default pfk), with
# its own extension cache and GPU-local NUMA binding (GPU=n picks the GPU):
#   HOLD_JOB=<id> onnode.sh "[GPU=n] [DEFINES='TP_T64=1 ...'] <X>/run.sh [TREE=pfk] script.py args..."
cd /e/project1/profound/alint77/vllm
TREE=pfk
[[ $1 == TREE=* ]] && { TREE=${1#TREE=}; shift; }
GPU=${GPU:-0}
export PATH=$PWD/.venv/bin:$PATH
export CUDA_VISIBLE_DEVICES=$GPU
export VLLM_TIERED_PREFILL_DEFINES="${DEFINES:-}"
ROOT=/e/fscratch/profound/${USER}/worktrees/${TREE}; [[ $TREE == main ]] && ROOT=$PWD
export PYTHONPATH=$ROOT:agent_space/experiments/2026-10-11-sliced-prefill-moe
export VLLM_CACHE_ROOT=/e/fscratch/profound/${USER}/caches/kernel-${TREE}
export TRITON_CACHE_DIR=/e/fscratch/profound/${USER}/cache/triton
numactl --cpunodebind=$GPU --membind=$GPU .venv/bin/python "$@" 2>&1 | grep -v -i warn

#!/usr/bin/env bash
# Native (non-tiered) GLM-5.2 W4A16 server for the 2-node TP4 x PP2 experiment.
#
# Invoked once per node by job-c1.sh / job-c4.sh via srun. Each node runs one
# vllm process; node_rank 0 is the leader (API server + EngineCore + 4 local
# workers = PP stage 0), node_rank 1 runs 4 workers = PP stage 1. They connect
# over the internal network via --master-addr/--master-port.
#
# This launcher deliberately uses NO --tiered-moe / --cpu-offload flags: all
# 361 GB of weights live in HBM across 8 ranks (~45 GB/rank). Modeled on
# agent_space/run-cpu-offload-baseline.sh, NOT the tiered run-server.sh.
#
# Env (set by the job script):
#   MODE         c1 | c4
#   NODE_RANK    0 | 1
#   MASTER_ADDR  leader node hostname
#   MASTER_PORT  int
#   JOB_CACHE    per-job scratch root (avoids parallel-cache corruption)

set -euo pipefail

mode="${MODE:?MODE must be c1 or c4}"
# NODE_RANK is set per-task by srun via SLURM_NODEID (0 on the leader, 1 on the
# worker node). Do NOT export NODE_RANK in the batch script (it would be 0 for
# both tasks); let each task read its own SLURM_NODEID here.
node_rank="${NODE_RANK:-${SLURM_NODEID:?no node rank}}"
master_addr="${MASTER_ADDR:?}"
master_port="${MASTER_PORT:?}"
job_cache="${JOB_CACHE:?}"

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." 2>/dev/null && pwd)"
[[ -f "${repo_dir}/agent_space/jupiter-env.sh" ]] || repo_dir=/e/project1/profound/alint77/vllm

cd "${repo_dir}"
# Per-job caches so the parallel c1/c4 jobs do not corrupt shared compile state.
export VLLM_CACHE_ROOT="${job_cache}/vllm-cache"
export TRTLLM_DG_CACHE_DIR="${job_cache}/trtllm-dg"
export FLASHINFER_WORKSPACE_BASE="${job_cache}/flashinfer"
export XDG_CACHE_HOME="${job_cache}/xdg-cache"
source agent_space/jupiter-env.sh

export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=WARN
export NCCL_NET_GDR_LEVEL=PHB
# HF offline (Booster has no external internet); jupiter-env already sets these.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

common_args=(
  --served-model-name glm52-w4a16
  --host 0.0.0.0
  --port 8000
  --tensor-parallel-size 4
  --pipeline-parallel-size 2
  --distributed-executor-backend mp
  --nnodes 2
  --node-rank "${node_rank}"
  --master-addr "${master_addr}"
  --master-port "${master_port}"
  --enable-expert-parallel
  --enable-ep-weight-filter
  --numa-bind
  --kv-cache-dtype fp8_ds_mla
  --max-model-len 400000
  --max-num-batched-tokens 8192
  --gpu-memory-utilization 0.90
  --optimization-level 2
)

case "${mode}" in
  c1)
    # V1 runner (default), batch one, no DCP.
    compilation_config='{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1],"compile_sizes":[1],"pass_config":{"fuse_allreduce_rms":false}}'
    common_args+=(
      --max-num-seqs 1
      --decode-context-parallel-size 1
      --compilation-config "${compilation_config}"
    )
    ;;
  c4)
    # V2 runner + DCP4 at 400K (4 concurrent). VLLM_USE_V2_MODEL_RUNNER=1 below.
    export VLLM_USE_V2_MODEL_RUNNER=1
    compilation_config='{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1,2,3,4],"compile_sizes":[1,2,3,4],"cudagraph_num_of_warmups":1,"pass_config":{"fuse_allreduce_rms":false}}'
    common_args+=(
      --max-num-seqs 4
      --decode-context-parallel-size 4
      --compilation-config "${compilation_config}"
    )
    ;;
  *)
    echo "unknown MODE=${mode}" >&2
    exit 2
    ;;
esac

exec "${VLLM_VENV_DIR:-${PWD}/.venv}/bin/vllm" serve "${GLM52_W4A16_MODEL}" "${common_args[@]}"

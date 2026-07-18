#!/usr/bin/env bash
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --output=agent_space/experiments/2026-07-18-dcp-port/slurm-%x-%j.out
#SBATCH --error=agent_space/experiments/2026-07-18-dcp-port/slurm-%x-%j.err

set -euo pipefail

depth="${1:?speculative depth is required}"
dcp_size="${2:?decode-context-parallel size is required}"
label="${3:?result label is required}"
profile="${4:-false}"
cudagraph_mode="${5:-FULL_AND_PIECEWISE}"
debug_env="${6:-false}"

if [[ "${debug_env}" == true ]]; then
  export NCCL_DEBUG=WARN
  export CUDA_LAUNCH_BLOCKING=1
fi

cd /e/project1/profound/alint77/vllm
source agent_space/jupiter-env.sh

export VLLM_CACHE_ROOT="/e/scratch/profound/naeimitabiei1/vllm-cache-${SLURM_JOB_ID}"
export FLASHINFER_WORKSPACE_BASE="/e/scratch/profound/naeimitabiei1/flashinfer-${SLURM_JOB_ID}"
export TRTLLM_DG_CACHE_DIR="/e/scratch/profound/naeimitabiei1/trtllm-deepgemm-${SLURM_JOB_ID}"

profile_dir=""
if [[ "${profile}" == true ]]; then
  profile_dir="/e/scratch/profound/naeimitabiei1/glm52-${label}-profile-${SLURM_JOB_ID}"
fi

result_dir=agent_space/experiments/2026-07-18-dcp-port
server_out="${result_dir}/${label}-server.out"
server_err="${result_dir}/${label}-server.err"
agent_space/experiments/2026-07-18-dcp-port/run-server.sh \
  "${depth}" "${dcp_size}" "${cudagraph_mode}" "${profile_dir}" \
  >"${server_out}" 2>"${server_err}" &
server_pid=$!
trap 'kill "${server_pid}" 2>/dev/null || true' EXIT

ready=false
for _ in $(seq 1 240); do
  if curl -fsS http://127.0.0.1:8027/health >/dev/null; then
    ready=true
    break
  fi
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    wait "${server_pid}"
  fi
  sleep 10
done

if [[ "${ready}" != true ]]; then
  echo "Server did not become ready" >&2
  exit 1
fi

agent_space/experiments/2026-07-18-dcp-port/run-benchmark.sh \
  "${label}" "${profile}"

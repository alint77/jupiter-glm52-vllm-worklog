#!/usr/bin/env bash
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --output=agent_space/experiments/2026-07-23-c4-mtp-depth/slurm-%x-%j.out
#SBATCH --error=agent_space/experiments/2026-07-23-c4-mtp-depth/slurm-%x-%j.err

set -euo pipefail

depth="${1:?MTP depth is required}"
label="${2:?result label is required}"
cache_root="${3:?a job-exclusive vLLM cache root is required}"

repo_dir=/e/project1/profound/alint77/vllm
result_dir="${repo_dir}/agent_space/experiments/2026-07-23-c4-mtp-depth"
trace_dir="/e/scratch/profound/naeimitabiei1/glm52-${label}-profile-${SLURM_JOB_ID}"

cd "${repo_dir}"
source agent_space/jupiter-env.sh

export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_CACHE_ROOT="${cache_root}"
export TRTLLM_DG_CACHE_DIR="/e/scratch/profound/naeimitabiei1/trtllm-deepgemm-${SLURM_JOB_ID}"
export TIERED_MOE_PLACEMENT_PROFILE="${repo_dir}/agent_space/experiments/2026-07-19-c1q4-placement/per-expert-profile.json"
export TIERED_MOE_HBM_RESERVE_GB=7
export VLLM_TORCH_PROFILER_DELAY_ITERATIONS=2

mkdir -p "${trace_dir}" "${VLLM_CACHE_ROOT}" "${TRTLLM_DG_CACHE_DIR}"
printf '%s\n' "${trace_dir}" >"${result_dir}/${label}-trace-path.txt"

agent_space/experiments/2026-07-18-dcp-port/run-server-c4.sh \
  "${depth}" 4 4 "${trace_dir}" \
  >"${result_dir}/${label}-server.out" \
  2>"${result_dir}/${label}-server.err" &
server_pid=$!
trap 'kill "${server_pid}" 2>/dev/null || true' EXIT

ready=false
for _ in $(seq 1 360); do
  if curl -fsS http://127.0.0.1:8027/health >/dev/null; then
    ready=true
    break
  fi
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    wait "${server_pid}"
  fi
  sleep 10
done
[[ "${ready}" == true ]]

agent_space/experiments/2026-07-23-c4-mtp-depth/run-benchmark.sh \
  "${label}"

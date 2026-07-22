#!/usr/bin/env bash
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --job-name=glm52-claude-c1
#SBATCH --output=agent_space/experiments/2026-07-22-claude-local/slurm-%j.out
#SBATCH --error=agent_space/experiments/2026-07-22-claude-local/slurm-%j.err

set -euo pipefail

repo_dir=/e/project1/profound/alint77/vllm
run_dir="${CLAUDE_LOCAL_STATE_DIR:-/e/scratch/profound/naeimitabiei1/claude-local}"
port="${CLAUDE_LOCAL_API_PORT:-8027}"
token_file="${run_dir}/api-token"
ready_file="${run_dir}/ready-${SLURM_JOB_ID}"
endpoint_file="${run_dir}/endpoint-${SLURM_JOB_ID}"
log_dir="${repo_dir}/agent_space/experiments/2026-07-22-claude-local"

cd "${repo_dir}"
source agent_space/jupiter-env.sh

[[ -s "${token_file}" ]]
api_key="$(<"${token_file}")"
export VLLM_API_KEY="${api_key}"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_CACHE_ROOT=/e/scratch/profound/naeimitabiei1/vllm-cache-977753
export FLASHINFER_WORKSPACE_BASE=/e/scratch/profound/naeimitabiei1/flashinfer
export TRTLLM_DG_CACHE_DIR=/e/scratch/profound/naeimitabiei1/trtllm-deepgemm
export TIERED_MOE_PLACEMENT_PROFILE="${repo_dir}/agent_space/experiments/2026-07-19-c1q4-placement/hybrid-p0.5-profile.json"
export TIERED_MOE_HBM_RESERVE_GB=10
export VLLM_USE_V2_MODEL_RUNNER=0

mkdir -p "${run_dir}" "${log_dir}"
rm -f "${ready_file}" "${endpoint_file}"

server_pid=""
cleanup() {
  rm -f "${ready_file}" "${endpoint_file}"
  if [[ -n "${server_pid}" ]]; then
    kill "${server_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

agent_space/experiments/2026-07-18-dcp-port/run-server.sh \
  3 1 FULL_AND_PIECEWISE "" \
  --host 0.0.0.0 \
  --port "${port}" \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --reasoning-parser glm47 \
  --generation-config vllm \
  >"${log_dir}/server-${SLURM_JOB_ID}.out" \
  2>"${log_dir}/server-${SLURM_JOB_ID}.err" &
server_pid=$!

ready=false
for _ in $(seq 1 240); do
  if curl -fsS -H "Authorization: Bearer ${api_key}" \
      "http://127.0.0.1:${port}/health" >/dev/null; then
    ready=true
    break
  fi
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    wait "${server_pid}"
  fi
  sleep 10
done
[[ "${ready}" == true ]]

umask 077
node="$(hostname -s)"
tmp="${endpoint_file}.tmp"
printf 'JOB_ID=%q\nNODE=%q\nPORT=%q\n' \
  "${SLURM_JOB_ID}" "${node}" "${port}" >"${tmp}"
mv "${tmp}" "${endpoint_file}"
touch "${ready_file}"

wait "${server_pid}"

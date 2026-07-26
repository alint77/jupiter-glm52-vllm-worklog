#!/usr/bin/env bash
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --job-name=glm52-claude-ar-c4
#SBATCH --output=agent_space/experiments/2026-07-26-claude-autoround-c4/slurm-%j.out
#SBATCH --error=agent_space/experiments/2026-07-26-claude-autoround-c4/slurm-%j.err

set -euo pipefail

repo_dir=/e/project1/profound/alint77/vllm
run_dir="${CLAUDE_LOCAL_STATE_DIR:-/e/scratch/profound/naeimitabiei1/claude-local-c4}"
port="${CLAUDE_LOCAL_API_PORT:-8027}"
token_file="${run_dir}/api-token"
ready_file="${run_dir}/ready-${SLURM_JOB_ID}"
endpoint_file="${run_dir}/endpoint-${SLURM_JOB_ID}"
log_dir="${repo_dir}/agent_space/experiments/2026-07-26-claude-autoround-c4"

cd "${repo_dir}"
source agent_space/jupiter-env.sh

[[ -s "${token_file}" ]]
api_key="$(<"${token_file}")"
export VLLM_API_KEY="${api_key}"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_SERVER_DEV_MODE=1
export VLLM_ENFORCE_STRICT_TOOL_CALLING=false
export VLLM_CACHE_ROOT=/e/scratch/profound/naeimitabiei1/vllm-cache-claude-autoround-c4
export TRTLLM_DG_CACHE_DIR=/e/scratch/profound/naeimitabiei1/trtllm-dg-claude-autoround-c4
export TIERED_MOE_MODEL_PATH="$(dirname -- "${repo_dir}")/models/GLM-5.2-AutoRound-W4G64-MTP-e1ba887"
export TIERED_MOE_PLACEMENT_PROFILE="${repo_dir}/agent_space/experiments/2026-07-26-autoround-w4g64/hybrid-p0.5-profile.json"
export VLLM_TIERED_MOE_PROFILE_CAP=0
export TIERED_MOE_HBM_RESERVE_GB=7
export TIERED_MOE_COMPILATION_CONFIG='{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[4,8,12,16],"compile_sizes":[4,8,12,16],"cudagraph_num_of_warmups":1,"pass_config":{"fuse_allreduce_rms":false}}'

mkdir -p "${run_dir}" "${log_dir}" "${VLLM_CACHE_ROOT}" \
  "${TRTLLM_DG_CACHE_DIR}"
rm -f "${ready_file}" "${endpoint_file}"

server_pid=
cleanup() {
  rm -f "${ready_file}" "${endpoint_file}"
  if [[ -n "${server_pid}" ]]; then
    kill "${server_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

agent_space/experiments/2026-07-17-end-to-end-tuning/run-server.sh \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --decode-context-parallel-size 4 \
  --max-num-seqs 4 \
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

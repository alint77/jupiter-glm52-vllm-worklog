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
max_num_seqs="${3:?max concurrent sequences is required}"
label="${4:?result label is required}"
comm_backend="${5:-ag_rs}"
profile_dir="${6:-}"
profile_target="${7:-c1}"
server_args=("${@:8}")

cd /e/project1/profound/alint77/vllm
source agent_space/jupiter-env.sh

export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT_OVERRIDE:-/e/scratch/profound/naeimitabiei1/vllm-cache-${SLURM_JOB_ID}}"

result_dir=agent_space/experiments/2026-07-18-dcp-port
server_out="${result_dir}/${label}-server.out"
server_err="${result_dir}/${label}-server.err"
agent_space/experiments/2026-07-18-dcp-port/run-server-c4.sh \
  "${depth}" "${dcp_size}" "${max_num_seqs}" "${profile_dir}" \
  --dcp-comm-backend "${comm_backend}" \
  "${server_args[@]}" \
  >"${server_out}" 2>"${server_err}" &
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

if [[ "${ready}" != true ]]; then
  echo "Server did not become ready" >&2
  exit 1
fi

agent_space/experiments/2026-07-18-dcp-port/run-benchmark-c4.sh \
  "${label}" "${profile_dir:+${profile_target}}"

#!/usr/bin/env bash
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --job-name=mimo26-claude
#SBATCH --output=agent_space/experiments/2026-09-22-mimo-v26-pro/slurm-%j.out
#SBATCH --error=agent_space/experiments/2026-09-22-mimo-v26-pro/slurm-%j.err

# MiMo-V2.6-Pro-RL tiered MoE as a claude-local.sh backend: 250K context,
# c=1, DFlash k=7, 3 GB HBM reserve, text only, cold-expert prefill prefetch.
#
# The serving flags are serve.sbatch's (the measured best config, see the
# README), run in the background here instead of exec'd so the two cannot
# drift. This script adds only what an agent client needs: tool and reasoning
# parsers, an API key, and the ready/endpoint files claude-local.sh polls.
#
# generation_config.json stays in force (unlike the GLM launchers'
# --generation-config vllm) for its temperature 1.0 / top_p 0.95 and three EOS
# ids, but its max_new_tokens 2048 is overridden below: vLLM treats it as a
# server-wide output cap, which ended opencode turns mid-reasoning.

set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")/../../.." 2>/dev/null && pwd)"
[[ -f "${repo_dir}/agent_space/jupiter-env.sh" ]] || repo_dir=/e/project1/profound/alint77/vllm
E=agent_space/experiments/2026-09-22-mimo-v26-pro
run_dir="${CLAUDE_LOCAL_STATE_DIR:-/e/fscratch/profound/${USER:-$(id -un)}/claude-mimo26}"
port="${CLAUDE_LOCAL_API_PORT:-8027}"
token_file="${run_dir}/api-token"
ready_file="${run_dir}/ready-${SLURM_JOB_ID}"
endpoint_file="${run_dir}/endpoint-${SLURM_JOB_ID}"
log_dir="${CLAUDE_LOCAL_LOG_DIR:-${repo_dir}/${E}}"
[[ -w "${log_dir}" ]] || log_dir="${run_dir}/logs"

cd "${repo_dir}"
source agent_space/jupiter-env.sh

[[ -s "${token_file}" ]]
export VLLM_API_KEY="$(<"${token_file}")"
export VLLM_SERVER_DEV_MODE=1
export VLLM_ENFORCE_STRICT_TOOL_CALLING=false

model=/e/fscratch/profound/${USER:-$(id -un)}/models/MiMo-V2.6-Pro-RL
expected_shards=130
actual_shards="$(ls "${model}"/*.safetensors 2>/dev/null | wc -l)"
if (( actual_shards != expected_shards )) || [[ ! -f "${model}/dflash/config.json" ]]; then
  echo "model: ${model} missing or incomplete (${actual_shards}/${expected_shards} shards)" >&2
  exit 1
fi
echo "model: ${model} (${actual_shards} shards, dflash drafter present)"

export MIMO_PORT="${port}"
# generation_config.json's max_new_tokens (2048) is a server-wide output cap;
# null drops it and keeps the sampling defaults (get_diff_sampling_param).
export MIMO_EXTRA_ARGS="--language-model-only --enable-prefix-caching \
--enable-auto-tool-choice --tool-call-parser mimo --reasoning-parser mimo \
--override-generation-config {\"max_new_tokens\":null}"

mkdir -p "${run_dir}" "${log_dir}"
rm -f "${ready_file}" "${endpoint_file}"

server_pid=
cleanup() {
  rm -f "${ready_file}" "${endpoint_file}"
  if [[ -n "${server_pid}" ]]; then
    kill "${server_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

sed -e 's/^exec \.venv/.venv/' "${E}/serve.sbatch" | bash \
  >"${log_dir}/server-${SLURM_JOB_ID}.out" \
  2>"${log_dir}/server-${SLURM_JOB_ID}.err" &
server_pid=$!

ready=false
for _ in $(seq 1 180); do
  if curl -fsS -H "Authorization: Bearer ${VLLM_API_KEY}" \
      "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
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

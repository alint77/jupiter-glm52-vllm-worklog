#!/usr/bin/env bash
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --job-name=glm53-claude-df2-dcp4
#SBATCH --output=agent_space/experiments/2026-09-27-glm53-mtp7-profile/slurm-claude-%j.out
#SBATCH --error=agent_space/experiments/2026-09-27-glm53-mtp7-profile/slurm-claude-%j.err

# GLM-5.3 W4A16 as a claude-local.sh backend, with serve.sh's defaults (the
# settled baseline): DFlash2 (7 tokens, eager drafter), DCP4, 400K context,
# 7 GB reserve, exact replicas, INT4 one-kernel decode MoE, agentic profile,
# prefix caching on. serve.sh runs unchanged, so the two cannot drift; this adds
# only what an agent client needs: the tool and reasoning parsers, a reachable
# host and port, an API key, and the ready/endpoint files claude-local.sh polls.

set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")/../../.." 2>/dev/null && pwd)"
[[ -f "${repo_dir}/agent_space/jupiter-env.sh" ]] || repo_dir=/e/project1/profound/alint77/vllm
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
run_dir="${CLAUDE_LOCAL_STATE_DIR:-/e/fscratch/profound/${USER:-$(id -un)}/claude-glm53-df2-dcp4}"
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
export VLLM_ENFORCE_STRICT_TOOL_CALLING=false
# serve.sh appends SERVE_EXTRA last, so these override run-server.sh's
# 127.0.0.1:8027.
export SERVE_EXTRA="--host 0.0.0.0 --port ${port} --enable-auto-tool-choice \
--tool-call-parser glm47 --reasoning-parser glm45 ${SERVE_EXTRA:-}"

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

bash "${E}/serve.sh" \
  >"${log_dir}/server-claude-${SLURM_JOB_ID}.out" \
  2>"${log_dir}/server-claude-${SLURM_JOB_ID}.err" &
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

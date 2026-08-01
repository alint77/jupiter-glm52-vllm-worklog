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

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")/../../.." 2>/dev/null && pwd)"
[[ -f "${repo_dir}/agent_space/jupiter-env.sh" ]] || repo_dir=/e/project1/profound/alint77/vllm
run_dir="${CLAUDE_LOCAL_STATE_DIR:-/e/scratch/profound/${USER:-$(id -un)}/claude-local-c4}"
port="${CLAUDE_LOCAL_API_PORT:-8027}"
token_file="${run_dir}/api-token"
ready_file="${run_dir}/ready-${SLURM_JOB_ID}"
endpoint_file="${run_dir}/endpoint-${SLURM_JOB_ID}"
log_dir="${CLAUDE_LOCAL_LOG_DIR:-${repo_dir}/agent_space/experiments/2026-07-26-claude-autoround-c4}"
[[ -w "${log_dir}" ]] || log_dir="${run_dir}/logs"

cd "${repo_dir}"
source agent_space/jupiter-env.sh

[[ -s "${token_file}" ]]
api_key="$(<"${token_file}")"
export VLLM_API_KEY="${api_key}"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_SERVER_DEV_MODE=1
export VLLM_ENFORCE_STRICT_TOOL_CALLING=false
export VLLM_CACHE_ROOT=/e/scratch/profound/${USER:-$(id -un)}/vllm-cache-claude-autoround-c4
export TRTLLM_DG_CACHE_DIR=/e/scratch/profound/${USER:-$(id -un)}/trtllm-dg-claude-autoround-c4
# Serve the checkpoint from exa_fscratch: 86 shards load in 2 min 11 s there
# against 14 min 08 s from exa_project1, because four ranks read concurrently and
# fscratch scales with parallel readers (11 -> 21 GB/s across 4 to 8) where
# project1 does not (0.55 -> 0.95). See
# agent_space/experiments/2026-07-29-marlin-smem-monopoly/README.md
#
# Falls back to project1 if the staged copy is missing or incomplete, so a purge
# of fscratch degrades startup instead of breaking the launch. Restage with
# agent_space/experiments/2026-07-29-marlin-smem-monopoly/stage_model_fscratch.sh
model_name=GLM-5.2-AutoRound-W4G64-MTP-e1ba887
model_project1="$(dirname -- "${repo_dir}")/models/${model_name}"
model_fscratch="/e/fscratch/profound/${USER:-$(id -un)}/models/${model_name}"
if [[ -d "${model_fscratch}" ]] && \
   [[ "$(ls "${model_fscratch}"/*.safetensors 2>/dev/null | wc -l)" \
      -eq "$(ls "${model_project1}"/*.safetensors 2>/dev/null | wc -l)" ]]; then
  export TIERED_MOE_MODEL_PATH="${model_fscratch}"
  echo "model: fscratch (fast path) ${TIERED_MOE_MODEL_PATH}"
else
  export TIERED_MOE_MODEL_PATH="${model_project1}"
  echo "model: project1 fallback - fscratch copy missing or incomplete;" \
       "expect ~12 min slower load. Restage with stage_model_fscratch.sh"
fi
# Same owners and HBM residency as hybrid-p0.5, plus 985 Grace replica copies
# per rank that per-step replica assignment can schedule onto.
export TIERED_MOE_PLACEMENT_PROFILE="${repo_dir}/agent_space/experiments/2026-07-31-replica-scheduling-v2/hybrid-p0.5-replicas-985.json"
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
  --tiered-moe-replica-assignment exact \
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

#!/usr/bin/env bash
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --job-name=glm53-route-cap
#SBATCH --output=agent_space/experiments/2026-08-29-glm53-routing-capture/slurm-%j.out
#SBATCH --error=agent_space/experiments/2026-08-29-glm53-routing-capture/slurm-%j.err

# GLM-5.3 NVFP4 routing capture host. Records routed-expert IDs from agentic
# coding turns so the hot-expert ranking can be re-derived for 5.3; the shipped
# nvfp4-profile-2400.json still carries GLM-5.2's ranking.
#
# Constraints inherited from the 2026-07-26 GLM-5.2 capture, all load-bearing:
#   * V1 model runner: routed-expert return is not implemented on V2.
#   * no --decode-context-parallel-size: DCP does not return routed experts.
#   * c1 + MTP3 => 4 tokens verified per step, so graph shapes are [4] and
#     VLLM_ROUTING_TRACE_VERIFICATION_SIZE=4 trims the ragged prefix rows.
#   * VLLM_TIERED_MOE_PROFILE_CAP=0 leaves the resident set uncapped so routing
#     is not perturbed by the profiler.
# Traces start at generation, so the conversation prefix is not counted twice.
# Only expert IDs are written -- no prompts, responses, or tool arguments.

set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")/../../.." 2>/dev/null && pwd)"
[[ -f "${repo_dir}/agent_space/jupiter-env.sh" ]] || repo_dir=/e/project1/profound/alint77/vllm
run_dir="${CLAUDE_LOCAL_STATE_DIR:-/e/scratch/profound/${USER:-$(id -un)}/claude-glm53-capture}"
port="${CLAUDE_LOCAL_API_PORT:-8029}"
token_file="${run_dir}/api-token"
ready_file="${run_dir}/ready-${SLURM_JOB_ID}"
endpoint_file="${run_dir}/endpoint-${SLURM_JOB_ID}"
log_dir="${CLAUDE_LOCAL_LOG_DIR:-${repo_dir}/agent_space/experiments/2026-08-29-glm53-routing-capture}"
[[ -w "${log_dir}" ]] || log_dir="${run_dir}/logs"

cd "${repo_dir}"
source agent_space/jupiter-env.sh

[[ -s "${token_file}" ]]
api_key="$(<"${token_file}")"
export VLLM_API_KEY="${api_key}"
export CUDA_VISIBLE_DEVICES=0,1,2,3
unset VLLM_USE_V2_MODEL_RUNNER
export VLLM_SERVER_DEV_MODE=1
export VLLM_ENFORCE_STRICT_TOOL_CALLING=false
export VLLM_CACHE_ROOT=/e/fscratch/profound/${USER:-$(id -un)}/caches/marlin/vllm-cache-glm53-capture
export TRTLLM_DG_CACHE_DIR=/e/fscratch/profound/${USER:-$(id -un)}/caches/marlin/trtllm-dg-glm53-capture
export VLLM_ROUTING_TRACE_DIR="${run_dir}/routes-${SLURM_JOB_ID}"
export VLLM_ROUTING_TRACE_VERIFICATION_SIZE=4
export VLLM_TIERED_MOE_PROFILE_CAP=0

model_name=GLM-5.3-NVFP4
model_project1="$(dirname -- "${repo_dir}")/models/${model_name}"
model_fscratch="/e/fscratch/profound/${USER:-$(id -un)}/models/${model_name}"
if [[ -d "${model_fscratch}" ]] && \
   [[ "$(ls "${model_fscratch}"/*.safetensors 2>/dev/null | wc -l)" \
      -eq "$(ls "${model_project1}"/*.safetensors 2>/dev/null | wc -l)" ]]; then
  export TIERED_MOE_MODEL_PATH="${model_fscratch}"
  echo "model: fscratch (fast path) ${TIERED_MOE_MODEL_PATH}"
else
  export TIERED_MOE_MODEL_PATH="${model_project1}"
  echo "model: project1 fallback - fscratch copy missing or incomplete"
fi

export TIERED_MOE_PLACEMENT_PROFILE="${repo_dir}/agent_space/experiments/2026-08-28-nvfp4-tiered/nvfp4-profile-2400.json"
export TIERED_MOE_HBM_RESERVE_GB=7
export TIERED_MOE_COMPILATION_CONFIG='{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[4],"compile_sizes":[4],"cudagraph_num_of_warmups":1,"pass_config":{"fuse_allreduce_rms":false}}'

mkdir -p "${run_dir}" "${log_dir}" "${VLLM_CACHE_ROOT}" \
  "${TRTLLM_DG_CACHE_DIR}" "${VLLM_ROUTING_TRACE_DIR}"
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
  --max-num-seqs 1 \
  --enable-return-routed-experts \
  --host 0.0.0.0 \
  --port "${port}" \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --generation-config vllm \
  --served-model-name glm53-nvfp4-tiered \
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

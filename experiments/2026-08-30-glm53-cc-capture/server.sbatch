#!/usr/bin/env bash
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --job-name=glm53-cc-capture
#SBATCH --output=agent_space/experiments/2026-08-30-glm53-cc-capture/slurm-%j.out
#SBATCH --error=agent_space/experiments/2026-08-30-glm53-cc-capture/slurm-%j.err

# GLM-5.3 W4A16 routing-capture host, for deriving the hot-expert ranking from
# real Claude Code usage rather than the synthetic agentic driver.
#
# This is the prod c4 launcher (2026-08-29-glm53-c4/server.sbatch) with routed-
# expert return enabled. Two deviations from prod are forced, not chosen:
#
#   * V1 model runner. vllm/v1/worker/gpu/model_runner.py -- the V2 runner --
#     has no routed_experts support at all, so gpu_worker's
#     init_routed_experts_capturer() call would fail on it.
#   * no --decode-context-parallel-size. Scheduler.__init__ asserts
#     dcp_world_size == 1 when enable_return_routed_experts is set.
#
# Neither changes which experts the router picks: routing is the gate's top-k
# over hidden states, and DCP shards attention KV without altering the math. So
# a ranking derived here transfers to the prod DCP4/V2 host.
#
# What it costs: dropping DCP quarters the KV token capacity for the same pin,
# so 400k-token context runs at 1.00x concurrency instead of 4.00x. Requests
# past that queue rather than fail, and queuing does not perturb routing.
#
# MTP3 verifies 4 tokens per step, so VLLM_ROUTING_TRACE_VERIFICATION_SIZE=4
# trims the ragged prefix rows. Traces begin at the last prompt token
# (routed_experts_prompt_start = len(prompt) - 1, set in the serving layer), so
# a conversation prefix is never counted twice across turns and only generated
# positions are recorded. Only expert IDs are written -- no prompts, no
# responses, no tool arguments.

set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")/../../.." 2>/dev/null && pwd)"
[[ -f "${repo_dir}/agent_space/jupiter-env.sh" ]] || repo_dir=/e/project1/profound/alint77/vllm
run_dir="${CLAUDE_LOCAL_STATE_DIR:-/e/scratch/profound/${USER:-$(id -un)}/claude-glm53-cap}"
port="${CLAUDE_LOCAL_API_PORT:-8028}"
token_file="${run_dir}/api-token"
ready_file="${run_dir}/ready-${SLURM_JOB_ID}"
endpoint_file="${run_dir}/endpoint-${SLURM_JOB_ID}"
log_dir="${CLAUDE_LOCAL_LOG_DIR:-${repo_dir}/agent_space/experiments/2026-08-30-glm53-cc-capture}"
[[ -w "${log_dir}" ]] || log_dir="${run_dir}/logs"

cd "${repo_dir}"
source agent_space/jupiter-env.sh

[[ -s "${token_file}" ]] || { echo "missing ${token_file}" >&2; exit 1; }
api_key="$(<"${token_file}")"
export VLLM_API_KEY="${api_key}"
export CUDA_VISIBLE_DEVICES=0,1,2,3
# V1 runner: the V2 runner has no routed-expert capturer.
unset VLLM_USE_V2_MODEL_RUNNER
export VLLM_SERVER_DEV_MODE=1
export VLLM_ENFORCE_STRICT_TOOL_CALLING=false

# Caches on fscratch: GPFS is slow for the many small files these produce.
# Separate from the prod c4 caches because the runner and parallelism differ,
# so the compiled graphs are not interchangeable.
cache_root=/e/fscratch/profound/${USER:-$(id -un)}/caches
export VLLM_CACHE_ROOT=${cache_root}/marlin/vllm-cache-glm53-cap
export TRTLLM_DG_CACHE_DIR=${cache_root}/marlin/trtllm-dg-glm53-cap
export TRITON_CACHE_DIR=${cache_root}/triton
export TORCHINDUCTOR_CACHE_DIR=${cache_root}/inductor
export FLASHINFER_CACHE_DIR=${cache_root}/flashinfer

# One .npy per request (the GLM-5.2 live capture wrote 154 files for an hour of
# real use), but they are small and numerous, so keep them off GPFS.
export VLLM_ROUTING_TRACE_DIR="${VLLM_ROUTING_TRACE_DIR:-${cache_root}/routes/glm53-cc-${SLURM_JOB_ID}}"
export VLLM_ROUTING_TRACE_VERIFICATION_SIZE=4

model_name=GLM-5.3-W4A16
model_fscratch="/e/fscratch/profound/${USER:-$(id -un)}/models/${model_name}"
expected_shards=282
actual_shards="$(ls "${model_fscratch}"/*.safetensors 2>/dev/null | wc -l)"
if [[ ! -d "${model_fscratch}" ]] || (( actual_shards != expected_shards )); then
  echo "model: ${model_fscratch} missing or incomplete" \
       "(${actual_shards}/${expected_shards} shards)" >&2
  exit 1
fi
export TIERED_MOE_MODEL_PATH="${model_fscratch}"
echo "model: ${TIERED_MOE_MODEL_PATH} (${actual_shards} shards)"
echo "routes: ${VLLM_ROUTING_TRACE_DIR}"

# Same placement as prod. Placement does not affect routing -- the gate picks
# the same experts wherever they live -- so this is here to keep the capture
# host's decode speed usable, not because the data depends on it.
export TIERED_MOE_PLACEMENT_PROFILE="${repo_dir}/agent_space/profiles/glm53-w4a16-2496.json"
export TIERED_MOE_HBM_RESERVE_GB=6
export TIERED_MOE_COMPILATION_CONFIG='{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[4,8,12,16],"compile_sizes":[4,8,12,16],"cudagraph_num_of_warmups":1,"pass_config":{"fuse_allreduce_rms":false}}'

mkdir -p "${run_dir}" "${log_dir}" "${VLLM_CACHE_ROOT}" "${TRTLLM_DG_CACHE_DIR}" \
  "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${FLASHINFER_CACHE_DIR}" \
  "${VLLM_ROUTING_TRACE_DIR}"
rm -f "${ready_file}" "${endpoint_file}"

server_pid=
cleanup() {
  [[ -n "${server_pid}" ]] && kill "${server_pid}" 2>/dev/null || true
  rm -f "${ready_file}" "${endpoint_file}"
}
trap cleanup EXIT

printf '%s\n' "${VLLM_ROUTING_TRACE_DIR}" >"${run_dir}/routes-${SLURM_JOB_ID}"

agent_space/experiments/2026-07-17-end-to-end-tuning/run-server.sh \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --max-num-seqs 4 \
  --enable-return-routed-experts \
  --host 0.0.0.0 \
  --port "${port}" \
  --gpu-memory-utilization 0.94 \
  --kv-cache-memory 21689598771 \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --generation-config vllm \
  --served-model-name glm53-w4a16-tiered \
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

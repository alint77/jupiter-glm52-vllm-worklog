#!/usr/bin/env bash
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --output=agent_space/experiments/2026-07-26-claude-routing-capture/slurm-%x-%j.out
#SBATCH --error=agent_space/experiments/2026-07-26-claude-routing-capture/slurm-%x-%j.err

set -euo pipefail

label="${1:?configuration label is required}"
repo_dir=/e/project1/profound/alint77/vllm
result_dir="${repo_dir}/agent_space/experiments/2026-07-26-claude-routing-capture"
model="$(dirname -- "${repo_dir}")/models/GLM-5.2-AutoRound-W4G64-MTP-e1ba887"
prompts="${repo_dir}/agent_space/experiments/2026-07-19-c1q4-placement/prompts.jsonl"
trace_dir="/e/scratch/profound/naeimitabiei1/claude-placement-profiles/${label}-${SLURM_JOB_ID}"

case "${label}" in
  control)
    profile="${repo_dir}/agent_space/experiments/2026-07-26-autoround-w4g64/hybrid-p0.5-profile.json"
    ;;
  balanced-owners-frequency)
    profile="${result_dir}/claude-balanced-owners-frequency-profile.json"
    ;;
  *)
    echo "Unknown configuration: ${label}" >&2
    exit 2
    ;;
esac

cd "${repo_dir}"
source agent_space/jupiter-env.sh

export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_SERVER_DEV_MODE=1
export VLLM_CACHE_ROOT=/e/scratch/profound/naeimitabiei1/vllm-cache-claude-routing-capture
export TRTLLM_DG_CACHE_DIR=/e/scratch/profound/naeimitabiei1/trtllm-dg-claude-routing-capture
export TIERED_MOE_MODEL_PATH="${model}"
export TIERED_MOE_PLACEMENT_PROFILE="${profile}"
export VLLM_TIERED_MOE_PROFILE_CAP=0
export TIERED_MOE_HBM_RESERVE_GB=7
export TIERED_MOE_COMPILATION_CONFIG='{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[4,8,12,16],"compile_sizes":[4,8,12,16],"cudagraph_num_of_warmups":1,"pass_config":{"fuse_allreduce_rms":false}}'
profiler_config="{\"profiler\":\"torch\",\"torch_profiler_dir\":\"${trace_dir}\",\"torch_profiler_with_stack\":false,\"torch_profiler_record_shapes\":true,\"torch_profiler_use_gzip\":false,\"ignore_frontend\":true,\"delay_iterations\":50,\"max_iterations\":8}"
mkdir -p "${VLLM_CACHE_ROOT}" "${TRTLLM_DG_CACHE_DIR}" "${trace_dir}"
printf '%s\n' "${trace_dir}" >"${result_dir}/profile-c4-${label}-trace-dir.txt"

prefix="profile-c4-${label}"
agent_space/experiments/2026-07-17-end-to-end-tuning/run-server.sh \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --decode-context-parallel-size 4 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.85 \
  --no-enable-prefix-caching \
  --profiler-config "${profiler_config}" \
  >"${result_dir}/${prefix}-server.out" \
  2>"${result_dir}/${prefix}-server.err" &
server_pid=$!
cleanup() {
  kill "${server_pid}" 2>/dev/null || true
}
trap cleanup EXIT

ready=false
for _ in $(seq 1 720); do
  if curl -fsS http://127.0.0.1:8027/health >/dev/null 2>&1; then
    ready=true
    break
  fi
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    tail -80 "${result_dir}/${prefix}-server.err"
    exit 1
  fi
  sleep 10
done
[[ "${ready}" == true ]]

bench_args=(
  --backend openai
  --base-url http://127.0.0.1:8027
  --endpoint /v1/completions
  --model glm52-w4a16-tiered
  --served-model-name glm52-w4a16-tiered
  --tokenizer "${model}"
  --dataset-name custom
  --dataset-path "${prompts}"
  --custom-output-len 256
  --skip-chat-template
  --disable-shuffle
  --num-prompts 24
  --max-concurrency 4
  --request-rate inf
  --temperature 0
  --ignore-eos
  --disable-tqdm
)

.venv/bin/vllm bench serve "${bench_args[@]}"
curl -fsS -X POST http://127.0.0.1:8027/start_profile >/dev/null
.venv/bin/vllm bench serve "${bench_args[@]}" \
  --save-result \
  --save-detailed \
  --result-dir "${result_dir}" \
  --result-filename "${prefix}.json"
curl -fsS -X POST http://127.0.0.1:8027/stop_profile >/dev/null
find "${trace_dir}" -maxdepth 1 -type f -printf '%f %s\n' \
  | sort >"${result_dir}/${prefix}-trace-files.txt"

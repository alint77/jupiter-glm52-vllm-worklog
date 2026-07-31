#!/usr/bin/env bash
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --output=agent_space/experiments/2026-07-31-replicated-expert-scheduling/slurm-%x-%j.out
#SBATCH --error=agent_space/experiments/2026-07-31-replicated-expert-scheduling/slurm-%x-%j.err

set -euo pipefail

assignment="${1:?assignment is required: off|greedy}"
concurrency="${2:?concurrency is required: 1|4}"
mode="${3:-benchmark}"
case "${assignment}" in
  off | greedy) ;;
  *) echo "Unknown assignment: ${assignment}" >&2; exit 2 ;;
esac
case "${concurrency}" in
  1)
    graph_sizes=4
    dcp_args=()
    ;;
  4)
    graph_sizes=4,8,12,16
    dcp_args=(--decode-context-parallel-size 4)
    ;;
  *) echo "Unsupported concurrency: ${concurrency}" >&2; exit 2 ;;
esac
case "${mode}" in
  benchmark) output_len=256; num_prompts=16 ;;
  profile) output_len=128; num_prompts=4 ;;
  *) echo "Unsupported mode: ${mode}" >&2; exit 2 ;;
esac

repo=/e/project1/profound/alint77/vllm
result_dir="${repo}/agent_space/experiments/2026-07-31-replicated-expert-scheduling"
model=/e/fscratch/profound/naeimitabiei1/models/GLM-5.2-AutoRound-W4G64-MTP-e1ba887
profile="${result_dir}/runtime-placements/replicas-985.json"
prompts="${repo}/agent_space/experiments/2026-07-29-marlin-smem-monopoly/agentic-prompts.jsonl"
tag_prefix="${mode}"
if [[ "${mode}" == benchmark ]]; then
  tag_prefix=runtime
fi
tag="${tag_prefix}-${SLURM_JOB_ID}-${assignment}-c${concurrency}"
profiler_args=()
if [[ "${mode}" == profile ]]; then
  trace_dir="/e/scratch/profound/naeimitabiei1/replica-${tag}"
  profiler_config="{\"profiler\":\"torch\",\"torch_profiler_dir\":\"${trace_dir}\",\"torch_profiler_with_stack\":false,\"torch_profiler_record_shapes\":true,\"ignore_frontend\":true,\"delay_iterations\":2,\"max_iterations\":8}"
  profiler_args=(--profiler-config "${profiler_config}")
  mkdir -p "${trace_dir}"
  printf '%s\n' "${trace_dir}" >"${result_dir}/${tag}-trace-path.txt"
fi

cd "${repo}"
source agent_space/jupiter-env.sh
unset VLLM_USE_V2_MODEL_RUNNER

export VLLM_SERVER_DEV_MODE=1
export VLLM_CACHE_ROOT="/e/project1/profound/alint77/.replica-caches/vllm-${assignment}-c${concurrency}"
export TRTLLM_DG_CACHE_DIR="/e/project1/profound/alint77/.replica-caches/trtllm-${assignment}-c${concurrency}"
export TIERED_MOE_MODEL_PATH="${model}"
export TIERED_MOE_PLACEMENT_PROFILE="${profile}"
export VLLM_TIERED_MOE_PROFILE_CAP=1
export TIERED_MOE_HBM_RESERVE_GB=7
export TIERED_MOE_COMPILATION_CONFIG="{\"mode\":3,\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"cudagraph_capture_sizes\":[${graph_sizes}],\"compile_sizes\":[${graph_sizes}],\"cudagraph_num_of_warmups\":1,\"pass_config\":{\"fuse_allreduce_rms\":false}}"
mkdir -p "${VLLM_CACHE_ROOT}" "${TRTLLM_DG_CACHE_DIR}"

agent_space/experiments/2026-07-17-end-to-end-tuning/run-server.sh \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --tiered-moe-replica-assignment "${assignment}" \
  --tiered-moe-host-reserve-gb 16 \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs "${concurrency}" \
  --no-enable-prefix-caching \
  "${dcp_args[@]}" \
  "${profiler_args[@]}" \
  >"${result_dir}/${tag}-server.out" \
  2>"${result_dir}/${tag}-server.err" &
server_pid=$!
monitor_pid=
cleanup() {
  if [[ -n "${monitor_pid}" ]]; then
    kill "${monitor_pid}" 2>/dev/null || true
  fi
  kill "${server_pid}" 2>/dev/null || true
}
trap cleanup EXIT

ready=false
for _ in $(seq 1 900); do
  if curl -fsS http://127.0.0.1:8027/health >/dev/null 2>&1; then
    ready=true
    break
  fi
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    tail -100 "${result_dir}/${tag}-server.err"
    exit 1
  fi
  sleep 5
done
[[ "${ready}" == true ]]

nvidia-smi \
  --query-gpu=timestamp,index,memory.total,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader,nounits \
  --loop=2 \
  >"${result_dir}/${tag}-vram.csv" &
monitor_pid=$!

bench_args=(
  --backend openai
  --base-url http://127.0.0.1:8027
  --endpoint /v1/completions
  --model glm52-w4a16-tiered
  --served-model-name glm52-w4a16-tiered
  --tokenizer "${model}"
  --dataset-name custom
  --dataset-path "${prompts}"
  --custom-output-len "${output_len}"
  --skip-chat-template
  --disable-shuffle
  --num-prompts "${num_prompts}"
  --max-concurrency "${concurrency}"
  --request-rate inf
  --temperature 0
  --ignore-eos
  --disable-tqdm
  --save-detailed
)

curl -fsS http://127.0.0.1:8027/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm52-w4a16-tiered","prompt":"The capital of France is","max_tokens":8,"temperature":0,"seed":0}' \
  >"${result_dir}/${tag}-smoke.json"
.venv/bin/vllm bench serve "${bench_args[@]}" >/dev/null
if [[ "${mode}" == profile ]]; then
  curl -fsS -X POST http://127.0.0.1:8027/start_profile >/dev/null
  .venv/bin/vllm bench serve "${bench_args[@]}" \
    --save-result \
    --result-dir "${result_dir}" \
    --result-filename "${tag}.json"
  curl -fsS -X POST http://127.0.0.1:8027/stop_profile >/dev/null
  exit 0
fi
curl -fsS http://127.0.0.1:8027/metrics >"${result_dir}/${tag}-metrics-before.txt"
for repeat in 1 2; do
  .venv/bin/vllm bench serve "${bench_args[@]}" \
    --save-result \
    --result-dir "${result_dir}" \
    --result-filename "${tag}-r${repeat}.json"
done
curl -fsS http://127.0.0.1:8027/metrics >"${result_dir}/${tag}-metrics-after.txt"

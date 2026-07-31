#!/usr/bin/env bash
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --job-name=replica-samenode
#SBATCH --output=agent_space/experiments/2026-07-31-replicated-expert-scheduling/slurm-%x-%j.out
#SBATCH --error=agent_space/experiments/2026-07-31-replicated-expert-scheduling/slurm-%x-%j.err

set -euo pipefail

concurrency="${1:?concurrency is required: 1|4}"
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

repo=/e/project1/profound/alint77/vllm
result_dir="${repo}/agent_space/experiments/2026-07-31-replicated-expert-scheduling"
model=/e/fscratch/profound/naeimitabiei1/models/GLM-5.2-AutoRound-W4G64-MTP-e1ba887
profile="${result_dir}/runtime-placements/replicas-985.json"
prompts="${repo}/agent_space/experiments/2026-07-29-marlin-smem-monopoly/agentic-prompts.jsonl"

cd "${repo}"
source agent_space/jupiter-env.sh
echo "node: $(hostname)"

for assignment in off greedy; do
  tag="samenode-${SLURM_JOB_ID}-${assignment}-c${concurrency}"
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
    --custom-output-len 256
    --skip-chat-template
    --disable-shuffle
    --num-prompts 16
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
  for repeat in 1 2 3; do
    .venv/bin/vllm bench serve "${bench_args[@]}" \
      --save-result \
      --result-dir "${result_dir}" \
      --result-filename "${tag}-r${repeat}.json"
  done

  kill "${monitor_pid}" 2>/dev/null || true
  monitor_pid=
  kill "${server_pid}" 2>/dev/null || true
  wait "${server_pid}" 2>/dev/null || true
  trap - EXIT
  sleep 20
done

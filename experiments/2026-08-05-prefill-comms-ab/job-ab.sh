#!/usr/bin/env bash
# Prefill A/B/C on one node: NCCL protocol and chunk size.
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --job-name=prefill-comms-ab
#SBATCH --output=agent_space/experiments/2026-08-05-prefill-comms-ab/slurm-%x-%j.out
#SBATCH --error=agent_space/experiments/2026-08-05-prefill-comms-ab/slurm-%x-%j.err

set -euo pipefail
repo_dir=/e/project1/profound/alint77/vllm
result_dir="${repo_dir}/agent_space/experiments/2026-08-05-prefill-comms-ab"
prompts="${repo_dir}/agent_space/experiments/2026-08-05-pytorch-16k-c1-c4/prompts.jsonl"
trace_root="/e/project1/profound/alint77/traces/prefill-comms-ab-${SLURM_JOB_ID}"
cd "${repo_dir}"
source agent_space/jupiter-env.sh
model=/e/fscratch/profound/naeimitabiei1/models/GLM-5.2-AutoRound-W4G64-MTP-e1ba887
[[ -d "${model}" ]] || model="${repo_dir}/../models/GLM-5.2-AutoRound-W4G64-MTP-e1ba887"
mkdir -p "${trace_root}"
echo "node: $(hostname)"
echo "trace root: ${trace_root}"

# arm | NCCL_PROTO | max_num_batched_tokens
arms=(
  "baseline::8192"
  "proto-simple:Simple:8192"
  "chunk16k::16384"
  "chunk16k-simple:Simple:16384"
)

for entry in "${arms[@]}"; do
  IFS=':' read -r label proto chunk <<<"${entry}"
  echo ""
  echo "################ arm ${label} (NCCL_PROTO='${proto}' chunk=${chunk})"

  unset NCCL_PROTO
  [[ -n "${proto}" ]] && export NCCL_PROTO="${proto}"

  export VLLM_USE_V2_MODEL_RUNNER=1
  export VLLM_SERVER_DEV_MODE=1
  # One cache root for every arm, not one per arm. NCCL_PROTO does not affect
  # compilation at all and Inductor keys on the graph hash, so the arms can
  # share; per-arm roots multiply the file count and /e/project1 is
  # inode-limited, which is what killed job 1245137. Reuse the prod-profile
  # root, already warm for this exact server config.
  export VLLM_CACHE_ROOT="/e/project1/profound/alint77/.marlin-caches/vllm-cache-prod-profile"
  export TRTLLM_DG_CACHE_DIR="/e/project1/profound/alint77/.marlin-caches/trtllm-dg-prod-profile"
  export TIERED_MOE_MODEL_PATH="${model}"
  export TIERED_MOE_PLACEMENT_PROFILE="${repo_dir}/agent_space/experiments/2026-07-31-replica-scheduling-v2/hybrid-p0.5-replicas-985.json"
  export VLLM_TIERED_MOE_PROFILE_CAP=0
  export TIERED_MOE_HBM_RESERVE_GB=7
  export TIERED_MOE_COMPILATION_CONFIG='{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[4,8,12,16],"compile_sizes":[4,8,12,16],"cudagraph_num_of_warmups":1,"pass_config":{"fuse_allreduce_rms":false}}'
  mkdir -p "${VLLM_CACHE_ROOT}" "${TRTLLM_DG_CACHE_DIR}"

  profiler_config="{\"profiler\":\"torch\",\"torch_profiler_dir\":\"${trace_root}\",\"torch_profiler_with_stack\":false,\"torch_profiler_record_shapes\":true,\"ignore_frontend\":true}"

  agent_space/experiments/2026-07-17-end-to-end-tuning/run-server.sh \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
    --decode-context-parallel-size 4 \
    --tiered-moe-replica-assignment exact \
    --max-num-seqs 4 \
    --max-num-batched-tokens "${chunk}" \
    --no-enable-prefix-caching \
    --profiler-config "${profiler_config}" \
    >"${result_dir}/${label}-server.out" \
    2>"${result_dir}/${label}-server.err" &
  server_pid=$!
  cleanup() { kill "${server_pid}" 2>/dev/null || true; }
  trap cleanup EXIT

  ready=false
  for _ in $(seq 1 400); do
    if curl -fsS http://127.0.0.1:8027/health >/dev/null 2>&1; then ready=true; break; fi
    if ! kill -0 "${server_pid}" 2>/dev/null; then
      echo "ARM ${label} FAILED TO START"; tail -40 "${result_dir}/${label}-server.err"; break
    fi
    sleep 5
  done
  if [[ "${ready}" != true ]]; then
    kill "${server_pid}" 2>/dev/null || true; wait "${server_pid}" 2>/dev/null || true
    continue
  fi

  # Correctness gate.
  curl -fsS http://127.0.0.1:8027/v1/completions -H 'Content-Type: application/json' \
    -d '{"model":"glm52-w4a16-tiered","prompt":"The capital of France is","max_tokens":8,"temperature":0}' \
    >"${result_dir}/${label}-semantic.json"

  # Unprofiled prefill timing: 16K in, 1 token out, so TTFT is the prefill.
  # One warmup pass is discarded, then two measured repetitions.
  bench_args=(
    --backend openai --base-url http://127.0.0.1:8027 --endpoint /v1/completions
    --model glm52-w4a16-tiered --served-model-name glm52-w4a16-tiered
    --tokenizer "${model}" --dataset-name custom --dataset-path "${prompts}"
    --custom-output-len 1 --disable-shuffle --num-prompts 5
    --max-concurrency 1 --request-rate inf --temperature 0 --ignore-eos --disable-tqdm
  )
  .venv/bin/vllm bench serve "${bench_args[@]}" >/dev/null 2>&1 || true
  for repeat in 1 2; do
    .venv/bin/vllm bench serve "${bench_args[@]}" \
      --save-result --result-dir "${result_dir}" \
      --result-filename "${label}-ttft-r${repeat}.json" \
      2>&1 | grep -E "Mean TTFT|Median TTFT|P99 TTFT|Total input tokens"
  done

  # One profiled prefill for the bucket and protocol breakdown.
  .venv/bin/python "${result_dir}/capture_prefill.py" \
    --prompts "${prompts}" --trace-root "${trace_root}" --label "${label}"

  kill "${server_pid}" 2>/dev/null || true
  wait "${server_pid}" 2>/dev/null || true
  trap - EXIT
  sleep 20
done

echo ""
echo "=== done ==="

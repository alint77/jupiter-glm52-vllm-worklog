#!/usr/bin/env bash
# Same-node c1/q4 MTP3 profiler capture with the shared-memory fix off and on.
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --job-name=marlin-smem-profile
#SBATCH --output=agent_space/experiments/2026-07-29-marlin-smem-monopoly/slurm-%x-%j.out
#SBATCH --error=agent_space/experiments/2026-07-29-marlin-smem-monopoly/slurm-%x-%j.err

set -euo pipefail

repo_dir=/e/project1/profound/alint77/vllm
result_dir="${repo_dir}/agent_space/experiments/2026-07-29-marlin-smem-monopoly"
model=/e/fscratch/profound/naeimitabiei1/models/GLM-5.2-AutoRound-W4G64-MTP-e1ba887
placement="${repo_dir}/agent_space/experiments/2026-07-26-autoround-w4g64/hybrid-p0.5-profile.json"
prompts="${repo_dir}/agent_space/experiments/2026-07-19-c1q4-placement/prompts.jsonl"
trace_root="/e/project1/profound/alint77/traces/marlin-smem-profile-${SLURM_JOB_ID}"

cd "${repo_dir}"
source agent_space/jupiter-env.sh
mkdir -p "${trace_root}"
printf '%s\n' "${trace_root}" \
  >"${result_dir}/profile-ab-trace-path-${SLURM_JOB_ID}.txt"
echo "node: $(hostname)"
echo "trace root: ${trace_root}"

for label in off on; do
  case "${label}" in
    off) tight=0 ;;
    on) tight=1 ;;
  esac
  tag="profile-${SLURM_JOB_ID}-${label}"
  trace_dir="${trace_root}/${label}"
  profiler_config="{\"profiler\":\"torch\",\"torch_profiler_dir\":\"${trace_dir}\",\"torch_profiler_with_stack\":false,\"torch_profiler_record_shapes\":true,\"ignore_frontend\":true,\"delay_iterations\":2,\"max_iterations\":12}"

  unset VLLM_USE_V2_MODEL_RUNNER
  export VLLM_SERVER_DEV_MODE=1
  export VLLM_CACHE_ROOT="/e/project1/profound/alint77/.marlin-caches/vllm-cache-marlin-profile-${label}"
  export TRTLLM_DG_CACHE_DIR="/e/project1/profound/alint77/.marlin-caches/trtllm-dg-marlin-profile-${label}"
  export TIERED_MOE_MODEL_PATH="${model}"
  export TIERED_MOE_PLACEMENT_PROFILE="${placement}"
  export VLLM_TIERED_MOE_PROFILE_CAP=0
  export VLLM_TIERED_MOE_TIGHT_SMEM="${tight}"
  export TIERED_MOE_HBM_RESERVE_GB=5
  export TIERED_MOE_COMPILATION_CONFIG='{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[4],"compile_sizes":[4],"cudagraph_num_of_warmups":1,"pass_config":{"fuse_allreduce_rms":false}}'
  mkdir -p "${trace_dir}" "${VLLM_CACHE_ROOT}" "${TRTLLM_DG_CACHE_DIR}"

  agent_space/experiments/2026-07-17-end-to-end-tuning/run-server.sh \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
    --profiler-config "${profiler_config}" \
    --gpu-memory-utilization 0.85 \
    --no-enable-prefix-caching \
    >"${result_dir}/${tag}-server.out" \
    2>"${result_dir}/${tag}-server.err" &
  server_pid=$!
  cleanup() { kill "${server_pid}" 2>/dev/null || true; }
  trap cleanup EXIT

  ready=false
  for _ in $(seq 1 360); do
    if curl -fsS http://127.0.0.1:8027/health >/dev/null 2>&1; then
      ready=true
      break
    fi
    if ! kill -0 "${server_pid}" 2>/dev/null; then
      tail -80 "${result_dir}/${tag}-server.err"
      exit 1
    fi
    sleep 5
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
    --custom-output-len 128
    --skip-chat-template
    --disable-shuffle
    --num-prompts 1
    --max-concurrency 1
    --request-rate inf
    --temperature 0
    --ignore-eos
    --disable-tqdm
  )

  .venv/bin/vllm bench serve "${bench_args[@]}" >/dev/null
  curl -fsS -X POST http://127.0.0.1:8027/start_profile >/dev/null
  .venv/bin/vllm bench serve "${bench_args[@]}" \
    --save-result \
    --result-dir "${result_dir}" \
    --result-filename "${tag}.json"
  curl -fsS -X POST http://127.0.0.1:8027/stop_profile >/dev/null

  find "${trace_dir}" -type f -name '*.trace.json.gz' -printf '%s %p\n'
  kill "${server_pid}" 2>/dev/null || true
  wait "${server_pid}" 2>/dev/null || true
  trap - EXIT
  sleep 30
done

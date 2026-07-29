#!/usr/bin/env bash
# Same-node control for the c1/q4 MTP3 A/B.
#
# The first A/B ran its arms on different nodes (1088069 on jpbo-049-30, 1088070
# on jpbo-101-13), so node-to-node variation could account for some or all of
# the +9.7%. This runs both arms sequentially inside one allocation.
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=03:00:00
#SBATCH --job-name=marlin-smem-samenode
#SBATCH --output=agent_space/experiments/2026-07-29-marlin-smem-monopoly/slurm-%x-%j.out
#SBATCH --error=agent_space/experiments/2026-07-29-marlin-smem-monopoly/slurm-%x-%j.err

set -euo pipefail

repo_dir=/e/project1/profound/alint77/vllm
result_dir="${repo_dir}/agent_space/experiments/2026-07-29-marlin-smem-monopoly"
model="$(dirname -- "${repo_dir}")/models/GLM-5.2-AutoRound-W4G64-MTP-e1ba887"
profile="${repo_dir}/agent_space/experiments/2026-07-26-autoround-w4g64/hybrid-p0.5-profile.json"
prompts="${repo_dir}/agent_space/experiments/2026-07-19-c1q4-placement/prompts.jsonl"

cd "${repo_dir}"
source agent_space/jupiter-env.sh

echo "node: $(hostname)"

for label in off on; do
  case "${label}" in
    on) tight=1 ;;
    off) tight=0 ;;
  esac
  tag="samenode-${label}"

  unset VLLM_USE_V2_MODEL_RUNNER
  export VLLM_SERVER_DEV_MODE=1
  export VLLM_CACHE_ROOT="/e/project1/profound/alint77/.marlin-caches/vllm-cache-marlin-sn-${label}"
  export TRTLLM_DG_CACHE_DIR="/e/project1/profound/alint77/.marlin-caches/trtllm-dg-marlin-sn-${label}"
  export TIERED_MOE_MODEL_PATH="${model}"
  export TIERED_MOE_PLACEMENT_PROFILE="${profile}"
  export VLLM_TIERED_MOE_PROFILE_CAP=0
  export VLLM_TIERED_MOE_TIGHT_SMEM="${tight}"
  export TIERED_MOE_HBM_RESERVE_GB=5
  export TIERED_MOE_COMPILATION_CONFIG='{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[4],"compile_sizes":[4],"cudagraph_num_of_warmups":1,"pass_config":{"fuse_allreduce_rms":false}}'
  mkdir -p "${VLLM_CACHE_ROOT}" "${TRTLLM_DG_CACHE_DIR}"

  agent_space/experiments/2026-07-17-end-to-end-tuning/run-server.sh \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
    --gpu-memory-utilization 0.85 \
    --no-enable-prefix-caching \
    >"${result_dir}/${tag}-server.out" \
    2>"${result_dir}/${tag}-server.err" &
  server_pid=$!
  cleanup() { kill "${server_pid}" 2>/dev/null || true; }
  trap cleanup EXIT

  ready=false
  for _ in $(seq 1 720); do
    if curl -fsS http://127.0.0.1:8027/health >/dev/null 2>&1; then
      ready=true
      break
    fi
    if ! kill -0 "${server_pid}" 2>/dev/null; then
      tail -60 "${result_dir}/${tag}-server.err"
      exit 1
    fi
    sleep 10
  done
  [[ "${ready}" == true ]]

  curl -fsS http://127.0.0.1:8027/v1/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"glm52-w4a16-tiered","prompt":"The capital of France is","max_tokens":8,"temperature":0,"seed":0}' \
    >"${result_dir}/${tag}-smoke.json"
  .venv/bin/python "${result_dir}/check_smoke.py" \
    "${result_dir}/${tag}-smoke.json" "${tag}"

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
    --max-concurrency 1
    --request-rate inf
    --temperature 0
    --ignore-eos
    --disable-tqdm
  )

  .venv/bin/vllm bench serve "${bench_args[@]}" >/dev/null
  for repeat in 1 2; do
    .venv/bin/vllm bench serve "${bench_args[@]}" \
      --save-result \
      --result-dir "${result_dir}" \
      --result-filename "${tag}-r${repeat}.json"
  done

  kill "${server_pid}" 2>/dev/null || true
  wait "${server_pid}" 2>/dev/null || true
  trap - EXIT
  sleep 30
done

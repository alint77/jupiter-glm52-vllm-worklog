#!/usr/bin/env bash
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --time-min=00:10:00
#SBATCH --job-name=glm52-c4-mtp3-mixed
#SBATCH --output=agent_space/experiments/2026-07-29-c4-mtp3-mixed-suite/slurm-%j.out
#SBATCH --error=agent_space/experiments/2026-07-29-c4-mtp3-mixed-suite/slurm-%j.err

set -euo pipefail

repo_dir=/e/project1/profound/alint77/vllm
result_dir="${repo_dir}/agent_space/experiments/2026-07-29-c4-mtp3-mixed-suite"
# Submission order is the whole point of the interleaved arm: `--disable-shuffle`
# means file order is submission order, and a domain-grouped file makes the last
# domain the draining one.
prompt_file="${PROMPT_FILE:-prompts-interleaved.jsonl}"
tag="${RESULT_TAG:-mixed-c4-interleaved}"
model=/e/fscratch/profound/naeimitabiei1/models/GLM-5.2-AutoRound-W4G64-MTP-e1ba887
profile="${repo_dir}/agent_space/experiments/2026-07-26-autoround-w4g64/hybrid-p0.5-profile.json"

cd "${repo_dir}"
source agent_space/jupiter-env.sh

export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_SERVER_DEV_MODE=1
export VLLM_CACHE_ROOT=/e/project1/profound/alint77/.marlin-caches/vllm-cache-c4-mixed
export TRTLLM_DG_CACHE_DIR=/e/project1/profound/alint77/.marlin-caches/trtllm-dg-c4-mixed
export TIERED_MOE_MODEL_PATH="${model}"
export TIERED_MOE_PLACEMENT_PROFILE="${profile}"
export VLLM_TIERED_MOE_PROFILE_CAP=0
export VLLM_TIERED_MOE_TIGHT_SMEM=1
export TIERED_MOE_HBM_RESERVE_GB=7
export TIERED_MOE_COMPILATION_CONFIG='{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[4,8,12,16],"compile_sizes":[4,8,12,16],"cudagraph_num_of_warmups":1,"pass_config":{"fuse_allreduce_rms":false}}'
mkdir -p "${VLLM_CACHE_ROOT}" "${TRTLLM_DG_CACHE_DIR}"

agent_space/experiments/2026-07-17-end-to-end-tuning/run-server.sh \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --decode-context-parallel-size 4 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.85 \
  --no-enable-prefix-caching \
  >"${result_dir}/${tag}-server.out" \
  2>"${result_dir}/${tag}-server.err" &
server_pid=$!
cleanup() {
  kill "${server_pid}" 2>/dev/null || true
}
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

curl -fsS http://127.0.0.1:8027/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm52-w4a16-tiered","prompt":"The capital of France is","max_tokens":8,"temperature":0}' \
  >"${result_dir}/${tag}-semantic.json"
jq -e '.choices[0].text | length > 0' "${result_dir}/${tag}-semantic.json" >/dev/null

bench_args=(
  --backend openai
  --base-url http://127.0.0.1:8027
  --endpoint /v1/completions
  --model glm52-w4a16-tiered
  --served-model-name glm52-w4a16-tiered
  --tokenizer "${model}"
  --dataset-name custom
  --dataset-path "${result_dir}/${prompt_file}"
  --custom-output-len 256
  --disable-shuffle
  --num-prompts 18
  --max-concurrency 4
  --request-rate inf
  --temperature 0
  --ignore-eos
  --disable-tqdm
)

.venv/bin/vllm bench serve "${bench_args[@]}"
for repeat in 1 2 3; do
  curl -fsS -X POST http://127.0.0.1:8027/reset_prefix_cache >/dev/null
  .venv/bin/vllm bench serve "${bench_args[@]}" \
    --save-result \
    --save-detailed \
    --result-dir "${result_dir}" \
    --result-filename "${tag}-r${repeat}.json"
done

.venv/bin/python "${result_dir}/analyze.py" "${tag}" "${prompt_file}" \
  >"${result_dir}/summary-${tag}.out"

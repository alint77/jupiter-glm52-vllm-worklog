#!/usr/bin/env bash
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --job-name=autoround-ab
#SBATCH --output=agent_space/experiments/2026-07-26-autoround-w4g64/slurm-%x-%j.out
#SBATCH --error=agent_space/experiments/2026-07-26-autoround-w4g64/slurm-%x-%j.err

set -euo pipefail

repo_dir=/e/project1/profound/alint77/vllm
result_dir="${repo_dir}/agent_space/experiments/2026-07-26-autoround-w4g64"
model="$(dirname -- "${repo_dir}")/models/GLM-5.2-AutoRound-W4G64-MTP-e1ba887"
profile="${result_dir}/hybrid-p0.5-profile.json"
prompts="${repo_dir}/agent_space/experiments/2026-07-19-c1q4-placement/prompts.jsonl"

cd "${repo_dir}"
source agent_space/jupiter-env.sh

export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_SERVER_DEV_MODE=1
export VLLM_CACHE_ROOT="/e/scratch/profound/naeimitabiei1/vllm-cache-${SLURM_JOB_ID}"
export TRTLLM_DG_CACHE_DIR="/e/scratch/profound/naeimitabiei1/trtllm-dg-${SLURM_JOB_ID}"
export TIERED_MOE_MODEL_PATH="${model}"
export TIERED_MOE_PLACEMENT_PROFILE="${profile}"
export VLLM_TIERED_MOE_PROFILE_CAP=1
export TIERED_MOE_HBM_RESERVE_GB=10
export TIERED_MOE_COMPILATION_CONFIG='{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[4],"compile_sizes":[4],"cudagraph_num_of_warmups":1,"pass_config":{"fuse_allreduce_rms":false}}'
mkdir -p "${VLLM_CACHE_ROOT}" "${TRTLLM_DG_CACHE_DIR}"

agent_space/experiments/2026-07-17-end-to-end-tuning/run-server.sh \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --no-enable-prefix-caching \
  >"${result_dir}/benchmark-server.out" \
  2>"${result_dir}/benchmark-server.err" &
server_pid=$!
trap 'kill "${server_pid}" 2>/dev/null || true' EXIT

ready=false
for _ in $(seq 1 720); do
  if curl -fsS http://127.0.0.1:8027/health >/dev/null 2>&1; then
    ready=true
    break
  fi
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    tail -80 "${result_dir}/benchmark-server.err"
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
  --max-concurrency 1
  --request-rate inf
  --temperature 0
  --ignore-eos
  --disable-tqdm
)

.venv/bin/vllm bench serve "${bench_args[@]}"
for repeat in 1 2; do
  .venv/bin/vllm bench serve "${bench_args[@]}" \
    --save-result \
    --save-detailed \
    --result-dir "${result_dir}" \
    --result-filename "benchmark-r${repeat}.json"
done

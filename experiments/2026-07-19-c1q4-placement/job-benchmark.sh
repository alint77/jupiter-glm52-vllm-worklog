#!/usr/bin/env bash
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --output=agent_space/experiments/2026-07-19-c1q4-placement/slurm-%x-%j.out
#SBATCH --error=agent_space/experiments/2026-07-19-c1q4-placement/slurm-%x-%j.err

set -euo pipefail

label="${1:?label is required}"
placement_profile="${2:?placement profile is required}"

cd /e/project1/profound/alint77/vllm
source agent_space/jupiter-env.sh

result_dir=agent_space/experiments/2026-07-19-c1q4-placement
model=/e/project1/profound/alint77/models/GLM-5.2-W4A16-FP8-MTP

export VLLM_CACHE_ROOT="/e/scratch/profound/naeimitabiei1/vllm-cache-${SLURM_JOB_ID}"
export FLASHINFER_WORKSPACE_BASE="/e/scratch/profound/naeimitabiei1/flashinfer-${SLURM_JOB_ID}"
export TRTLLM_DG_CACHE_DIR="/e/scratch/profound/naeimitabiei1/trtllm-deepgemm-${SLURM_JOB_ID}"
export TIERED_MOE_PLACEMENT_PROFILE="${placement_profile}"
export TIERED_MOE_HBM_RESERVE_GB=10

agent_space/experiments/2026-07-18-dcp-port/run-server.sh \
  3 1 FULL_AND_PIECEWISE "" \
  >"${result_dir}/${label}-server.out" \
  2>"${result_dir}/${label}-server.err" &
server_pid=$!
trap 'kill "${server_pid}" 2>/dev/null || true' EXIT

ready=false
for _ in $(seq 1 240); do
  if curl -fsS http://127.0.0.1:8027/health >/dev/null; then
    ready=true
    break
  fi
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    wait "${server_pid}"
  fi
  sleep 10
done
[[ "${ready}" == true ]]

curl -fsS http://127.0.0.1:8027/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm52-w4a16-tiered","prompt":"The capital of France is","max_tokens":8,"temperature":0,"logprobs":1}' \
  -o "${result_dir}/${label}-semantic.json"

.venv/bin/vllm bench serve \
  --backend openai \
  --base-url http://127.0.0.1:8027 \
  --endpoint /v1/completions \
  --model glm52-w4a16-tiered \
  --served-model-name glm52-w4a16-tiered \
  --tokenizer "${model}" \
  --dataset-name custom \
  --dataset-path "${result_dir}/prompts.jsonl" \
  --custom-output-len 256 \
  --skip-chat-template \
  --disable-shuffle \
  --num-prompts 24 \
  --max-concurrency 1 \
  --request-rate inf \
  --temperature 0 \
  --ignore-eos \
  --disable-tqdm \
  --save-result \
  --save-detailed \
  --result-dir "${result_dir}" \
  --result-filename "${label}.json"

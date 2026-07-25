#!/usr/bin/env bash

set -euo pipefail

label="${1:?result label is required}"
result_dir=agent_space/experiments/2026-07-25-dflash
model=/e/project1/profound/alint77/models/GLM-5.2-W4A16-FP8-MTP
prompts=agent_space/experiments/2026-07-19-c1q4-placement/prompts.jsonl

source agent_space/jupiter-env.sh

curl -fsS http://127.0.0.1:8027/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm52-w4a16-tiered","prompt":"The capital of France is","max_tokens":8,"temperature":0,"logprobs":1}' \
  -o "${result_dir}/${label}-semantic.json"
test "$(jq -r '.choices[0].text' "${result_dir}/${label}-semantic.json")" = \
  " Paris. Distance from Paris to Lyon is"

reset_cache() {
  curl -fsS -X POST http://127.0.0.1:8027/reset_prefix_cache |
    jq -e '.success' >/dev/null
}

run_random() {
  local suffix="$1"
  local conc="${2:-4}"
  local input_len="$2"
  local output_len="$3"
  local seed="$4"
  local prompts_count="$5"
  local concurrency="$6"
  .venv/bin/vllm bench serve \
    --backend openai \
    --base-url http://127.0.0.1:8027 \
    --endpoint /v1/completions \
    --model glm52-w4a16-tiered \
    --served-model-name glm52-w4a16-tiered \
    --tokenizer "${model}" \
    --dataset-name random \
    --input-len "${input_len}" \
    --output-len "${output_len}" \
    --seed "${seed}" \
    --num-prompts "${prompts_count}" \
    --max-concurrency "${concurrency}" \
    --request-rate inf \
    --temperature 0 \
    --ignore-eos \
    --disable-tqdm \
    --save-result \
    --save-detailed \
    --result-dir "${result_dir}" \
    --result-filename "${label}-${suffix}.json"
}

run_realistic() {
  local suffix="$1"
  local conc="${2:-4}"
  .venv/bin/vllm bench serve \
    --backend openai \
    --base-url http://127.0.0.1:8027 \
    --endpoint /v1/completions \
    --model glm52-w4a16-tiered \
    --served-model-name glm52-w4a16-tiered \
    --tokenizer "${model}" \
    --dataset-name custom \
    --dataset-path "${prompts}" \
    --custom-output-len 256 \
    --skip-chat-template \
    --disable-shuffle \
    --num-prompts 24 \
    --max-concurrency "${conc}" \
    --request-rate inf \
    --temperature 0 \
    --ignore-eos \
    --disable-tqdm \
    --save-result \
    --save-detailed \
    --result-dir "${result_dir}" \
    --result-filename "${label}-${suffix}.json"
}

run_random warmup-4k-16-c1 4096 16 17 1 1

for repeat in 1 2; do
  run_realistic "realistic-c1-r${repeat}" 1
done

run_random 4k-256-c1-r1 4096 256 13 1 1
run_random 4k-256-c1-r2 4096 256 13 1 1

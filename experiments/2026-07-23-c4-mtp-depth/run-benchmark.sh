#!/usr/bin/env bash

set -euo pipefail

label="${1:?result label is required}"
result_dir=agent_space/experiments/2026-07-23-c4-mtp-depth
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
    --max-concurrency 4 \
    --request-rate inf \
    --temperature 0 \
    --ignore-eos \
    --disable-tqdm \
    --save-result \
    --save-detailed \
    --result-dir "${result_dir}" \
    --result-filename "${label}-${suffix}.json"
}

run_random warmup-4k-16-c4 4096 16 17 4 4

for repeat in 1 2; do
  reset_cache
  run_realistic "realistic-c4-r${repeat}"
done

for repeat in 1 2; do
  reset_cache
  run_random "4k-256-c4-r${repeat}" 4096 256 17 4 4
done

reset_cache
run_random 399744-256-c1 399744 256 13 1 1
test "$(jq -j '.generated_texts[0]' \
  "${result_dir}/${label}-399744-256-c1.json" | sha256sum | cut -d' ' -f1)" = \
  d594e4d4268600a0d9e2d51355913907c638ef0c61615547598615384dcfc528

reset_cache
run_random prime-395904-1-c4 395904 1 17 4 1
curl -fsS -X POST http://127.0.0.1:8027/start_profile >/dev/null
run_random 395904-1024-c4 395904 1024 17 4 4
curl -fsS -X POST http://127.0.0.1:8027/stop_profile >/dev/null

#!/usr/bin/env bash

set -euo pipefail

label="${1:?result label is required}"

source agent_space/jupiter-env.sh

result_dir=agent_space/experiments/2026-07-18-dcp-port

curl -fsS http://127.0.0.1:8027/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm52-w4a16-tiered","prompt":"The capital of France is","max_tokens":8,"temperature":0,"logprobs":1}' \
  -o "${result_dir}/${label}-semantic.json"

run_case() {
  .venv/bin/vllm bench serve \
    --backend openai \
    --base-url http://127.0.0.1:8027 \
    --endpoint /v1/completions \
    --model glm52-w4a16-tiered \
    --served-model-name glm52-w4a16-tiered \
    --tokenizer "${GLM52_W4A16_MODEL}" \
    --dataset-name random \
    --input-len "$2" \
    --output-len "$3" \
    --seed "$4" \
    --num-prompts "$5" \
    --max-concurrency "$5" \
    --request-rate inf \
    --temperature 0 \
    --ignore-eos \
    --disable-tqdm \
    --save-result \
    --save-detailed \
    --result-dir "${result_dir}" \
    --result-filename "$1.json"
}

# c=4 cases use seed 17 so their prompts are disjoint from the seed-13
# golden prompt (no prefix-cache cross-talk with the SHA gate).
run_case "${label}-warmup-4k-16-c4" 4096 16 17 4
run_case "${label}-4k-256-c4" 4096 256 17 4
run_case "${label}-399744-256-c1" 399744 256 13 1
run_case "${label}-399744-256-c4" 399744 256 17 4

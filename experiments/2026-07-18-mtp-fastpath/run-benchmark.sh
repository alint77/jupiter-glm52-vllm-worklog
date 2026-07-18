#!/usr/bin/env bash

set -euo pipefail

label="${1:?result label is required}"
profile="${2:-false}"

source agent_space/jupiter-env.sh

result_dir=agent_space/experiments/2026-07-18-mtp-fastpath

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
    --seed 13 \
    --num-prompts 1 \
    --max-concurrency 1 \
    --request-rate inf \
    --temperature 0 \
    --ignore-eos \
    --disable-tqdm \
    --save-result \
    --save-detailed \
    --result-dir "${result_dir}" \
    --result-filename "$1.json"
}

run_case "${label}-warmup-4k-16" 4096 16
run_case "${label}-4k-256" 4096 256
run_case "${label}-399744-256" 399744 256

if [[ "${profile}" == true ]]; then
  curl -fsS -X POST http://127.0.0.1:8027/start_profile
  run_case "${label}-profiled-399744-256" 399744 256
  curl -fsS -X POST http://127.0.0.1:8027/stop_profile
fi

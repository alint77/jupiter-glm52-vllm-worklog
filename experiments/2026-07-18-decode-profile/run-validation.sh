#!/usr/bin/env bash

set -euo pipefail

source agent_space/jupiter-env.sh

result_dir=agent_space/experiments/2026-07-18-decode-profile

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

curl -fsS http://127.0.0.1:8027/v1/completions \
  -H 'Content-Type: application/json' \
  --data '{"model":"glm52-w4a16-tiered","prompt":"The capital of France is","max_tokens":8,"temperature":0,"logprobs":1}' \
  > "${result_dir}/final-semantic-smoke.json"

run_case final-warmup-4k-16 4096 16 10
run_case final-4k-256 4096 256 13
run_case final-399744-256 399744 256 13
run_case final-seed14-399744-256 399744 256 14

nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader \
  > "${result_dir}/final-memory.csv"

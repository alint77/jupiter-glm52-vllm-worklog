#!/usr/bin/env bash

set -euo pipefail

source agent_space/jupiter-env.sh

result_prefix="${RESULT_PREFIX:-}"
full_seed="${FULL_SEED:-13}"

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
    --result-dir agent_space/experiments/2026-07-17-end-to-end-tuning \
    --result-filename "$1.json"
}

run_case "${result_prefix}warmup-4k-16" 4096 16 10
run_case "${result_prefix}profiled-399744-256" 399744 256 "${full_seed}"

#!/usr/bin/env bash

set -euo pipefail

source agent_space/jupiter-env.sh

result_dir=agent_space/experiments/2026-07-18-mtp-prompt-profile
categories=("$@")
if (( ${#categories[@]} == 0 )); then
  categories=(python pytorch cuda-cpp math email explanation)
fi

for category in "${categories[@]}"; do
  .venv/bin/vllm bench serve \
    --backend openai \
    --base-url http://127.0.0.1:8027 \
    --endpoint /v1/completions \
    --model glm52-w4a16-tiered \
    --served-model-name glm52-w4a16-tiered \
    --tokenizer "${GLM52_W4A16_MODEL}" \
    --dataset-name custom \
    --dataset-path "${result_dir}/${category}.jsonl" \
    --custom-output-len 256 \
    --num-prompts 3 \
    --max-concurrency 1 \
    --request-rate inf \
    --temperature 0 \
    --disable-tqdm \
    --save-result \
    --save-detailed \
    --result-dir "${result_dir}" \
    --result-filename "${category}.json"
done

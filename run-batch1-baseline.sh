#!/usr/bin/env bash

set -euo pipefail

source "$(dirname -- "${BASH_SOURCE[0]}")/jupiter-env.sh"

server_host="${1:-jpbo-021-29}"
result_dir="${VLLM_REPO_DIR}/agent_space/baseline"

run_case() {
  local name="$1"
  local input_len="$2"
  local output_len="$3"
  local seed="$4"

  "${VLLM_REPO_DIR}/.venv/bin/vllm" bench serve \
    --backend openai \
    --base-url "http://${server_host}:8000" \
    --endpoint /v1/completions \
    --model glm52-w4a16 \
    --served-model-name glm52-w4a16 \
    --tokenizer "${GLM52_W4A16_MODEL}" \
    --dataset-name random \
    --input-len "${input_len}" \
    --output-len "${output_len}" \
    --seed "${seed}" \
    --num-prompts 1 \
    --max-concurrency 1 \
    --request-rate inf \
    --temperature 0 \
    --ignore-eos \
    --disable-tqdm \
    --save-result \
    --save-detailed \
    --result-dir "${result_dir}" \
    --result-filename "${name}.json"
}

run_case warmup-4k-16out 4096 16 10
run_case batch1-4k-256out 4096 256 11
run_case batch1-32k-256out 32768 256 12
run_case batch1-399744-256out 399744 256 13

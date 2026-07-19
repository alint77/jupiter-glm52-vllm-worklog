#!/usr/bin/env bash

set -euo pipefail

label="${1:?result label is required}"

source agent_space/jupiter-env.sh

result_dir=agent_space/experiments/2026-07-18-dcp-port

curl -fsS http://127.0.0.1:8027/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm52-w4a16-tiered","prompt":"The capital of France is","max_tokens":8,"temperature":0,"logprobs":1}' \
  -o "${result_dir}/${label}-semantic.json"
test "$(jq -r '.choices[0].text' "${result_dir}/${label}-semantic.json")" = \
  " Paris. Distance from Paris to Lyon is"

run_case() {
  local result_path="${result_dir}/$1.json"
  local concurrency="${6:-$5}"
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
    --max-concurrency "${concurrency}" \
    --request-rate inf \
    --temperature 0 \
    --ignore-eos \
    --disable-tqdm \
    --save-result \
    --save-detailed \
    --result-dir "${result_dir}" \
    --result-filename "$1.json"

  jq -e --argjson outputs "$3" --argjson prompts "$5" \
    '.failed == 0 and .completed == $prompts and
     (.output_lens | length) == $prompts and
     all(.output_lens[]; . == $outputs)' \
    "${result_path}" >/dev/null
  curl -fsS http://127.0.0.1:8027/health >/dev/null
}

# c=4 cases use seed 17 so their prompts are disjoint from the seed-13
# golden prompt (no prefix-cache cross-talk with the SHA gate).
run_case "${label}-warmup-4k-16-c4" 4096 16 17 4
run_case "${label}-4k-256-c4" 4096 256 17 4
run_case "${label}-4k-256-sequential" 4096 256 17 4 1
test "$(jq -c '.generated_texts' "${result_dir}/${label}-4k-256-c4.json")" = \
  "$(jq -c '.generated_texts' "${result_dir}/${label}-4k-256-sequential.json")"
run_case "${label}-399744-256-c1" 399744 256 13 1
test "$(jq -j '.generated_texts[0]' \
  "${result_dir}/${label}-399744-256-c1.json" | sha256sum | cut -d' ' -f1)" = \
  d594e4d4268600a0d9e2d51355913907c638ef0c61615547598615384dcfc528
# 395,904 + 4,096 = exactly max_model_len: prefills serialize (~110 s
# each), so a long output leaves a real 4-way steady decode window after
# the last prefill; steady TPOT comes from the detailed ITLs.
run_case "${label}-395904-4096-c4" 395904 4096 17 4

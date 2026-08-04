#!/usr/bin/env bash
# Benchmark the running production server on 16 unique 16,384-token PyTorch
# code-generation prompts, 512 forced output tokens, at concurrency 1 and 4.
#
# Runs inside the server's own allocation so the client adds no network hop:
#   srun --jobid=<jobid> --overlap --ntasks=1 --cpu-bind=none \
#     bash agent_space/experiments/2026-08-05-pytorch-16k-c1-c4/run-benchmark.sh
set -euo pipefail

repo_dir=/e/project1/profound/alint77/vllm
cd "${repo_dir}"
result_dir="${repo_dir}/agent_space/experiments/2026-08-05-pytorch-16k-c1-c4"
base_url=http://127.0.0.1:8027
model=/e/fscratch/profound/naeimitabiei1/models/GLM-5.2-AutoRound-W4G64-MTP-e1ba887
[[ -d "${model}" ]] || model="${repo_dir}/../models/GLM-5.2-AutoRound-W4G64-MTP-e1ba887"

# The production host is authenticated; the bench client reads OPENAI_API_KEY.
OPENAI_API_KEY="$(cat "/e/scratch/profound/${USER:-$(id -un)}/claude-local-c4/api-token")"
export OPENAI_API_KEY
auth=(-H "Authorization: Bearer ${OPENAI_API_KEY}")

curl -fsS -m 10 "${base_url}/health" >/dev/null

# Correctness gate: the project's deterministic smoke continuation.
curl -fsS "${base_url}/v1/completions" "${auth[@]}" \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm52-w4a16-tiered","prompt":"The capital of France is","max_tokens":8,"temperature":0}' \
  >"${result_dir}/semantic.json"

snapshot() {
  curl -fsS "${base_url}/metrics" "${auth[@]}" \
    | grep -E '^vllm:spec_decode_num_(draft|accepted)_tokens_total|^vllm:(prompt|generation)_tokens_total' \
    >"${result_dir}/metrics-$1.txt"
}

bench() {
  local concurrency=$1 tag=$2
  .venv/bin/vllm bench serve \
    --backend openai \
    --base-url "${base_url}" \
    --endpoint /v1/completions \
    --model glm52-w4a16-tiered \
    --served-model-name glm52-w4a16-tiered \
    --tokenizer "${model}" \
    --dataset-name custom \
    --dataset-path "${result_dir}/prompts.jsonl" \
    --custom-output-len 512 \
    --disable-shuffle \
    --num-prompts 16 \
    --max-concurrency "${concurrency}" \
    --request-rate inf \
    --temperature 0 \
    --ignore-eos \
    --disable-tqdm \
    "${@:3}"
}

echo "=== warmup (excluded) ==="
bench 4 warmup >/dev/null

for concurrency in 1 4; do
  for repeat in 1 2; do
    tag="c${concurrency}-r${repeat}"
    echo "=== ${tag} ==="
    curl -fsS -X POST "${base_url}/reset_prefix_cache" "${auth[@]}" >/dev/null
    snapshot "${tag}-before"
    bench "${concurrency}" "${tag}" \
      --save-result --save-detailed \
      --result-dir "${result_dir}" \
      --result-filename "${tag}.json"
    snapshot "${tag}-after"
  done
done

echo "=== done ==="

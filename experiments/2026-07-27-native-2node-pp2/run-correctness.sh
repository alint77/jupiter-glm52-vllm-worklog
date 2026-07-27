#!/usr/bin/env bash
# Losslessness gate for the 2-node native PP2 runs. Run from the login node
# against a job's leader node (node 0) after /health is up.
#
# Verifies, against the single-node native baseline's saved outputs:
#   1. semantic smoke: "The capital of France is" -> " Paris. Distance from Paris..."
#   2. 4K/seed-11/256  generated-text SHA == 692e494f...
#   3. 400K/seed-13/256 generated-text SHA == d594e4d4... (the project golden SHA)
#
# A mismatch means PP2 (or DCP4) perturbed greedy numerics -> investigate
# before trusting any throughput number.
#
# Usage: run-correctness.sh <server_host> <result_dir> [served_model_name]

set -euo pipefail

server_host="${1:?server_host required}"
result_dir="${2:?result_dir required}"
model="${3:-glm52-w4a16}"
mkdir -p "${result_dir}"

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." 2>/dev/null && pwd)"
[[ -f "${repo_dir}/agent_space/jupiter-env.sh" ]] || repo_dir=/e/project1/profound/alint77/vllm
cd "${repo_dir}"
source agent_space/jupiter-env.sh

base_url="http://${server_host}:8000"
expected_semantic=" Paris. Distance from Paris to Lyon is"
sha_4k="692e494fea186991f7755d4e44e93a1df7628c22ce8eeb22b764ea0257405203"
sha_400k="d594e4d4268600a0d9e2d51355913907c638ef0c61615547598615384dcfc528"

echo "[correctness] semantic smoke"
curl -fsS "${base_url}/v1/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"${model}\",\"prompt\":\"The capital of France is\",\"max_tokens\":8,\"temperature\":0,\"seed\":13}" \
  -o "${result_dir}/semantic.json"
got="$(jq -r '.choices[0].text' "${result_dir}/semantic.json")"
echo "  got:      ${got}"
echo "  expected: ${expected_semantic}"
[ "${got}" = "${expected_semantic}" ] || { echo "  SEMANTIC MISMATCH"; exit 1; }
echo "  PASS"

rand_case() {
  local name="$1" ilen="$2" seed="$3" want="$4"
  echo "[correctness] ${name} (input ${ilen}, seed ${seed})"
  "${VLLM_VENV_DIR:-${PWD}/.venv}/bin/vllm" bench serve \
    --backend openai --base-url "${base_url}" --endpoint /v1/completions \
    --model "${model}" --served-model-name "${model}" \
    --tokenizer "${GLM52_W4A16_MODEL}" \
    --dataset-name random --input-len "${ilen}" --output-len 256 \
    --seed "${seed}" --num-prompts 1 --max-concurrency 1 --request-rate inf \
    --temperature 0 --ignore-eos --disable-tqdm \
    --save-result --save-detailed \
    --result-dir "${result_dir}" --result-filename "${name}.json" >/dev/null 2>&1
  gotsha="$(.venv/bin/python -c "import json,hashlib; d=json.load(open('${result_dir}/${name}.json')); t=d['generated_texts'][0]; t=''.join(t) if isinstance(t,list) else t; print(hashlib.sha256(str(t).encode()).hexdigest())")"
  echo "  got:      ${gotsha}"
  echo "  expected: ${want}"
  [ "${gotsha}" = "${want}" ] || { echo "  ${name} SHA MISMATCH"; exit 1; }
  echo "  PASS"
}

rand_case correctness-4k 4096 11 "${sha_4k}"
rand_case correctness-400k 399744 13 "${sha_400k}"

echo "[correctness] ALL GATES PASSED"

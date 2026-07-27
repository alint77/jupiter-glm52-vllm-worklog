#!/usr/bin/env bash
# c4 throughput client for the 2-node native PP2 run. Run from the login node
# against a job's leader node after run-correctness.sh has passed.
#
# Four concurrent 400K agents (DCP4) is the project's "c4" regime. Measures
# aggregate output throughput; per-agent = aggregate / 4.
#
# Usage: run-bench-c4.sh <server_host> <result_dir> [served_model_name]

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
vllm="${VLLM_VENV_DIR:-${PWD}/.venv}/bin/vllm"

run_c4() {
  local name="$1" ilen="$2" olen="$3" seed="$4"
  echo "[c4] ${name}: 4 concurrent, input ${ilen}, output ${olen}"
  "${vllm}" bench serve \
    --backend openai --base-url "${base_url}" --endpoint /v1/completions \
    --model "${model}" --served-model-name "${model}" \
    --tokenizer "${GLM52_W4A16_MODEL}" \
    --dataset-name random --input-len "${ilen}" --output-len "${olen}" \
    --seed "${seed}" --num-prompts 4 --max-concurrency 4 --request-rate inf \
    --temperature 0 --ignore-eos --disable-tqdm \
    --save-result --save-detailed \
    --result-dir "${result_dir}" --result-filename "${name}.json"
}

# warmup (short output, fills the 4-seq pipeline)
run_c4 warmup-4k-16-c4 4096 16 17

# short-context c4
run_c4 4k-256-c4-r1 4096 256 11
run_c4 4k-256-c4-r2 4096 256 11

# the headline: 4 x exact-400K agents
run_c4 400k-256-c4-r1 399744 256 13
run_c4 400k-256-c4-r2 399744 256 13

echo "[c4] done. Aggregate output_throughput (tok/s) per file:"
for f in warmup-4k-16-c4 4k-256-c4-r1 4k-256-c4-r2 400k-256-c4-r1 400k-256-c4-r2; do
  .venv/bin/python -c "import json; d=json.load(open('${result_dir}/${f}.json')); print(f'  ${f}: out {d[\"output_throughput\"]:.2f} tok/s  total {d[\"total_token_throughput\"]:.2f}  mean_tpot {d[\"mean_tpot_ms\"]:.2f} ms  completed {d[\"completed\"]}')" 2>/dev/null || true
done

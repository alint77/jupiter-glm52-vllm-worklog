#!/usr/bin/env bash
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --output=agent_space/experiments/2026-07-25-dspark/slurm-%x-%j.out
#SBATCH --error=agent_space/experiments/2026-07-25-dspark/slurm-%x-%j.err

set -euo pipefail

label="${1:?result label is required}"
spec_tokens="${2:-8}"
repo_dir=/e/project1/profound/alint77/vllm
result_dir="${repo_dir}/agent_space/experiments/2026-07-25-dspark"

cd "${repo_dir}"
source agent_space/jupiter-env.sh

# DSpark is V2-runner only (gpu/model_runner.py gates eagle3/dflash/dspark).
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_SERVER_DEV_MODE=1
export VLLM_CACHE_ROOT="/e/scratch/profound/naeimitabiei1/vllm-cache-${SLURM_JOB_ID}"
export TRTLLM_DG_CACHE_DIR="/e/scratch/profound/naeimitabiei1/trtllm-dg-${SLURM_JOB_ID}"
mkdir -p "${VLLM_CACHE_ROOT}" "${TRTLLM_DG_CACHE_DIR}"

agent_space/experiments/2026-07-25-dspark/run-server-dspark.sh \
  "${spec_tokens}" 1 "" \
  >"${result_dir}/${label}-server.out" \
  2>"${result_dir}/${label}-server.err" &
server_pid=$!
trap 'kill "${server_pid}" 2>/dev/null || true' EXIT

ready=false
for _ in $(seq 1 420); do
  if curl -fsS http://127.0.0.1:8027/health >/dev/null 2>&1; then ready=true; break; fi
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    echo "SERVER EXITED EARLY"; tail -40 "${result_dir}/${label}-server.err"; exit 1
  fi
  sleep 10
done
[[ "${ready}" == true ]] || { echo "SERVER NOT READY"; tail -40 "${result_dir}/${label}-server.err"; exit 1; }
echo "server ready"

# Gate 1: the project's deterministic semantic smoke.
curl -fsS http://127.0.0.1:8027/v1/completions -H 'Content-Type: application/json' -d '{
  "model":"glm52-w4a16-tiered",
  "prompt":"The capital of France is","max_tokens":8,"temperature":0,"seed":13
}' -o "${result_dir}/${label}-semantic.json"
echo "--- semantic completion:"
jq -r '.choices[0].text' "${result_dir}/${label}-semantic.json"
echo "--- expected: ' Paris. Distance from Paris to Lyon is'"

# Gate 2: a longer greedy generation, to shake out block/Markov layout errors
# that a short completion can mask.
curl -fsS http://127.0.0.1:8027/v1/completions -H 'Content-Type: application/json' -d '{
  "model":"glm52-w4a16-tiered",
  "prompt":"Write a Python function that returns the n-th Fibonacci number.",
  "max_tokens":128,"temperature":0,"seed":13
}' -o "${result_dir}/${label}-python.json"
echo "--- 128-token greedy sample:"
jq -r '.choices[0].text' "${result_dir}/${label}-python.json" | head -20

echo "--- acceptance / throughput:"
grep -iE "acceptance|draft" "${result_dir}/${label}-server.out" | tail -12 || true

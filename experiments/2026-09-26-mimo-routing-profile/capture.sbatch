#!/usr/bin/env bash
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --job-name=mimo26-route-cap
#SBATCH --output=agent_space/experiments/2026-09-26-mimo-routing-profile/slurm-%j.out
#SBATCH --error=agent_space/experiments/2026-09-26-mimo-routing-profile/slurm-%j.err

# MiMo-V2.6 routing capture: the production server (serve.sbatch, DFlash k=7,
# linear placement) plus --enable-return-routed-experts, driven on the same
# node by the GLM capture's agentic coding loop over one task shard.
#
#   sbatch --export=ALL,SHARD=0 capture.sbatch     # shards 0..3
#
# DFlash k=7 verifies 8 tokens per step, hence VERIFICATION_SIZE=8. Only
# expert IDs are written. max_tokens is 8192 because capped generations come
# back as identity routes (2026-08-29 README) and MiMo reasons at length.
set -euo pipefail
cd /e/project1/profound/alint77/vllm
source agent_space/jupiter-env.sh
E=agent_space/experiments/2026-09-26-mimo-routing-profile
shard="${SHARD:?set SHARD}"
run="/e/fscratch/profound/${USER}/mimo26-route-cap/${SLURM_JOB_ID}"
mkdir -p "${run}"
port=8041
key="$(head -c 24 /dev/urandom | base64 | tr -dc A-Za-z0-9)"
export VLLM_API_KEY="${key}"
export VLLM_SERVER_DEV_MODE=1
export VLLM_ENFORCE_STRICT_TOOL_CALLING=false
export VLLM_ROUTING_TRACE_DIR="${run}/routes"
export VLLM_ROUTING_TRACE_VERIFICATION_SIZE=8
export MIMO_PORT="${port}"
export MIMO_EXTRA_ARGS="--language-model-only --enable-prefix-caching \
--enable-auto-tool-choice --tool-call-parser mimo --reasoning-parser mimo \
--enable-return-routed-experts \
--override-generation-config {\"max_new_tokens\":null}"
echo "shard ${shard} run ${run} node $(hostname -s)"

sed -e 's/^exec \.venv/.venv/' agent_space/experiments/2026-09-22-mimo-v26-pro/serve.sbatch \
  | bash >"${run}/server.out" 2>"${run}/server.err" &
server_pid=$!
trap 'kill ${server_pid} 2>/dev/null || true' EXIT

for _ in $(seq 1 180); do
  curl -fsS -H "Authorization: Bearer ${key}" "http://127.0.0.1:${port}/health" \
    >/dev/null 2>&1 && break
  kill -0 "${server_pid}" 2>/dev/null || { tail -40 "${run}/server.err"; exit 1; }
  sleep 10
done
echo "server ready at $(date +%T)"

.venv/bin/python agent_space/experiments/2026-08-29-glm53-routing-capture/run_agentic_capture.py \
  --base-url "http://127.0.0.1:${port}" --api-key "${key}" --model mimo26-pro \
  --tasks "${E}/tasks-${shard}.json" --output-dir "${run}/driver" \
  --sandbox "${run}/sandbox" --max-tokens 8192 --passes 4 || true
echo "driver exited at $(date +%T)"

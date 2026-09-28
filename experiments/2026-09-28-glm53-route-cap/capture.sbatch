#!/usr/bin/env bash
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --job-name=glm53-route-cap
#SBATCH --output=agent_space/experiments/2026-09-28-glm53-route-cap/slurm-%j.out
#SBATCH --error=agent_space/experiments/2026-09-28-glm53-route-cap/slurm-%j.err
# GLM-5.3 W4A16 routing capture on the same 16 agentic tasks, split the same
# four ways, driven the same way as MiMo's (../2026-09-26-mimo-routing-profile/
# capture.sbatch), so the two models' profiles come from the same workload.
#
#   sbatch --export=ALL,SHARD=0 capture.sbatch     # shards 0..3
#
# MTP7 verifies 8 tokens per step, hence VERIFICATION_SIZE=8 (as MiMo's DFlash
# k=7). Routed-expert return needs the V1 runner and no DCP, so this runs
# DCP1 / V1 with replicas and the new decode kernels off; none of that changes
# which experts are routed, only the speed of the capture.
set -euo pipefail
cd /e/project1/profound/alint77/vllm
source agent_space/jupiter-env.sh
M=agent_space/experiments/2026-09-26-mimo-routing-profile
S=agent_space/experiments/2026-09-27-glm53-mtp7-profile/serve.sh
shard="${SHARD:?set SHARD}"
run="/e/fscratch/profound/${USER}/glm53-route-cap/${SLURM_JOB_ID}"
mkdir -p "${run}"
export VLLM_ROUTING_TRACE_DIR="${run}/routes"
export VLLM_ROUTING_TRACE_VERIFICATION_SIZE=8
export VLLM_ENFORCE_STRICT_TOOL_CALLING=false
export VLLM_USE_V2_MODEL_RUNNER=0 DCP=1 REPLICAS= VLLM_TIERED_MOE_DECODE_KERNEL=0 \
  VLLM_DCP_ONE_SHOT_COLLECTIVES=0
export SERVE_EXTRA="--enable-return-routed-experts --enable-prefix-caching \
--enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45"
echo "shard ${shard} run ${run} node $(hostname -s)"
bash "${S}" >"${run}/server.out" 2>"${run}/server.err" &
server_pid=$!
trap 'kill ${server_pid} 2>/dev/null || true' EXIT
for _ in $(seq 1 240); do
  curl -fsS "http://127.0.0.1:8027/health" >/dev/null 2>&1 && break
  kill -0 "${server_pid}" 2>/dev/null || { tail -40 "${run}/server.err"; exit 1; }
  sleep 10
done
echo "server ready at $(date +%T)"
.venv/bin/python agent_space/experiments/2026-08-29-glm53-routing-capture/run_agentic_capture.py \
  --base-url "http://127.0.0.1:8027" --api-key none --model glm53-w4a16-tiered \
  --tasks "${M}/tasks-${shard}.json" --output-dir "${run}/driver" \
  --sandbox "${run}/sandbox" --max-tokens 8192 --passes 4 || true
echo "driver exited at $(date +%T)"

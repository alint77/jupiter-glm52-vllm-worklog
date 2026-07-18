#!/usr/bin/env bash
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --output=agent_space/experiments/2026-07-18-dcp-port/slurm-%x-%j.out
#SBATCH --error=agent_space/experiments/2026-07-18-dcp-port/slurm-%x-%j.err

# Boot one DCP4 FULL-graph server variant and report whether CUDA graph
# capture completes. Usage: job-variant.sh <name> [ENV=VAL ...]

set -uo pipefail

name="${1:?variant name is required}"
shift

cd /e/project1/profound/alint77/vllm
source agent_space/jupiter-env.sh

export VLLM_CACHE_ROOT="/e/scratch/profound/naeimitabiei1/vllm-cache-${SLURM_JOB_ID}"
export FLASHINFER_WORKSPACE_BASE="/e/scratch/profound/naeimitabiei1/flashinfer-${SLURM_JOB_ID}"
export TRTLLM_DG_CACHE_DIR="/e/scratch/profound/naeimitabiei1/trtllm-deepgemm-${SLURM_JOB_ID}"

result_dir=agent_space/experiments/2026-07-18-dcp-port
server_out="${result_dir}/bisect-${name}-server.out"
server_err="${result_dir}/bisect-${name}-server.err"
echo "=== variant ${name}: env $* ===" >&2
env "$@" agent_space/experiments/2026-07-18-dcp-port/run-server.sh \
  3 4 FULL_AND_PIECEWISE "" \
  >"${server_out}" 2>"${server_err}" &
pid=$!
trap 'kill "${pid}" 2>/dev/null || true' EXIT

verdict=TIMEOUT
for _ in $(seq 1 360); do
  if curl -fsS http://127.0.0.1:8027/health >/dev/null 2>&1; then
    verdict=CAPTURE_OK
    break
  fi
  if ! kill -0 "${pid}" 2>/dev/null; then
    verdict=CRASHED
    break
  fi
  sleep 10
done
echo "=== variant ${name}: ${verdict} ===" >&2

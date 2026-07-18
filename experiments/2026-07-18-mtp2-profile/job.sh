#!/usr/bin/env bash
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --job-name=glm52-mtp2-profile
#SBATCH --output=agent_space/experiments/2026-07-18-mtp2-profile/slurm-%j.out
#SBATCH --error=agent_space/experiments/2026-07-18-mtp2-profile/slurm-%j.err

set -euo pipefail

cd /e/project1/profound/alint77/vllm
source agent_space/jupiter-env.sh

server_out=agent_space/experiments/2026-07-18-mtp2-profile/server.out
server_err=agent_space/experiments/2026-07-18-mtp2-profile/server.err
agent_space/experiments/2026-07-18-mtp2-profile/run-server.sh \
  >"${server_out}" 2>"${server_err}" &
server_pid=$!
trap 'kill "${server_pid}" 2>/dev/null || true' EXIT

ready=false
for _ in $(seq 1 240); do
  if curl -fsS http://127.0.0.1:8027/health >/dev/null; then
    ready=true
    break
  fi
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    wait "${server_pid}"
  fi
  sleep 10
done

if [[ "${ready}" != true ]]; then
  echo "Server did not become ready" >&2
  exit 1
fi

agent_space/experiments/2026-07-18-mtp2-profile/run-benchmark.sh

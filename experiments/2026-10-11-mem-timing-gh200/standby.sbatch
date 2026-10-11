#!/usr/bin/env bash
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --job-name=memtim-standby
#SBATCH --output=agent_space/experiments/2026-10-11-mem-timing-gh200/logs/standby-%j.out
#SBATCH --error=agent_space/experiments/2026-10-11-mem-timing-gh200/logs/standby-%j.err

# Standby allocation for the mem-timing research thread: hold one full
# GH200 node so ad-hoc probes can be launched into it with node-submit.sh
# (srun --jobid --overlap). No GPU work runs here.
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")/../../.." 2>/dev/null && pwd)"
[[ -f "${repo_dir}/agent_space/jupiter-env.sh" ]] || repo_dir=/e/project1/profound/alint77/vllm
cd "$repo_dir"
E=agent_space/experiments/2026-10-11-mem-timing-gh200
printf 'JOB_ID=%s\nNODE=%s\nSTART=%s\n' \
  "${SLURM_JOB_ID}" "$(hostname -s)" "$(date -Is)" >"${E}/alloc.node.tmp"
mv "${E}/alloc.node.tmp" "${E}/alloc.node"
nvidia-smi -L >"${E}/alloc.gpus.tmp"
mv "${E}/alloc.gpus.tmp" "${E}/alloc.gpus"
trap 'rm -f "${E}/alloc.node" "${E}/alloc.gpus"' EXIT
sleep 3540

#!/usr/bin/env bash
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --job-name=native-pp2-c4
#SBATCH --output=agent_space/experiments/2026-07-27-native-2node-pp2/slurm-c4-%j.out
#SBATCH --error=agent_space/experiments/2026-07-27-native-2node-pp2/slurm-c4-%j.err

# Native GLM-5.2 W4A16, 2 nodes, TP4 x PP2, DCP4 @ 400K, 4 concurrent (c4),
# V2 model runner, no offload. sbatch this file; it sruns run-server.sh per node.

set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." 2>/dev/null && pwd)"
[[ -f "${repo_dir}/agent_space/experiments/2026-07-27-native-2node-pp2/run-server.sh" ]] \
  || repo_dir=/e/project1/profound/alint77/vllm
exp_dir="${repo_dir}/agent_space/experiments/2026-07-27-native-2node-pp2"
cd "${repo_dir}"

export MODE=c4
export JOB_CACHE="/e/scratch/profound/${USER:-$(id -un)}/native-pp2-${SLURM_JOB_ID}"
mkdir -p "${JOB_CACHE}"
_first_node="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)"
export MASTER_ADDR="$(getent hosts "${_first_node}" | awk '{print $1}')"
export MASTER_ADDR="${MASTER_ADDR:-${_first_node}}"
export MASTER_PORT="$((29500 + SLURM_JOB_ID % 1000))"

echo "[$(date)] native-pp2 c4 job ${SLURM_JOB_ID} nodes=${SLURM_JOB_NODELIST} master=${MASTER_ADDR}:${MASTER_PORT}"

srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --cpu-bind=none \
  bash "${exp_dir}/run-server.sh"

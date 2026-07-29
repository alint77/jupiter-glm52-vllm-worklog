#!/usr/bin/env bash
# exa_project1 vs exa_fscratch for checkpoint loading, on a Booster node.
# Login-node I/O paths differ from compute; checkpoint load happens on Booster.
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --job-name=fs-bench
#SBATCH --output=agent_space/experiments/2026-07-29-marlin-smem-monopoly/slurm-%x-%j.out
#SBATCH --error=agent_space/experiments/2026-07-29-marlin-smem-monopoly/slurm-%x-%j.err

set -euo pipefail
repo_dir=/e/project1/profound/alint77/vllm
result_dir="${repo_dir}/agent_space/experiments/2026-07-29-marlin-smem-monopoly"
cd "${repo_dir}"
source agent_space/jupiter-env.sh

echo "node $(hostname)"
free -g | head -2
echo
.venv/bin/python "${result_dir}/analysis/fs_bench.py" \
  --out "${result_dir}/fs-bench-booster.json"

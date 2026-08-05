#!/usr/bin/env bash
# Sweep the Marlin launch configuration at the production prefill chunk shape.
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:40:00
#SBATCH --job-name=prefill-marlin
#SBATCH --output=agent_space/experiments/2026-08-05-prefill-marlin-launch/slurm-%x-%j.out
#SBATCH --error=agent_space/experiments/2026-08-05-prefill-marlin-launch/slurm-%x-%j.err

set -euo pipefail
repo_dir=/e/project1/profound/alint77/vllm
result_dir="${repo_dir}/agent_space/experiments/2026-08-05-prefill-marlin-launch"
cd "${repo_dir}"
source agent_space/jupiter-env.sh
echo "node: $(hostname)"

.venv/bin/python benchmarks/kernels/benchmark_moe_wna16_marlin_decode.py \
  --mode prefill \
  --prefill-tokens 512 2048 8192 \
  --prefill-bps 1 2 3 \
  --prefill-iters 20 \
  --output "${result_dir}/prefill-sweep-${SLURM_JOB_ID}.json"
echo "=== done ==="

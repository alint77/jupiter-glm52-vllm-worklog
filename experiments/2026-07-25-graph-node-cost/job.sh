#!/usr/bin/env bash
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:40:00
#SBATCH --output=agent_space/experiments/2026-07-25-graph-node-cost/slurm-%x-%j.out
#SBATCH --error=agent_space/experiments/2026-07-25-graph-node-cost/slurm-%x-%j.err

set -euo pipefail
repo_dir=/e/project1/profound/alint77/vllm
result_dir="${repo_dir}/agent_space/experiments/2026-07-25-graph-node-cost"
cd "${repo_dir}"
source agent_space/jupiter-env.sh
export VLLM_CACHE_ROOT="/e/scratch/profound/naeimitabiei1/vllm-cache-${SLURM_JOB_ID}"
mkdir -p "${VLLM_CACHE_ROOT}"

numactl --cpunodebind=0 --membind=0 .venv/bin/python \
  "${repo_dir}/benchmarks/kernels/benchmark_cudagraph_node_cost.py" \
  --output "${result_dir}/node-cost.json"

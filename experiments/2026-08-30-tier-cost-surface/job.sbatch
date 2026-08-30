#!/usr/bin/env bash
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --time=02:00:00
#SBATCH --job-name=tier-cost-surface
#SBATCH --output=agent_space/experiments/2026-08-30-tier-cost-surface/slurm-%j.out
#SBATCH --error=agent_space/experiments/2026-08-30-tier-cost-surface/slurm-%j.err

# Measure t(hot, cold, tokens) for the two Marlin tiers on Booster.
#
# Booster, not the login node: the login node prefers a different Marlin grid
# (66 CTAs against 132) and its C2C behaviour differs, so a cost surface fitted
# there would describe the wrong machine.
#
# numactl is load-bearing. GraceAllocation.allocate_pinned binds nothing -- its
# docstring requires the caller to already be NUMA-bound -- and production gets
# that from --numa-bind, which a bare sbatch does not. Without this the cold
# tier's weights land at 0% locality and the C2C figures are meaningless.

set -euo pipefail

repo_dir=/e/project1/profound/alint77/vllm
here="${repo_dir}/agent_space/experiments/2026-08-30-tier-cost-surface"
cd "${repo_dir}"
source agent_space/jupiter-env.sh

export CUDA_VISIBLE_DEVICES=0
export VLLM_CACHE_ROOT=/e/fscratch/profound/${USER:-$(id -un)}/caches/marlin/vllm-cache-tiercost
export TRITON_CACHE_DIR=/e/fscratch/profound/${USER:-$(id -un)}/caches/triton
mkdir -p "${VLLM_CACHE_ROOT}" "${TRITON_CACHE_DIR}"

echo "node: $(hostname -s)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

srun --nodes=1 --ntasks=1 --cpus-per-task=288 --gres=gpu:4 --mem=0 \
  numactl --cpunodebind=0 --membind=0 \
  "${repo_dir}/.venv/bin/python" \
  "${repo_dir}/agent_space/benchmarks/tier_cost_surface.py" \
  --numa-node 0 \
  --output "${here}/surface-${SLURM_JOB_ID}.json"

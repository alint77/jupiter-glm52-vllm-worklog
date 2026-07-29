#!/usr/bin/env bash
# Kernel-level A/B of the tight-shared-memory Marlin launch, on Booster.
# The login GH200 runs a different power cap and a slightly different C2C, so
# the hot/cold balance - and therefore the size of the overlap win - has to be
# measured here.
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:40:00
#SBATCH --job-name=marlin-smem-kernel-ab
#SBATCH --output=agent_space/experiments/2026-07-29-marlin-smem-monopoly/slurm-%x-%j.out
#SBATCH --error=agent_space/experiments/2026-07-29-marlin-smem-monopoly/slurm-%x-%j.err

set -euo pipefail

repo_dir=/e/project1/profound/alint77/vllm
result_dir="${repo_dir}/agent_space/experiments/2026-07-29-marlin-smem-monopoly"
cd "${repo_dir}"
source agent_space/jupiter-env.sh

export CUDA_DEVICE_MAX_CONNECTIONS=8
export VLLM_CACHE_ROOT="/e/project1/profound/alint77/.marlin-caches/vllm-cache-smem-${SLURM_JOB_ID}"
mkdir -p "${VLLM_CACHE_ROOT}"

# GPU0's paired Grace node is node-specific; never hardcode it.
numa_node="$(.venv/bin/python "${result_dir}/detect_numa.py" 0)"
echo "GPU0 paired NUMA node: ${numa_node}"

# Mechanism probes first: they are hardware-level and explain any A/B result.
echo "=== smem monopoly probe ==="
"${result_dir}/analysis/smem_monopoly" 20000000 || true
echo "=== two-tier union probe (HBM vs pinned Grace) ==="
"${result_dir}/analysis/tier_union" 4 || true

echo "=== in-tree kernel A/B ==="
# Strict NUMA binding, as the production server uses: without it the pinned
# Grace allocations intermittently land off-node and the locality audit fails.
numactl --cpunodebind="${numa_node}" --membind="${numa_node}" \
  .venv/bin/python "${result_dir}/analysis/kernel_ab.py" \
  --numa-node "${numa_node}" \
  --rounds 7 \
  --out "${result_dir}/booster-kernel-ab.json"

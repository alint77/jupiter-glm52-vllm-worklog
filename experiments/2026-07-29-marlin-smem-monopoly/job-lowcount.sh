#!/usr/bin/env bash
# Why the low activated-expert shapes show no benefit, and what fixes them.
#
# Everything here is timed under CUDA graph replay. Timing the two-tier fork/join
# eagerly charges two stream barriers per iteration, which on Booster costs
# ~110 us - larger than the entire kernel at these sizes, and enough to hide any
# difference. Production replays the fork/join from inside a graph.
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --job-name=marlin-smem-lowcount
#SBATCH --output=agent_space/experiments/2026-07-29-marlin-smem-monopoly/slurm-%x-%j.out
#SBATCH --error=agent_space/experiments/2026-07-29-marlin-smem-monopoly/slurm-%x-%j.err

set -euo pipefail

repo_dir=/e/project1/profound/alint77/vllm
result_dir="${repo_dir}/agent_space/experiments/2026-07-29-marlin-smem-monopoly"
cd "${repo_dir}"
source agent_space/jupiter-env.sh

export CUDA_DEVICE_MAX_CONNECTIONS=8
export VLLM_CACHE_ROOT="/e/project1/profound/alint77/.marlin-caches/vllm-cache-lowcount-${SLURM_JOB_ID}"
mkdir -p "${VLLM_CACHE_ROOT}"

numa_node="$(.venv/bin/python "${result_dir}/detect_numa.py" 0)"
echo "node $(hostname), GPU0 paired NUMA node ${numa_node}"

# The production c1/q4 point is ~5.9 activated experts per rank with ~1.1 cold.
for shape in "4 7 1" "4 6 2" "8 10 2" "8 13 3"; do
  set -- ${shape}
  echo ""
  echo "########## m=$1 hot=$2 cold=$3 ##########"
  numactl --cpunodebind="${numa_node}" --membind="${numa_node}" \
    .venv/bin/python "${result_dir}/analysis/lowcount_search.py" \
      --m "$1" --hot "$2" --cold "$3" \
      --numa-node "${numa_node}" \
      --out "${result_dir}/lowcount-m$1-h$2-c$3.json"
done

# and the full sweep again, now measured under graph replay
numactl --cpunodebind="${numa_node}" --membind="${numa_node}" \
  .venv/bin/python "${result_dir}/analysis/kernel_ab.py" \
    --numa-node "${numa_node}" \
    --rounds 5 \
    --out "${result_dir}/booster-kernel-ab-graph.json"

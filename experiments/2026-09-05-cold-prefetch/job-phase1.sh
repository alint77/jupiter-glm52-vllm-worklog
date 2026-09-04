#!/usr/bin/env bash
# Phase 1: is the Grace->HBM staging copy fast, and is it quiet?
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:40:00
#SBATCH --job-name=cpfphase1
#SBATCH --output=agent_space/experiments/2026-09-05-cold-prefetch/slurm-%x-%j.out
#SBATCH --error=agent_space/experiments/2026-09-05-cold-prefetch/slurm-%x-%j.err

set -euo pipefail
repo_dir=/e/project1/profound/alint77/vllm
result_dir="${repo_dir}/agent_space/experiments/2026-09-05-cold-prefetch"
cd "${repo_dir}"
source agent_space/jupiter-env.sh

export VLLM_CACHE_ROOT="/e/fscratch/profound/${USER:-$(id -un)}/caches/marlin/vllm-cache-cpf-${SLURM_JOB_ID}"
mkdir -p "${VLLM_CACHE_ROOT}"

# The GPU's own sysfs NUMA node is its HBM and has no CPUs; use the platform
# helper, as the tiered runtime does. Binding matters: an unbound pin_memory
# lands pages on a remote socket and the C2C run goes ~6x slow.
numa_node="$(.venv/bin/python agent_space/experiments/2026-07-29-marlin-smem-monopoly/detect_numa.py 0)"
echo "node $(hostname), GPU0 paired NUMA node ${numa_node}"

numactl --cpunodebind="${numa_node}" --membind="${numa_node}" \
  .venv/bin/python agent_space/benchmarks/cold_prefetch_dma.py \
    --numa-node "${numa_node}" \
    --out "${result_dir}/phase1-${SLURM_JOB_ID}.json"

echo "=== control: same run without NUMA binding ==="
.venv/bin/python agent_space/benchmarks/cold_prefetch_dma.py \
  --numa-node "${numa_node}" \
  --out "${result_dir}/phase1-unbound-${SLURM_JOB_ID}.json" || true
echo "=== done ==="

#!/usr/bin/env bash
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --output=agent_space/experiments/2026-07-25-tier-overlap/slurm-%x-%j.out
#SBATCH --error=agent_space/experiments/2026-07-25-tier-overlap/slurm-%x-%j.err

set -euo pipefail

repo_dir=/e/project1/profound/alint77/vllm
result_dir="${repo_dir}/agent_space/experiments/2026-07-25-tier-overlap"
bench="${repo_dir}/benchmarks/kernels/benchmark_moe_wna16_marlin_decode.py"

cd "${repo_dir}"
source agent_space/jupiter-env.sh
export VLLM_CACHE_ROOT="/e/scratch/profound/naeimitabiei1/vllm-cache-${SLURM_JOB_ID}"
mkdir -p "${VLLM_CACHE_ROOT}"

bind=(numactl --cpunodebind=0 --membind=0)
command -v numactl >/dev/null || bind=()

# Each stage is its own process: an invalid launch config can raise an async
# CUDA error that poisons the context uncatchably (see job 1040908), so a
# failure must not take the rest of the sweep with it.
for stage in a1 a2 a3 a4; do
  echo "=== stage ${stage} ==="
  "${bind[@]}" .venv/bin/python "${bench}" --mode "${stage}" \
    --tokens 16 --hot-act 19 --cold-act 3 --cold-share 0.13 \
    --bps 1 2 3 \
    --output "${result_dir}/phase-a-${stage}.json" || echo "stage ${stage} FAILED"
done

echo
echo "=== full phase-a in one process (grid control + all stages) ==="
"${bind[@]}" .venv/bin/python "${bench}" --mode phase-a \
  --tokens 16 --hot-act 19 --cold-act 3 --cold-share 0.13 --bps 1 2 3 \
  --output "${result_dir}/phase-a-full.json" || echo "phase-a FAILED"

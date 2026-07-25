#!/usr/bin/env bash
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --output=agent_space/experiments/2026-07-25-grace-bandwidth/slurm-%x-%j.out
#SBATCH --error=agent_space/experiments/2026-07-25-grace-bandwidth/slurm-%x-%j.err

set -euo pipefail

repo_dir=/e/project1/profound/alint77/vllm
result_dir="${repo_dir}/agent_space/experiments/2026-07-25-grace-bandwidth"
bench="${repo_dir}/benchmarks/kernels/benchmark_moe_wna16_marlin_decode.py"

cd "${repo_dir}"
source agent_space/jupiter-env.sh
export VLLM_CACHE_ROOT="/e/scratch/profound/naeimitabiei1/vllm-cache-${SLURM_JOB_ID}"
mkdir -p "${VLLM_CACHE_ROOT}"

bind=(numactl --cpunodebind=0 --membind=0)
command -v numactl >/dev/null || bind=()

echo "=== A: production cold share (13% of routing mass) ==="
"${bind[@]}" .venv/bin/python "${bench}" --mode both \
  --tokens 8 16 32 --cold-share 0.13 \
  --output "${result_dir}/share-013.json"

echo
echo "=== B: cold share swept — isolates re-streaming from C2C bandwidth ==="
for share in 0.05 0.25 0.50; do
  echo "--- cold_share=${share}"
  "${bind[@]}" .venv/bin/python "${bench}" --mode overlap \
    --tokens 16 --cold-share "${share}" --block-sizes 16 \
    --output "${result_dir}/share-${share}.json"
done

echo
echo "=== C: original probe regime (M=1 dense) for comparison ==="
"${bind[@]}" .venv/bin/python agent_space/benchmarks/marlin_grace_uva.py \
  --experts 8 --groups 300 --k 6144 --n 4096 || echo "legacy probe failed"

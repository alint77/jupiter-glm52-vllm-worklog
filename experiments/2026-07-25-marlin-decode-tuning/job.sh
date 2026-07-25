#!/usr/bin/env bash
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --output=agent_space/experiments/2026-07-25-marlin-decode-tuning/slurm-%x-%j.out
#SBATCH --error=agent_space/experiments/2026-07-25-marlin-decode-tuning/slurm-%x-%j.err

set -euo pipefail

repo_dir=/e/project1/profound/alint77/vllm
result_dir="${repo_dir}/agent_space/experiments/2026-07-25-marlin-decode-tuning"
bench="${repo_dir}/benchmarks/kernels/benchmark_moe_wna16_marlin_decode.py"

cd "${repo_dir}"
source agent_space/jupiter-env.sh

export VLLM_CACHE_ROOT="/e/scratch/profound/naeimitabiei1/vllm-cache-${SLURM_JOB_ID}"
mkdir -p "${VLLM_CACHE_ROOT}"

# Bind to the Grace node paired with GPU 0 so the cold tier's pinned pages land
# locally; the benchmark audits page placement and will report if they do not.
bind=(numactl --cpunodebind=0 --membind=0)
command -v numactl >/dev/null || bind=()

echo "=== smoke: tiny shapes, catches API errors before the real sweep ==="
"${bind[@]}" .venv/bin/python "${bench}" \
  --mode both --tokens 16 --activated 4 --experts 8 \
  --hot-experts 6 --cold-experts 4 --hot-act 4 --cold-act 2 \
  --hot-bps -1 3 --cold-bps -1 1

echo
echo "=== sweep: GLM-5.2 decode shapes ==="
"${bind[@]}" .venv/bin/python "${bench}" \
  --mode both \
  --tokens 8 12 16 32 \
  --activated 5 16 22 \
  --experts 64 \
  --hot-experts 40 --cold-experts 24 --hot-act 19 --cold-act 3 \
  --hot-bps -1 1 2 3 --cold-bps -1 1 2 3 \
  --output "${result_dir}/marlin-decode-tuning.json"

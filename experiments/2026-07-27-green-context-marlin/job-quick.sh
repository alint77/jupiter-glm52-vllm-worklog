#!/usr/bin/env bash
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:20:00
#SBATCH --job-name=gc-marlin-q
#SBATCH --output=agent_space/experiments/2026-07-27-green-context-marlin/slurm-%j.out
#SBATCH --error=agent_space/experiments/2026-07-27-green-context-marlin/slurm-%j.err

# Lean unbuffered diagnostic for the §4.5 Marlin green-context probe.
# Single split (cold=16 SM), python -u so phase prints flush live, low iters,
# 1 GPU, 15-min cap. Localizes whether the wall time is tier-build, solo, or the
# concurrent fork/join — the blind --sweep run (job 1069196) gave no signal.

set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." 2>/dev/null && pwd)"
[[ -f "${repo_dir}/agent_space/jupiter-env.sh" ]] || repo_dir=/e/project1/profound/alint77/vllm
exp="${repo_dir}/agent_space/experiments/2026-07-27-green-context-marlin"
cd "${repo_dir}"
source agent_space/jupiter-env.sh

export CUDA_DEVICE_MAX_CONNECTIONS=8
export CUDA_VISIBLE_DEVICES=0
export VLLM_CACHE_ROOT="/e/scratch/profound/naeimitabiei1/vllm-cache-gcq-${SLURM_JOB_ID}"
mkdir -p "${VLLM_CACHE_ROOT}" "${exp}/logs"

echo "[$(date)] quick-diag job ${SLURM_JOB_ID} on $(hostname -s)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | head -1
echo "numactl node2 size: $(numactl -H 2>/dev/null | grep 'node 2 size:' | head -1)"

python -u "${exp}/marlin_green_probe.py" \
  --cold-sm 16 --m 16 --hot-experts 19 --cold-experts 3 --iters 15 --numa-node 2 \
  --out "${exp}/results-quick-16.json" \
  2>&1 | tee "${exp}/logs/quick-16-$(date +%Y%m%d-%H%M%S).log"

echo "[$(date)] done; results:"
cat "${exp}/results-quick-16.json" 2>/dev/null | python3 -m json.tool 2>/dev/null | head -30 || true

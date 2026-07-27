#!/usr/bin/env bash
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=01:30:00
#SBATCH --job-name=gc-marlin
#SBATCH --output=agent_space/experiments/2026-07-27-green-context-marlin/slurm-%j.out
#SBATCH --error=agent_space/experiments/2026-07-27-green-context-marlin/slurm-%j.err

# Track A §4.5 on a Booster node: real Marlin hot(HBM)/cold(Grace-C2C) under
# disjoint green contexts. The make-or-break measurement for Track A.
# Does NOT touch the running Claude host.

set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." 2>/dev/null && pwd)"
[[ -f "${repo_dir}/agent_space/jupiter-env.sh" ]] || repo_dir=/e/project1/profound/alint77/vllm
exp="${repo_dir}/agent_space/experiments/2026-07-27-green-context-marlin"
cd "${repo_dir}"
source agent_space/jupiter-env.sh

export CUDA_DEVICE_MAX_CONNECTIONS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3
# Per-job cache dirs (parallel-job safety)
export VLLM_CACHE_ROOT="/e/scratch/profound/naeimitabiei1/vllm-cache-gc-${SLURM_JOB_ID}"

mkdir -p "${VLLM_CACHE_ROOT}" "${exp}/logs" "${exp}/nsys"
NSYS=/e/software/default/stages/2026/software/Nsight-Systems/2025.5.1-GCCcore-14.3.0/bin/nsys

echo "[$(date)] job ${SLURM_JOB_ID} on $(hostname -s)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | head -4
echo "numactl -H (top -> Grace nodes):"
numactl -H | grep -E "node [0-9]+ size:" | head -8

# Sanity: the green-context mechanism probe first (cheap, ~1s)
echo "=== mechanism probe (sanity) ==="
"${exp}/green_context_probe" 1000000 || { echo "mechanism probe failed — abort"; exit 1; }

# Marlin green-context §4.5 sweep
echo "=== Marlin §4.5 green-context sweep ==="
python "${exp}/marlin_green_probe.py" \
  --sweep --m 16 --hot-experts 19 --cold-experts 3 --iters 50 --numa-node 2 \
  --out "${exp}/results-marlin-green.json" \
  2>&1 | tee "${exp}/logs/marlin-sweep-$(date +%Y%m%d-%H%M%S).log"

echo "=== one-split nsys capture (cold=16) for greenCtxId attribution ==="
python "${exp}/marlin_green_probe.py" \
  --cold-sm 16 --m 16 --iters 20 --out "${exp}/results-marlin-green-16.json" 2>&1 | tail -8
# (nsys wrap is optional; the probe's own CUDA-event timing is the primary signal)

echo "[$(date)] done; results:"
cat "${exp}/results-marlin-green.json" 2>/dev/null | python3 -m json.tool 2>/dev/null | head -40 || true

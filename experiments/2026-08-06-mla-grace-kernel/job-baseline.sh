#!/usr/bin/env bash
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --job-name=mla-grace-p0
#SBATCH --output=agent_space/experiments/2026-08-06-mla-grace-kernel/slurm-%x-%j.out
#SBATCH --error=agent_space/experiments/2026-08-06-mla-grace-kernel/slurm-%x-%j.err
set -euo pipefail
cd /e/project1/profound/alint77/vllm
source agent_space/jupiter-env.sh
D=agent_space/experiments/2026-08-06-mla-grace-kernel
echo "node: $(hostname)"
NODE=$(.venv/bin/python agent_space/experiments/2026-07-29-marlin-smem-monopoly/detect_numa.py 0)
numactl --cpunodebind="${NODE}" --membind="${NODE}" \
  .venv/bin/python agent_space/benchmarks/mla_cache_full_footprint.py \
  --numa-node "${NODE}" --query-tokens 16 --index-mode shared \
  --warmups 5 --iterations 40 \
  | tee "${D}/${LABEL:-baseline}-${SLURM_JOB_ID}.txt" \
  | grep -vE "^[[:space:]]*[\"{}]" || true
echo "=== done ==="

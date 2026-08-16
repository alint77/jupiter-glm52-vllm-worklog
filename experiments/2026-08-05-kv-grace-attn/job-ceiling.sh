#!/usr/bin/env bash
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --job-name=grace-ceiling4
#SBATCH --output=agent_space/experiments/2026-08-05-kv-grace-attn/slurm-%x-%j.out
#SBATCH --error=agent_space/experiments/2026-08-05-kv-grace-attn/slurm-%x-%j.err
set -euo pipefail
cd /e/project1/profound/alint77/vllm
source agent_space/jupiter-env.sh
D=agent_space/experiments/2026-08-05-kv-grace-attn
echo "node: $(hostname)"
NODE=$(.venv/bin/python agent_space/experiments/2026-07-29-marlin-smem-monopoly/detect_numa.py 0)
echo "paired Grace NUMA node: ${NODE}"
numactl --cpunodebind="${NODE}" --membind="${NODE}" \
  .venv/bin/python agent_space/benchmarks/grace_read_ceiling.py \
  --numa-node "${NODE}" --gib 4 \
  | tee "${D}/ceiling-${SLURM_JOB_ID}.txt"
echo "=== done ==="

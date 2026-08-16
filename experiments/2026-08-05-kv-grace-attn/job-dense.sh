#!/usr/bin/env bash
# Dense (non-DSA) attention: KV in HBM vs Grace.
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:45:00
#SBATCH --job-name=kv-grace-dense2
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
  .venv/bin/python agent_space/benchmarks/dense_attn_kv_tier.py \
  --numa-node "${NODE}" \
  --contexts 32768 131072 \
  --query-tokens 4 \
  --batch 1 8 32 \
  --warmups 5 --iterations 30 \
  | tee "${D}/dense-${SLURM_JOB_ID}.txt"
echo "=== done ==="

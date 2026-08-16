#!/usr/bin/env bash
# Isolated attention-kernel test: KV in HBM vs Grace, swept over query tokens.
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:45:00
#SBATCH --job-name=kv-grace-attn
#SBATCH --output=agent_space/experiments/2026-08-05-kv-grace-attn/slurm-%x-%j.out
#SBATCH --error=agent_space/experiments/2026-08-05-kv-grace-attn/slurm-%x-%j.err
set -euo pipefail
cd /e/project1/profound/alint77/vllm
source agent_space/jupiter-env.sh
D=agent_space/experiments/2026-08-05-kv-grace-attn
echo "node: $(hostname)"

# MTP3 means 4 query tokens at c1 and 16 at c4. 1 is the Phase 5 gate's shape,
# kept as the control so the old number is reproduced on today's stack.
# GraceAllocation.allocate_pinned binds nothing: its docstring requires the
# caller to already be CPU- and memory-bound to the paired node. Production gets
# that from vLLM's --numa-bind; a bare sbatch does not, and the allocation lands
# on the wrong node with 0% locality. Bind explicitly.
NODE=$(.venv/bin/python agent_space/experiments/2026-07-29-marlin-smem-monopoly/detect_numa.py 0)
echo "paired Grace NUMA node for GPU 0: ${NODE}"
BIND=(numactl --cpunodebind="${NODE}" --membind="${NODE}")

for mode in shared independent; do
  echo ""
  echo "######## index-mode=${mode}"
  "${BIND[@]}" .venv/bin/python agent_space/benchmarks/mla_cache_full_footprint.py \
    --query-tokens 1 4 16 \
    --numa-node "${NODE}" \
    --index-mode "${mode}" \
    --warmups 5 --iterations 40 \
    | tee "${D}/footprint-${mode}-${SLURM_JOB_ID}.txt" \
    | grep -vE "^[[:space:]]*[\"{}]" || true
done
echo ""
echo "=== done ==="

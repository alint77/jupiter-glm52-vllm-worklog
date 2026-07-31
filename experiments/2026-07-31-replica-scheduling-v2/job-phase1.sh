#!/usr/bin/env bash
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --time-min=00:25:00
#SBATCH --job-name=replica-v2-phase1
#SBATCH --output=agent_space/experiments/2026-07-31-replica-scheduling-v2/slurm-%x-%j.out
#SBATCH --error=agent_space/experiments/2026-07-31-replica-scheduling-v2/slurm-%x-%j.err

set -euo pipefail

repo=/e/project1/profound/alint77/vllm
result_dir="${repo}/agent_space/experiments/2026-07-31-replica-scheduling-v2"
trace_dir=/e/scratch/profound/naeimitabiei1/claude-routing-profile-1047954-108
placements="${repo}/agent_space/experiments/2026-07-31-replicated-expert-scheduling/runtime-placements"

cd "${repo}"
source agent_space/jupiter-env.sh

echo "node: $(hostname)  job: ${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,clocks.max.sm --format=csv,noheader | head -1

numa_node="$(
  .venv/bin/python \
    agent_space/experiments/2026-07-29-marlin-smem-monopoly/detect_numa.py 0
)"
echo "GPU0 paired NUMA node: ${numa_node}"

cd "${result_dir}"

# Correctness first: the kernel against the host reference and against vLLM's
# own moe_align_block_size, for every rank of every layer.
for copies in 985 1697; do
  echo "=== correctness, ${copies} copies/rank ==="
  numactl --cpunodebind="${numa_node}" --membind="${numa_node}" \
    "${repo}/.venv/bin/python" -u fused_assign_align.py \
    --trace-dir "${trace_dir}" \
    --placement "${placements}/replicas-${copies}.json" \
    --steps 40 --concurrency 1 4 \
    --output "${result_dir}/phase1-correctness-${copies}-${SLURM_JOB_ID}.json"
done

# Then the gate: one fused kernel per layer against today's two alignments.
for copies in 985 1697; do
  echo "=== timing, ${copies} copies/rank ==="
  numactl --cpunodebind="${numa_node}" --membind="${numa_node}" \
    "${repo}/.venv/bin/python" -u bench_fused.py \
    --trace-dir "${trace_dir}" \
    --placement "${placements}/replicas-${copies}.json" \
    --concurrency 1 4 --replays 4000 --num-warps 2 4 8 16 \
    --output "${result_dir}/phase1-timing-${copies}-${SLURM_JOB_ID}.json"
done

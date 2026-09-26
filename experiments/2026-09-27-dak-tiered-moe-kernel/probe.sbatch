#!/usr/bin/env bash
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --job-name=dak-tma-probe
#SBATCH --output=agent_space/experiments/2026-09-27-dak-tiered-moe-kernel/probe-%j.out
#SBATCH --error=agent_space/experiments/2026-09-27-dak-tiered-moe-kernel/probe-%j.err

# TMA streaming probe sweep on a Booster GH200 (GPU 0, NUMA-bound to its Grace).
set -uo pipefail
cd /e/project1/profound/alint77/vllm
source agent_space/jupiter-env.sh
E=agent_space/experiments/2026-09-27-dak-tiered-moe-kernel
node="$(.venv/bin/python agent_space/experiments/2026-07-29-marlin-smem-monopoly/detect_numa.py 0)"
echo "GPU0 Grace node ${node} on $(hostname)"
run() { CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind="${node}" --membind="${node}" "${E}/tma_probe" "$@"; }

echo "### cold alone: CTAs x in-flight"
for c in 1 2 4 8 16 32; do for cfg in "16 4" "32 4" "32 6" "64 3"; do run "$c" 0 $cfg; done; done
echo "### hot alone"
for h in 33 66 100 132 264; do for cfg in "32 4" "64 3"; do run 0 "$h" $cfg; done; done
echo "### co-run: cold c + hot (132 - c), and cold c + hot 264 (2 CTAs/SM)"
for c in 2 4 8 16 32; do run "$c" $((132 - c)) 32 4; run "$c" 264 32 3; done
echo "### done"

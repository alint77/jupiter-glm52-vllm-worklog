#!/usr/bin/env bash
# skinny GEMM sweep and ncu comparison on a held node (GPU 0).
set -uo pipefail
cd /e/project1/profound/alint77/vllm
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
export CUDA_VISIBLE_DEVICES=0 VLLM_CACHE_ROOT=/e/fscratch/profound/${USER}/vllm-cache-dev
tag="$1"
.venv/bin/python "$E/bench_skinny.py" 2>&1 | grep -vE "Warning|warn" > "$E/bench-skinny-$tag.txt"
which ncu && ncu --section SpeedOfLight --section MemoryWorkloadAnalysis --section WarpStateStats \
  --section LaunchStats --section Occupancy -k 'regex:skinny|nvjet|sm90|gemm|Kernel' \
  --launch-skip 4 --launch-count 2 .venv/bin/python "$E/ncu_skinny.py" 0 > "$E/ncu-skinny-$tag.txt" 2>&1
echo done

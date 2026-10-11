#!/usr/bin/env bash
# ncu (base clock) SASS stall samples of one kernel at one size:
#   ncu_src.sh <tag> <size> <kernel regex> [launch-skip]   -> logs/src-<tag>.csv
cd /e/project1/profound/alint77/vllm
export PATH=$PWD/.venv/bin:$PATH
X=agent_space/experiments/2026-10-11-sliced-prefill-moe
NCU=/e/software/default/stages/2026/software/Nsight-Compute/2025.3.1-GCCcore-14.3.0/ncu
export PYTHONPATH=/e/fscratch/profound/${USER}/worktrees/${TREE:-pfk}:$X
export VLLM_TIERED_PREFILL_DEFINES="${DEFINES:-}"
export VLLM_CACHE_ROOT=/e/fscratch/profound/${USER}/caches/kernel-${TREE:-pfk}
CUDA_VISIBLE_DEVICES=${GPU:-0} numactl --cpunodebind=${GPU:-0} --membind=${GPU:-0} "$NCU" --target-processes all --profile-from-start off \
  --clock-control base -k regex:"$3" --launch-skip ${4:-0} --launch-count 1 --section SourceCounters --section WarpStateStats \
  -f -o $X/logs/src-$1 .venv/bin/python $X/one_call.py $2 > $X/logs/src-$1.log 2>&1
"$NCU" --import $X/logs/src-$1.ncu-rep --page source --csv --print-source sass > $X/logs/src-$1.csv 2>/dev/null
.venv/bin/python $X/src_top.py $X/logs/src-$1.csv

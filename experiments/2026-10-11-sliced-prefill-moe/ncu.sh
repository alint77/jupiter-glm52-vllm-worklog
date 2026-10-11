#!/usr/bin/env bash
# ncu (base clock) of every kernel of one MoE call per size, worktree TREE:
#   ncu.sh <tag> <sizes> [TREE]   -> logs/ncu-<tag>.ncu-rep + summary
cd /e/project1/profound/alint77/vllm
export PATH=$PWD/.venv/bin:$PATH
X=agent_space/experiments/2026-10-11-sliced-prefill-moe
NCU=/e/software/default/stages/2026/software/Nsight-Compute/2025.3.1-GCCcore-14.3.0/ncu
TREE=${3:-pfk}
export PYTHONPATH=/e/fscratch/profound/${USER}/worktrees/${TREE}:$X
export VLLM_TIERED_PREFILL_DEFINES="${DEFINES:-}"
export VLLM_CACHE_ROOT=/e/fscratch/profound/${USER}/caches/kernel-${TREE}
M=gpu__time_duration.sum,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,dram__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes_read.sum,dram__bytes_write.sum,smsp__issue_active.avg.pct_of_peak_sustained_active,sm__warps_active.avg.pct_of_peak_sustained_active,launch__grid_size,launch__registers_per_thread,sm__cycles_active.avg,gpc__cycles_elapsed.max,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed
CUDA_VISIBLE_DEVICES=${GPU:-0} numactl --cpunodebind=${GPU:-0} --membind=${GPU:-0} "$NCU" --target-processes all --profile-from-start off \
  --clock-control base --section WarpStateStats --metrics $M -f -o $X/logs/ncu-$1 \
  .venv/bin/python $X/one_call.py $2 > $X/logs/ncu-$1.log 2>&1
"$NCU" --import $X/logs/ncu-$1.ncu-rep --page raw --csv --metrics $M,smsp__average_warp_latency_issue_stalled_long_scoreboard.ratio > $X/logs/ncu-$1.raw.csv 2>/dev/null
.venv/bin/python $X/ncu_table.py $X/logs/ncu-$1.raw.csv

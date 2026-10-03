#!/usr/bin/env bash
# ncu (base clock) per-SASS warp stall samples of the 128-token w13 tile,
# mode 0 (full) vs mode 2 (compute only), for VLLM_TIERED_PREFILL_DEFINES=$1.
# Prints stall samples grouped by instruction kind for each mode.
cd /e/project1/profound/alint77/vllm
D=agent_space/experiments/2026-10-02-ingraph-prefetch
NCU=/e/software/default/stages/2026/software/Nsight-Compute/2025.3.1-GCCcore-14.3.0/ncu
OUT=/e/fscratch/profound/${USER}/ingraph-prefetch/src128; mkdir -p "$OUT"
export VLLM_CACHE_ROOT=/e/fscratch/profound/${USER}/caches/prefill-kernel-iter
export VLLM_TIERED_PREFILL_DEFINES="$1"
tag=$(echo "${1:-default}" | tr ' =' '__')
cp /e/fscratch/profound/${USER}/ingraph-prefetch/t128/one.py "$OUT/one.py"
numactl --cpunodebind=0 --membind=0 "$NCU" --target-processes all -k regex:gemm_kernel --clock-control base \
  --section SourceCounters --section WarpStateStats \
  --metrics sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,gpu__time_duration.sum \
  -f -o "$OUT/$tag" .venv/bin/python "$OUT/one.py" > "$OUT/$tag.log" 2>&1
for i in 0 1; do
  "$NCU" --import "$OUT/$tag.ncu-rep" --launch-skip $i --launch-count 1 --page source --csv --print-source sass > "$OUT/$tag.m$i.csv" 2>/dev/null
done
"$NCU" --import "$OUT/$tag.ncu-rep" --page raw --csv --metrics sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,gpu__time_duration.sum > "$OUT/$tag.raw.csv"
.venv/bin/python $D/src_stalls.py "$OUT/$tag.raw.csv" "$OUT/$tag.m0.csv" "$OUT/$tag.m1.csv"

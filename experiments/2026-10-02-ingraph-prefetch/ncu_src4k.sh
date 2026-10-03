#!/usr/bin/env bash
# ncu (base clock) per-SASS stall samples of the 128- and 96-wide GEMMs at a
# real 4096-token chunk (w13 and w2): where do warps wait?
cd /e/project1/profound/alint77/vllm
D=agent_space/experiments/2026-10-02-ingraph-prefetch
NCU=/e/software/default/stages/2026/software/Nsight-Compute/2025.3.1-GCCcore-14.3.0/ncu
OUT=/e/fscratch/profound/${USER}/ingraph-prefetch/src4k; mkdir -p "$OUT"
export VLLM_CACHE_ROOT=/e/fscratch/profound/${USER}/caches/prefill-kernel-iter
cp /e/fscratch/profound/${USER}/ingraph-prefetch/m4k/one.py "$OUT/one.py"
numactl --cpunodebind=0 --membind=0 "$NCU" --target-processes all --kernel-name-base demangled -k 'regex:gemm_kernel<\(int\)(128|96)' --clock-control base \
  --section SourceCounters --section WarpStateStats \
  --metrics sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,gpu__time_duration.sum \
  -f -o "$OUT/r" .venv/bin/python "$OUT/one.py" > "$OUT/r.log" 2>&1
for i in 0 1 2 3; do
  "$NCU" --import "$OUT/r.ncu-rep" --launch-skip $i --launch-count 1 --page source --csv --print-source sass > "$OUT/r.m$i.csv" 2>/dev/null
done
.venv/bin/python $D/src_where.py "$OUT"/r.m{0,1,2,3}.csv

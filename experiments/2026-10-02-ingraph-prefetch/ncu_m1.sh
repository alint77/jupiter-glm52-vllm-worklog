#!/usr/bin/env bash
# ncu on the prefill GEMM kernel: one w13 call at N=16 and N=64.
cd /e/project1/profound/alint77/vllm
NCU=/e/software/default/stages/2026/software/Nsight-Compute/2025.3.1-GCCcore-14.3.0/ncu
OUT=/e/fscratch/profound/${USER}/ingraph-prefetch/m1; mkdir -p "$OUT"
tag=${1:-m1}
cat > "$OUT/one.py" <<'PY'
import sys, torch
from vllm.model_executor.layers.fused_moe import tiered_prefill
dev = torch.device("cuda", 0)
k, f = 6144, 4096
q = torch.randint(-2**31, 2**31 - 1, (64, k // 16, f * 2), dtype=torch.int32, device=dev)
s = (torch.rand((64, k // 32, f), device=dev) / 64).to(torch.bfloat16)
for n in (16, 64):
    x = torch.randn((n, k), dtype=torch.bfloat16, device=dev)
    tiered_prefill.dense(x, q, s)
torch.accelerator.synchronize()
PY
numactl --cpunodebind=0 --membind=0 "$NCU" --target-processes all -k regex:gemm_kernel --clock-control none \
  --section SpeedOfLight --section Occupancy --section WarpStateStats --section SchedulerStats \
  --section ComputeWorkloadAnalysis --section MemoryWorkloadAnalysis --metrics sm__pipe_tensor_op_gmma_cycles_active.avg.pct_of_peak_sustained_active,sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active,sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active \
  -f -o "$OUT/$tag" .venv/bin/python "$OUT/one.py" > "$OUT/$tag.log" 2>&1
"$NCU" --import "$OUT/$tag.ncu-rep" --page raw --csv > "$OUT/$tag.raw.csv"
echo done

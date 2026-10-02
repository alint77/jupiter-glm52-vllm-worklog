#!/usr/bin/env bash
# One kernel iteration on a held node: benchmark (full + compute-only), then
# ncu at the base clock (w13 N=16 and N=64, full and compute-only) with stalls.
#   ./onnode.sh <D>/iter.sh <tag>
cd /e/project1/profound/alint77/vllm
D=agent_space/experiments/2026-10-02-ingraph-prefetch
OUT=/e/fscratch/profound/${USER}/ingraph-prefetch/m1; mkdir -p "$OUT"
tag=$1
:
:
NCU=/e/software/default/stages/2026/software/Nsight-Compute/2025.3.1-GCCcore-14.3.0/ncu
cat > "$OUT/iter_one.py" <<'PY'
import torch
from vllm.model_executor.layers.fused_moe import tiered_prefill
dev = torch.device("cuda", 0)
k, f = 6144, 4096
q = torch.randint(-2**31, 2**31 - 1, (64, k // 16, f * 2), dtype=torch.int32, device=dev)
s = (torch.rand((64, k // 32, f), device=dev) / 64).to(torch.bfloat16)
k_exp = tiered_prefill.scale_exponent(s)
for n in (64, 128):
    x = torch.randn((n, k), dtype=torch.bfloat16, device=dev)
    for mode in (0, 2):
        tiered_prefill.dense(x, q, s, mode, k_exp)
torch.accelerator.synchronize()
PY
numactl --cpunodebind=0 --membind=0 "$NCU" --target-processes all -k regex:gemm_kernel --clock-control base \
  --section SpeedOfLight --section Occupancy --section WarpStateStats --section SchedulerStats \
  --section ComputeWorkloadAnalysis --section MemoryWorkloadAnalysis \
  --metrics sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active,sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active \
  -f -o "$OUT/$tag" .venv/bin/python "$OUT/iter_one.py" > "$OUT/$tag.log" 2>&1
"$NCU" --import "$OUT/$tag.ncu-rep" --page raw --csv > "$OUT/$tag.raw.csv"
echo "## ncu (base clock): N=64 full, N=64 compute, N=128 full, N=128 compute"
.venv/bin/python $D/ncu_summary.py "$OUT/$tag.raw.csv"

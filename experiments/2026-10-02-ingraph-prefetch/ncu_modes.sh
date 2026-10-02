#!/usr/bin/env bash
# Clock and time per probe mode (0 full, 1 loads only, 2 compute only), w13 N=16.
cd /e/project1/profound/alint77/vllm
NCU=/e/software/default/stages/2026/software/Nsight-Compute/2025.3.1-GCCcore-14.3.0/ncu
OUT=/e/fscratch/profound/${USER}/ingraph-prefetch/m1; mkdir -p "$OUT"
cat > "$OUT/modes.py" <<'PY'
import torch
from vllm.model_executor.layers.fused_moe import tiered_prefill
dev = torch.device("cuda", 0)
k, f = 6144, 4096
q = torch.randint(-2**31, 2**31 - 1, (64, k // 16, f * 2), dtype=torch.int32, device=dev)
s = (torch.rand((64, k // 32, f), device=dev) / 64).to(torch.bfloat16)
x = torch.randn((16, k), dtype=torch.bfloat16, device=dev)
for _ in range(30):   # warm, as in back-to-back layers
    for mode in (0, 1, 2):
        tiered_prefill.dense(x, q, s, mode)
torch.accelerator.synchronize()
for mode in (0, 1, 2):
    tiered_prefill.dense(x, q, s, mode)
torch.accelerator.synchronize()
PY
numactl --cpunodebind=0 --membind=0 "$NCU" --target-processes all -k regex:gemm_kernel --launch-skip 90 --clock-control none \
  --metrics gpu__time_duration.sum,gpc__cycles_elapsed.avg.per_second,dram__bytes.sum.per_second,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active \
  --csv .venv/bin/python "$OUT/modes.py" 2>/dev/null | grep -E "gpu__time|gpc__cycles|dram__bytes|tensor_cycles|pipe_alu" | awk -F'","' '{print $5, $(NF-2), $NF}'

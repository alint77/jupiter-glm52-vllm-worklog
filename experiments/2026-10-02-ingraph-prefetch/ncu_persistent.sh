#!/usr/bin/env bash
cd /e/project1/profound/alint77/vllm
D=agent_space/experiments/2026-10-02-ingraph-prefetch
NCU=/e/software/default/stages/2026/software/Nsight-Compute/2025.3.1-GCCcore-14.3.0/ncu
OUT=/e/fscratch/profound/${USER}/ingraph-prefetch/m2; mkdir -p "$OUT"
cat > "$OUT/one.py" <<'PY'
import sys
sys.path.insert(0, "agent_space/experiments/2026-10-02-ingraph-prefetch")
import numpy as np, torch, real_routing, bench_moe_real as b
from vllm.model_executor.layers.fused_moe import tiered_prefill
smp = real_routing.samples(512, 2, seed=512)[1]
ids = torch.from_numpy(smp.topk_ids).to(b.dev); wts = torch.rand(ids.shape, device=b.dev).softmax(-1)
x = torch.randn((512, b.H), dtype=torch.bfloat16, device=b.dev)
local = np.flatnonzero(smp.local_map >= 0)
hm = torch.full((256,), -1, dtype=torch.int32); cm = torch.full((256,), -1, dtype=torch.int32)
hm[torch.from_numpy(local[:40])] = torch.arange(40, dtype=torch.int32); cm[torch.from_numpy(local[40:])] = torch.arange(24, dtype=torch.int32)
for sch in (0, 4):
    tiered_prefill.tiered_prefill_moe(x, ids, wts, hm.to(b.dev), cm.to(b.dev), b.hot, b.cold, b.k_exp, sch)
torch.accelerator.synchronize()
PY
numactl --cpunodebind=0 --membind=0 "$NCU" --target-processes all -k regex:gemm_kernel --clock-control base \
  --section SpeedOfLight --section Occupancy --section WarpStateStats --section SchedulerStats --section ComputeWorkloadAnalysis \
  --metrics sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active,sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active,dram__bytes.sum.per_second,sm__cycles_active.avg,sm__cycles_active.max,sm__cycles_active.min \
  -f -o "$OUT/p512" .venv/bin/python "$OUT/one.py" > "$OUT/p512.log" 2>&1
"$NCU" --import "$OUT/p512.ncu-rep" --page raw --csv > "$OUT/p512.raw.csv"
.venv/bin/python - "$OUT/p512.raw.csv" <<'PY'
import csv, sys
r = list(csv.reader(open(sys.argv[1]))); h = r[0]
def g(d, n):
    try: return float(d[h.index(n)].replace(",", ""))
    except Exception: return float("nan")
for d in r[2:]:
    name = d[h.index("Kernel Name")][:40]
    print(f"{name:40s} t={g(d,'gpu__time_duration.sum')/1e3:7.1f}us dram={g(d,'dram__bytes.sum.per_second')/1e12:.2f}TB/s "
          f"tensor={g(d,'sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active'):.0f}% alu={g(d,'sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active'):.0f}% "
          f"issue={g(d,'smsp__issue_active.avg.pct_of_peak_sustained_active'):.0f}% occ={g(d,'sm__warps_active.avg.pct_of_peak_sustained_active'):.0f}% "
          f"active cyc min/avg/max {g(d,'sm__cycles_active.min'):.0f}/{g(d,'sm__cycles_active.avg'):.0f}/{g(d,'sm__cycles_active.max'):.0f}")
    st = sorted(((n.split("issue_stalled_")[1].split("_per_issue")[0], g(d, n)) for n in h if n.startswith("smsp__average_warps_issue_stalled_") and n.endswith("_per_issue_active.ratio")), key=lambda kv: -kv[1])
    print("    stalls:", ", ".join(f"{a} {b:.2f}" for a, b in st[:6]))
PY

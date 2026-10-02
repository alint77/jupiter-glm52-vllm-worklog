"""Per-launch timeline of the grouped w13 on one real 512/2048-token chunk:
each tile width's units, start, duration, gaps (profiler, eager)."""
import sys
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import real_routing  # noqa: E402
from vllm.model_executor.layers.fused_moe import tiered_prefill

H, I, E = 6144, 2048, 64
dev = torch.device("cuda", 0)
ext = tiered_prefill._extension()
q = torch.randint(-2**31, 2**31 - 1, (E, H // 16, 4 * I), dtype=torch.int32, device=dev)
s = ((0.5 + torch.rand((E, H // 32, 2 * I), device=dev) / 2) / 64).to(torch.bfloat16)
k_exp = tiered_prefill.scale_exponent(s)
for chunk in (512, 2048):
    smp = real_routing.samples(chunk, 3, seed=chunk)[1]
    ids = torch.from_numpy(smp.topk_ids).to(dev)
    lmap = torch.from_numpy(smp.local_map).to(dev)
    x = torch.randn((chunk, H), dtype=torch.bfloat16, device=dev)
    for _ in range(3):
        tiered_prefill.grouped_gemm(x, ids, lmap, q, s, k_exp)
    torch.accelerator.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        tiered_prefill.grouped_gemm(x, ids, lmap, q, s, k_exp)
        torch.accelerator.synchronize()
    ks = sorted((e for e in prof.events() if e.device_type.name == "CUDA"
                 and "gemm_kernel" in e.name), key=lambda e: e.time_range.start)
    c = smp.counts()
    print(f"chunk {chunk}: counts sorted {sorted(c.tolist(), reverse=True)[:8]} ... used {(c > 0).sum()}")
    t0 = ks[0].time_range.start
    for e in ks:
        nt = e.name.split("gemm_kernel<")[1].split(",")[0]
        print(f"   NT={nt:>3}: start {e.time_range.start - t0:7.1f} us, dur {e.device_time:6.1f} us")
    print(f"   total {ks[-1].time_range.end - t0:.1f} us")

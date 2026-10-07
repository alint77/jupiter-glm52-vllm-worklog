"""How much faster is an M=8 bf16 GEMM (F.linear, cuBLAS) when a fraction
of its weight is already in L2? Per iteration: flush L2 (write 256 MB), read
the first f of W (a side read standing in for a prefetch issued during an
earlier latency-bound kernel), then the GEMM. Kernel times from the torch
profiler; the GEMM is the only nvjet/cutlass kernel.

    l2_prefetch_probe.py
"""
import torch
from torch.profiler import ProfilerActivity, profile

SHAPES = {"o_proj": (6144, 4096), "fused_qkv_a": (2624, 6144), "q_b": (4096, 2048),
          "shared_gate_up": (1024, 6144)}
dev = torch.device("cuda")
flush = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
x = torch.randn(8, 1, dtype=torch.bfloat16, device=dev)
print(f"L2 {torch.cuda.get_device_properties(0).L2_cache_size / 2**20:.0f} MiB")
for name, (n, k) in SHAPES.items():
    w = torch.randn(n, k, dtype=torch.bfloat16, device=dev)
    x = torch.randn(8, k, dtype=torch.bfloat16, device=dev)
    flat = w.view(-1)
    sink = torch.empty(1, dtype=torch.float32, device=dev)
    row = []
    for f in (0.0, 0.25, 0.5, 0.75, 1.0):
        m = int(flat.numel() * f)

        def it():
            flush.fill_(1)
            if m:
                torch.sum(flat[:m], dim=0, dtype=torch.float32, out=sink[0])
            torch.nn.functional.linear(x, w)

        for _ in range(5):
            it()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as p:
            for _ in range(50):
                it()
            torch.cuda.synchronize()
        ks = [e for e in p.events() if e.device_type.name == "CUDA"
              and ("nvjet" in e.name or "gemm" in e.name.lower() or "splitK" in e.name)]
        per = sum(e.device_time for e in ks) / 50
        row.append(per)
    mb = n * k * 2 / 2**20
    floor = n * k * 2 / 3.35e12 * 1e6
    print(f"{name:15s} {mb:5.1f} MiB  floor@3.35TB/s {floor:5.1f} us  GEMM(+reduce) us at L2-prefetched "
          + " ".join(f"{f:.0%}:{t:.1f}" for f, t in zip((0, .25, .5, .75, 1), row)))

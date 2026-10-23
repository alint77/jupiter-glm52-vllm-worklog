"""PDL race test for decode_gemm: a primary that triggers dependents at entry
and writes x late, then decode_gemm(pdl=True) on x. Its output must equal the
serialized (pdl=False) result on the same data, bit for bit
(int16 views). M covers 1-4 token tiles, full and partial.

    race_test.py <dir with decode_gemm.cu> [iters]
"""
import sys
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

src_dir = Path(sys.argv[1]).resolve()
iters = int(sys.argv[2]) if len(sys.argv) > 2 else 200
flags = ["-O3", "-gencode=arch=compute_90a,code=sm_90a"]
dg = load(name=f"dg_{src_dir.name}", sources=[str(src_dir / "decode_gemm.cu")],
          extra_cuda_cflags=flags, build_directory=str(src_dir))
pb = Path(__file__).resolve().parent / "pbuild"
pb.mkdir(exist_ok=True)
pr = load(name="dg_producer", sources=[str(pb.parent / "producer.cu")],
          extra_cuda_cflags=flags, build_directory=str(pb))

torch.manual_seed(0)
SHAPES = {"q_b": (4096, 2048), "qkv_a": (2112, 6144)}  # (N, K)
CFGS = [(2, 4), (2, 8), (4, 2), (4, 4), (8, 2), (8, 4)]
SENTINEL = 1e4
bad_total = 0
for name, (n, k) in SHAPES.items():
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02
    for m in (4, 8, 9, 16, 17, 24, 25, 32):
        # two sources alternate, so a stale x from the previous run is caught too
        srcs = [torch.randn(m, k, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
        x = torch.empty_like(srcs[0])
        for cfg in CFGS:
            if k % (32 * cfg[0] * cfg[1]):
                continue
            refs = [dg.gemm(s, w, *cfg, False).view(torch.int16) for s in srcs]
            # queued pairs, checked after one sync
            ys = []
            for i in range(iters):
                x.fill_(SENTINEL)
                pr.produce(srcs[i % 2], x, 20000)
                ys.append(dg.gemm(x, w, *cfg, True))
            torch.cuda.synchronize()
            bad = sum(int(not torch.equal(y.view(torch.int16), refs[i % 2]))
                      for i, y in enumerate(ys))
            # same under CUDA graph replay
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for i in range(2):
                    x.fill_(SENTINEL)
                    pr.produce(srcs[i], x, 20000)
                    ys[i] = dg.gemm(x, w, *cfg, True)
            gbad = 0
            for _ in range(iters // 2):
                g.replay()
                gbad += sum(int(not torch.equal(ys[i].view(torch.int16), refs[i]))
                            for i in range(2))
            torch.cuda.synchronize()
            bad_total += bad + gbad
            print(f"{src_dir.name} {name:5s} M={m:2d} cfg={cfg}  eager {bad:4d}/{iters}"
                  f"  graph {gbad:4d}/{iters}", flush=True)
print(f"{src_dir.name}: {bad_total} mismatching runs")

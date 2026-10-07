"""One o_proj-shaped launch each of F.linear, skinny_v2 (w16 u4) and the LDG
stream probe, for base-clock ncu:
    ncu --set full -k regex:"nvjet|skinny|ldg_stream" --launch-skip 3 --launch-count 3 python ncu_v2.py
"""
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

HERE = Path(__file__).resolve().parent
v2 = load(name="skinny_v2", sources=[str(HERE / "skinny_v2.cu")],
          extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
          build_directory="/e/fscratch/profound/naeimitabiei1/caches/skinny_v2")
sp = load(name="stream_probe", sources=[str(HERE / "stream_probe.cu")],
          extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
          build_directory="/e/fscratch/profound/naeimitabiei1/caches/stream_probe")
w = (torch.randn(6144, 4096, device="cuda") * 0.02).to(torch.bfloat16)
x = torch.randn(8, 4096, device="cuda").to(torch.bfloat16)
sink = torch.zeros(1, dtype=torch.int32, device="cuda")
for _ in range(2):
    F.linear(x, w), v2.gemm(x, w, 16, 4), sp.ldg(w, sink, 132, 512, 8)
torch.cuda.synchronize()
F.linear(x, w), v2.gemm(x, w, 16, 4), sp.ldg(w, sink, 132, 512, 8)
torch.cuda.synchronize()

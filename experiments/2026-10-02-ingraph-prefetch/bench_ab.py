"""Same-process A/B: the committed per-group kernel (77ce7f5, reference) vs
the working tree's kernel, w13/w2 at several N. Graph + profiler timing."""
import sys
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile
from torch.utils.cpp_extension import load

import vllm.envs as envs
from vllm.model_executor.layers.fused_moe import tiered_prefill

HERE = Path(__file__).resolve().parent
ref = load(name="tp_ref_77ce7f5", sources=[str(HERE / "tp_ref_77ce7f5.cu")],
           extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-std=c++17"],
           extra_ldflags=["-lcuda"],
           build_directory=str(Path(envs.VLLM_CACHE_ROOT) / "torch_extensions" / "tp_ref"))
E, GROUP = 64, 32
dev = torch.device("cuda", 0)
NS = [int(a) for a in sys.argv[1:]] or [16, 32, 64, 128]


def timed(fn):
    for _ in range(3):
        fn()
    torch.accelerator.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    g.replay()
    torch.accelerator.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(20):
            g.replay()
        torch.accelerator.synchronize()
    return sum(e.device_time for e in prof.events()
               if e.device_type.name == "CUDA" and "gemm_kernel" in e.name) / 20


Path(envs.VLLM_CACHE_ROOT, "torch_extensions", "tp_ref").mkdir(parents=True, exist_ok=True)
for k, f, name in ((6144, 4096, "w13"), (2048, 6144, "w2")):
    q = torch.randint(-2**31, 2**31 - 1, (E, k // 16, f * 2), dtype=torch.int32, device=dev)
    s = ((0.5 + torch.rand((E, k // GROUP, f), device=dev) / 2) / 64).to(torch.bfloat16)
    k_exp = tiered_prefill.scale_exponent(s)
    for n in NS:
        x = torch.randn((n, k), dtype=torch.bfloat16, device=dev)
        y = torch.empty((E, n, f), dtype=torch.bfloat16, device=dev)
        t_ref = timed(lambda: ref.dense(y, x, q, s, 0))
        t_new = timed(lambda: tiered_prefill.dense(x, q, s, 0, k_exp))
        print(f"{name} N={n:3d}: ref {t_ref:7.1f} us  new {t_new:7.1f} us  "
              f"({100 * (t_new / t_ref - 1):+.1f}%)")

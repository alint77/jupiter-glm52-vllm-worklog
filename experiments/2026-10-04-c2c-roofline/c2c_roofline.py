#!/usr/bin/env python3
"""NVLink-C2C read roofline on one GH200 vs the tiered decode MoE's cold tier.

1. Ceiling: GPU 0 reading NUMA-local pinned Grace memory (GraceAllocation,
   the cold tier's allocator), 2 GiB per measurement, three ways:
   - copy engine (cudaMemcpy H2D)
   - SM loads: 16 B per thread, grid-stride, N CTAs x 512 threads
   - TMA bulk copies (cp.async.bulk, 32 KB chunks, 4-stage mbarrier ring,
     one issuing thread per CTA): the decode kernel's mechanism
   swept over CTA counts (the decode kernel gives the cold tier 16 / 24).
2. Ours: tiered_decode_moe with only cold experts (h = 0, c = 1..4), CUDA-graph
   replay, effective GB/s = c x 21.23 MB / call time (call = route_prep, w13,
   act, w2, finalize).
Run NUMA-bound to the GPU's Grace node:
    numactl --cpunodebind=0 --membind=0 python c2c_roofline.py
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--roof-only", action="store_true", help="Measure transfer ceilings only"
)
args = parser.parse_args()

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "bench_fmt", HERE.parent / "2026-09-27-glm53-mtp7-profile/bench_fmt.py")
BF = importlib.util.module_from_spec(spec)
spec.loader.exec_module(BF)
B = BF.B

SRC = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
__global__ void sm_read(const uint4* __restrict__ p, int64_t n, uint4* out) {
  uint4 acc = make_uint4(0, 0, 0, 0);
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n;
       i += (int64_t)gridDim.x * blockDim.x) {
    uint4 v = p[i];
    acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
  }
  if (acc.x == 0x9e3779b9u) out[0] = acc;
}
__device__ __forceinline__ uint32_t s32(const void* p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}
__global__ void tma_read(const char* src, int64_t per_cta, int chunk, int stages) {
  extern __shared__ __align__(128) unsigned char ring[];
  __shared__ __align__(8) uint64_t bar[8];
  if (threadIdx.x != 0) return;
  for (int s = 0; s < stages; ++s)
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(s32(&bar[s])));
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  const char* base = src + blockIdx.x * per_cta;
  const int n = (int)(per_cta / chunk);
  auto wait = [&](int s, int parity) {
    asm volatile("{\n .reg .pred p;\n W_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra W_%=;\n}"
                 ::"r"(s32(&bar[s])), "r"(parity) : "memory");
  };
  for (int i = 0; i < n; ++i) {
    const int s = i % stages;
    if (i >= stages) wait(s, ((i / stages) - 1) & 1);
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                 ::"r"(s32(&bar[s])), "r"(chunk) : "memory");
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 ::"r"(s32(ring + (size_t)s * chunk)), "l"(base + (int64_t)i * chunk), "r"(chunk),
                   "r"(s32(&bar[s])) : "memory");
  }
  for (int i = n > stages ? n - stages : 0; i < n; ++i) wait(i % stages, (i / stages) & 1);
}
void run_sm(int64_t ptr, int64_t bytes, int ctas, torch::Tensor out) {
  sm_read<<<ctas, 512, 0, at::cuda::getCurrentCUDAStream()>>>(
      (const uint4*)ptr, bytes / 16, (uint4*)out.data_ptr());
}
void run_tma(int64_t ptr, int64_t bytes, int ctas, int chunk, int stages) {
  auto f = tma_read;
  int smem = chunk * stages;
  cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  tma_read<<<ctas, 32, smem, at::cuda::getCurrentCUDAStream()>>>(
      (const char*)ptr, bytes / ctas / chunk * chunk, chunk, stages);
}
"""
ext = load_inline("c2c_roofline", cpp_sources=(
    "void run_sm(int64_t, int64_t, int, torch::Tensor);"
    "void run_tma(int64_t, int64_t, int, int, int);"), cuda_sources=SRC,
    functions=["run_sm", "run_tma"],
    extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"])

from vllm.model_executor.offloader.grace import GraceAllocation  # noqa: E402

dev = torch.device("cuda:0")
GB = 2 << 30
host = GraceAllocation.allocate_pinned((GB,), torch.uint8, 0, 0)
host_t = host.cuda_alias  # device-visible alias of the Grace buffer
ptr = host_t.data_ptr()
dst = torch.empty(GB, dtype=torch.uint8, device=dev)
out = torch.empty(16, dtype=torch.uint8, device=dev)
ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]


def gbps(fn, nbytes, reps=5):
    fn()
    torch.cuda.synchronize()
    best = 0.0
    for _ in range(reps):
        ev[0].record()
        fn()
        ev[1].record()
        torch.cuda.synchronize()
        best = max(best, nbytes / (ev[0].elapsed_time(ev[1]) * 1e-3) / 1e9)
    return best


print(f"copy engine H2D: {gbps(lambda: dst.copy_(host_t), GB):.0f} GB/s", flush=True)
for ctas in (8, 16, 24, 32, 48, 64, 132, 264):
    sm = gbps(lambda: ext.run_sm(ptr, GB, ctas, out), GB)
    nb = GB // ctas // 32768 * 32768 * ctas
    tma = gbps(lambda: ext.run_tma(ptr, GB, ctas, 32768, 4), nb)
    print(f"{ctas:4d} CTAs: SM loads {sm:4.0f} GB/s   TMA 32KBx4 {tma:4.0f} GB/s", flush=True)
for chunk, stages in ((32768, 6), (16384, 8), (65536, 3)):
    print(f"  TMA {chunk // 1024} KB x {stages}, 24 CTAs: "
          f"{gbps(lambda: ext.run_tma(ptr, GB, 24, chunk, stages), GB // 24 // chunk * chunk * 24):.0f} GB/s",
          flush=True)
if args.roof_only:
    sys.exit(0)
# Keep the registered Grace backing alive through the MoE measurements.
del dst
torch.cuda.empty_cache()

# 2. our kernel, cold experts only
gen = torch.Generator().manual_seed(0)
hot, _ = BF.tier("int4", 16, dev, False, 0, gen, 32)
cold, keep = BF.tier("int4", 8, dev, True, 0, gen, 100)
x = (torch.randn((B.TOKENS, B.HIDDEN), generator=gen) * 0.3).to(torch.bfloat16).to(dev)
EXPERT = 21_233_672
for c in (1, 2, 3, 4):
    calls = [B.routing(0, c, 32, 100, r, gen, dev) for r in range(20)]
    for ids, w, hmap, cmap in calls:  # fresh cold experts every call
        on = cmap >= 0
        cmap[on] = torch.randint(0, 100, (int(on.sum()),), device=dev, dtype=cmap.dtype)
    us = B.wall_us(x, calls, hot, cold)
    print(f"tiered_decode_moe, 0 hot + {c} cold: {us:6.1f} us/call -> "
          f"{c * EXPERT / (us * 1e-6) / 1e9:4.0f} GB/s effective "
          f"(link-only floor at 390 GB/s: {c * EXPERT / 390e9 * 1e6:.0f} us)", flush=True)
del keep

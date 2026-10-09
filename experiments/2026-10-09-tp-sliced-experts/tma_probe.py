#!/usr/bin/env python3
"""Streaming throughput of the kernel's weight-load patterns, nothing else:
GRID persistent CTAs, a producer lane issuing one unit per ring stage, 8
consumer warps that only release the stage. Per pattern and ring depth: GB/s.

  w13box  {128 int32, 64 k16 rows} of [E][384][2048] int32 (512 B per row, 8 KB stride)
  w2box   {256 int32, 32 k16 rows} of [E][32][12288] int32 (1 KB per row, 48 KB stride)
  flat    one 32 KB cp.async.bulk of contiguous memory

    tma_probe.py [--numa-node 0]
"""
import argparse
import json

import torch
from torch.utils.cpp_extension import load_inline

SRC = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cudaTypedefs.h>
__device__ __forceinline__ uint32_t su(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void init(uint64_t* b, int c) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(su(b)), "r"(c)); }
__device__ __forceinline__ void etx(uint64_t* b, int n) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(su(b)), "r"(n) : "memory"); }
__device__ __forceinline__ void arr(uint64_t* b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(su(b)) : "memory"); }
__device__ __forceinline__ void wt(uint64_t* b, int ph) { asm volatile("{ .reg .pred p; W%=: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1; @!p bra W%=; }" :: "r"(su(b)), "r"(ph) : "memory"); }
template <int STAGES>
__global__ void __launch_bounds__(288, 1) stream(const __grid_constant__ CUtensorMap map, int mode,
    const char* flat, long units, int box_k, unsigned long long* t) {
  extern __shared__ __align__(1024) unsigned char sm[];
  uint64_t* full = (uint64_t*)sm; uint64_t* empty = full + 8;
  unsigned char* ring = sm + 1024;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  if (threadIdx.x == 0) { for (int s = 0; s < STAGES; ++s) { init(&full[s], 1); init(&empty[s], 8); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
  __syncthreads();
  unsigned long long t0; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
  long u0 = units * blockIdx.x / gridDim.x, u1 = units * (blockIdx.x + 1) / gridDim.x;
  if (warp == 8) {
    if (lane == 0)
      for (long u = u0, it = 0; u < u1; ++u, ++it) {
        int s = it % STAGES; unsigned char* d = ring + s * 33792;
        if (it >= STAGES) wt(&empty[s], ((it / STAGES) - 1) & 1);
        etx(&full[s], 32768);
        if (mode == 2) {
          asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], 32768, [%2];"
                       :: "r"(su(d)), "l"(flat + u * 32768), "r"(su(&full[s])) : "memory");
        } else {
          // mode 0: w13 box (tile t of 16, chunk c of 6, expert e); mode 1: w2 box (tile of 48, expert)
          int c0, c1, c2;
          if (mode == 0) { long e = u / 96, r = u % 96; c0 = (r / 6) * 128; c1 = (r % 6) * 64; c2 = e; }
          else { long e = u / 48, r = u % 48; c0 = r * 256; c1 = 0; c2 = e; }
          asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];"
                       :: "r"(su(d)), "l"((unsigned long long)&map), "r"(c0), "r"(c1), "r"(c2), "r"(su(&full[s])) : "memory");
        }
      }
  } else {
    for (long u = u0, it = 0; u < u1; ++u, ++it) {
      int s = it % STAGES; wt(&full[s], (it / STAGES) & 1);
      __syncwarp(); if (lane == 0) arr(&empty[s]);
    }
  }
  __syncthreads();
  unsigned long long t1; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t1));
  if (threadIdx.x == 0) { atomicMin(&t[0], t0); atomicMax(&t[1], t1); }
}
static CUtensorMap mk(torch::Tensor w, int b0, int b1) {
  CUtensorMap m; cuuint64_t dims[3] = {(cuuint64_t)w.size(2), (cuuint64_t)w.size(1), (cuuint64_t)w.size(0)};
  cuuint64_t st[2] = {(cuuint64_t)w.stride(1) * 4, (cuuint64_t)w.stride(0) * 4};
  cuuint32_t box[3] = {(cuuint32_t)b0, (cuuint32_t)b1, 1}, unit[3] = {1, 1, 1};
  CUresult r = cuTensorMapEncodeTiled(&m, CU_TENSOR_MAP_DATA_TYPE_UINT32, 3, w.data_ptr(), dims, st, box, unit,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "map"); return m;
}
torch::Tensor run(torch::Tensor w, int mode, int stages, int64_t units, int grid) {
  CUtensorMap m{};
  if (mode == 0) m = mk(w, 128, 64); else if (mode == 1) m = mk(w, 256, 32);
  auto t = torch::tensor({(int64_t)-1, (int64_t)0}, torch::dtype(torch::kInt64).device(w.device()));
  size_t smem = 1024 + stages * 33792;
  auto f = stages == 4 ? stream<4> : stages == 6 ? stream<6> : stream<3>;
  cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  f<<<grid, 288, smem>>>(m, mode, (const char*)w.data_ptr(), units, 0, (unsigned long long*)t.data_ptr());
  return t;
}
"""
CPP = "torch::Tensor run(torch::Tensor w, int mode, int stages, int64_t units, int grid);"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--numa-node", type=int, default=0)
    a = ap.parse_args()
    mod = load_inline("tma_probe", cpp_sources=CPP, cuda_sources=SRC, functions=["run"],
                      extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
                      extra_ldflags=["-lcuda"])
    E = 48  # experts' worth of slices: 48 x 3 MB (w13) / 1.5 MB (w2) -> 144 / 72 MB
    w13 = torch.empty((E, 384, 2048), dtype=torch.int32, device="cuda").random_()
    w2 = torch.empty((E, 32, 12288), dtype=torch.int32, device="cuda").random_()
    for name, mode, w, units in (("w13box", 0, w13, E * 96), ("w2box", 1, w2, E * 48),
                                 ("flat", 2, w13, E * 96)):
        for stages in (3, 4, 6):
            for grid in (132, 116):
                best = 0
                for _ in range(5):
                    t = mod.run(w, mode, stages, units, grid).cpu().tolist()
                    best = max(best, units * 32768 / (t[1] - t[0]))
                print(json.dumps({"pattern": name, "stages": stages, "grid": grid,
                                  "gbs": round(best)}), flush=True)


if __name__ == "__main__":
    main()

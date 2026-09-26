// TMA (1D cp.async.bulk) streaming probe for GH200: HBM and pinned Grace, one launch.
//
// CTAs [0, cold_ctas) stream a pinned host buffer through its UVA alias; the
// rest stream an HBM buffer. Each CTA runs a producer thread issuing bulk
// copies into a ring of `stages` shared-memory slots and a consumer warp that
// waits on each slot's full barrier, touches the data, and frees the slot.
// Every CTA records globaltimer start/end and its byte count, so the host gets
// per-tier bandwidth for tiers that run concurrently.
//
//   tma_probe <cold_ctas> <hot_ctas> <stage_kb> <stages> [copies_per_stage]
//
// Run NUMA-bound to the GPU's Grace node (numactl --cpunodebind --membind), or
// the host buffer lands on a remote node and C2C reads run ~5x slow.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <algorithm>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  printf("CUDA error %s at %d: %s\n", #x, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

__device__ __forceinline__ uint64_t gtimer() {
  uint64_t t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t;
}
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(bar)), "r"(count));
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
               :: "r"(smem_u32(bar)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
  asm volatile(
      "{\n .reg .pred p;\n WAIT_%=:\n"
      " mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      " @!p bra WAIT_%=;\n}\n" :: "r"(smem_u32(bar)), "r"(parity) : "memory");
}
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
      :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)) : "memory");
}

struct CtaStat { uint64_t start, end, bytes; uint32_t sink; };

__global__ void __launch_bounds__(64) stream_kernel(
    const char* __restrict__ cold_src, size_t cold_bytes,
    const char* __restrict__ hot_src, size_t hot_bytes,
    int cold_ctas, int stage_bytes, int stages, int copies, int iters, CtaStat* stats) {
  extern __shared__ __align__(128) unsigned char smem[];
  uint64_t* full = reinterpret_cast<uint64_t*>(smem);
  uint64_t* empty = full + stages;
  unsigned char* ring = smem + 128;
  const bool cold = blockIdx.x < cold_ctas;
  const int role_ctas = cold ? cold_ctas : gridDim.x - cold_ctas;
  const int role_index = cold ? blockIdx.x : blockIdx.x - cold_ctas;
  const char* base = cold ? cold_src : hot_src;
  size_t region = (cold ? cold_bytes : hot_bytes) / role_ctas;
  region -= region % stage_bytes;
  const char* mine = base + role_index * region;
  const int chunks = static_cast<int>(region / stage_bytes);
  const int total = chunks * iters;
  if (threadIdx.x == 0) {
    for (int s = 0; s < stages; ++s) { mbar_init(&full[s], 1); mbar_init(&empty[s], 1); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  uint64_t t0 = gtimer();
  uint32_t sink = 0;
  if (threadIdx.x == 0) {  // producer
    for (int i = 0; i < total; ++i) {
      int s = i % stages;
      if (i >= stages) mbar_wait(&empty[s], ((i / stages) - 1) & 1);
      mbar_expect_tx(&full[s], stage_bytes);
      const char* src = mine + static_cast<size_t>(i % chunks) * stage_bytes;
      unsigned char* dst = ring + static_cast<size_t>(s) * stage_bytes;
      int piece = stage_bytes / copies;
      for (int c = 0; c < copies; ++c) bulk_g2s(dst + c * piece, src + c * piece, piece, &full[s]);
    }
  } else if (threadIdx.x >= 32) {  // consumer warp
    int lane = threadIdx.x - 32;
    for (int i = 0; i < total; ++i) {
      int s = i % stages;
      mbar_wait(&full[s], (i / stages) & 1);
      const uint32_t* w = reinterpret_cast<const uint32_t*>(ring + static_cast<size_t>(s) * stage_bytes);
      sink += w[lane * 17 % (stage_bytes / 4)];
      __syncwarp();
      if (lane == 0) mbar_arrive(&empty[s]);
    }
  }
  __syncthreads();
  if (threadIdx.x == 32) {
    stats[blockIdx.x].start = t0;
    stats[blockIdx.x].end = gtimer();
    stats[blockIdx.x].bytes = static_cast<uint64_t>(total) * stage_bytes;
    stats[blockIdx.x].sink = sink;
  }
}

static void report(const char* tag, const std::vector<CtaStat>& st, int lo, int hi) {
  if (hi <= lo) return;
  uint64_t start = UINT64_MAX, end = 0, bytes = 0;
  for (int i = lo; i < hi; ++i) {
    start = std::min(start, st[i].start); end = std::max(end, st[i].end); bytes += st[i].bytes;
  }
  double s = (end - start) * 1e-9;
  printf("  %-5s ctas=%3d  %8.1f MB in %8.3f ms  -> %7.1f GB/s\n", tag, hi - lo, bytes / 1e6, s * 1e3,
         bytes / s / 1e9);
}

int main(int argc, char** argv) {
  if (argc < 5) { printf("usage: %s cold_ctas hot_ctas stage_kb stages [copies]\n", argv[0]); return 1; }
  int cold_ctas = atoi(argv[1]), hot_ctas = atoi(argv[2]);
  int stage_bytes = atoi(argv[3]) * 1024, stages = atoi(argv[4]);
  int copies = argc > 5 ? atoi(argv[5]) : 1;
  size_t cold_bytes = size_t(1) << 30, hot_bytes = size_t(4) << 30;
  char* host = static_cast<char*>(aligned_alloc(1 << 21, cold_bytes));
  memset(host, 1, cold_bytes);  // first touch on the bound NUMA node
  CK(cudaHostRegister(host, cold_bytes, cudaHostRegisterMapped));
  char* host_dev;
  CK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&host_dev), host, 0));
  char* hbm;
  CK(cudaMalloc(&hbm, hot_bytes));
  CK(cudaMemset(hbm, 1, hot_bytes));
  int ctas = cold_ctas + hot_ctas;
  CtaStat* d_stats;
  CK(cudaMalloc(&d_stats, sizeof(CtaStat) * ctas));
  size_t smem = 128 + static_cast<size_t>(stage_bytes) * stages;
  CK(cudaFuncSetAttribute(stream_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
  // Size iterations so each tier runs long enough to time; cold reads its
  // buffer once, hot loops over its 4 GiB buffer.
  int iters = 1;
  for (int rep = 0; rep < 3; ++rep) {
    stream_kernel<<<ctas, 64, smem>>>(host_dev, cold_bytes, hbm, hot_bytes, cold_ctas, stage_bytes,
                                      stages, copies, iters, d_stats);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
  }
  std::vector<CtaStat> st(ctas);
  CK(cudaMemcpy(st.data(), d_stats, sizeof(CtaStat) * ctas, cudaMemcpyDeviceToHost));
  printf("cold_ctas=%d hot_ctas=%d stage=%dKB stages=%d copies=%d (in flight/CTA %d KB)\n",
         cold_ctas, hot_ctas, stage_bytes / 1024, stages, copies, stage_bytes * stages / 1024);
  report("cold", st, 0, cold_ctas);
  report("hot", st, cold_ctas, ctas);
  return 0;
}

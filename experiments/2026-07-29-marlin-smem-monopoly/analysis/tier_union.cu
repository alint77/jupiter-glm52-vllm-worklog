// Production-shaped two-tier bandwidth probe, without Marlin.
//
// A ("hot")  streams ~256 MB from device HBM   (production: 19 experts, 246 MB)
// B ("cold") streams ~39 MB  from pinned Grace (production:  3 experts,  39 MB)
//
// Question: when A and B are genuinely co-resident on the same SMs, is the
// union ~= max(A,B) (different memory systems, so overlap is free), or do they
// destructively interfere (shared L2 / fabric)?
//
// Swept: dynamic shared memory per CTA (co-residency on/off) and L2 eviction
// hints on each tier's streaming loads.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

#define CK(x)                                                              \
  do {                                                                     \
    cudaError_t e = (x);                                                   \
    if (e != cudaSuccess) {                                                \
      printf("CUDA error %s at %d: %s\n", #x, __LINE__,                    \
             cudaGetErrorString(e));                                       \
      exit(1);                                                             \
    }                                                                      \
  } while (0)

struct CtaRec { unsigned long long start, stop; unsigned int smid, pad; };

__device__ __forceinline__ unsigned int smid_() {
  unsigned int r; asm volatile("mov.u32 %0, %%smid;" : "=r"(r)); return r;
}
__device__ __forceinline__ unsigned long long gtime() {
  unsigned long long r; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(r)); return r;
}
__device__ __forceinline__ float4 ld_normal(const float4* p) {
  float4 v;
  asm volatile("ld.global.nc.v4.f32 {%0,%1,%2,%3}, [%4];"
               : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(p));
  return v;
}
__device__ __forceinline__ unsigned long long evict_first_policy() {
  unsigned long long pol;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
  return pol;
}
__device__ __forceinline__ float4 ld_evict_first(const float4* p,
                                                 unsigned long long pol) {
  float4 v;
  asm volatile("ld.global.nc.L2::cache_hint.v4.f32 {%0,%1,%2,%3}, [%4], %5;"
               : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
               : "l"(p), "l"(pol));
  return v;
}

template <bool EVICT_FIRST>
__global__ __launch_bounds__(256) void streamer(const float4* __restrict__ src,
                                                size_t n4, float* sink,
                                                CtaRec* rec, int reps) {
  extern __shared__ char sh[];
  if (threadIdx.x == 0) { rec[blockIdx.x].start = gtime(); rec[blockIdx.x].smid = smid_(); }
  if (threadIdx.x == 0) sh[0] = 1;  // touch, keep the allocation real
  float acc = 0.f;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  unsigned long long pol = evict_first_policy();
  for (int r = 0; r < reps; ++r) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
      float4 v = EVICT_FIRST ? ld_evict_first(src + i, pol) : ld_normal(src + i);
      acc += v.x + v.y + v.z + v.w;
    }
  }
  if (threadIdx.x == 0) {
    rec[blockIdx.x].stop = gtime();
    if (acc == 1.2345e30f) sink[blockIdx.x] = acc;
  }
}

static void span(const std::vector<CtaRec>& r, unsigned long long* lo,
                 unsigned long long* hi) {
  *lo = ~0ull; *hi = 0;
  for (auto& c : r) { *lo = std::min(*lo, c.start); *hi = std::max(*hi, c.stop); }
}
static double ms(const std::vector<CtaRec>& r) {
  unsigned long long lo, hi; span(r, &lo, &hi); return (hi - lo) / 1e6;
}
static int peak_per_sm(const std::vector<CtaRec>& a, const std::vector<CtaRec>& b) {
  int peak = 0;
  std::vector<unsigned long long> pts;
  for (auto& c : a) pts.push_back(c.start);
  for (auto& c : b) pts.push_back(c.start);
  std::sort(pts.begin(), pts.end());
  for (auto t : pts) {
    int cnt[256] = {0};
    for (auto& c : a) if (c.start <= t && t < c.stop && c.smid < 256) cnt[c.smid]++;
    for (auto& c : b) if (c.start <= t && t < c.stop && c.smid < 256) cnt[c.smid]++;
    for (int i = 0; i < 256; ++i) peak = std::max(peak, cnt[i]);
  }
  return peak;
}

int main(int argc, char** argv) {
  CK(cudaSetDevice(0));
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
  int optin = 0;
  CK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
  CK(cudaFuncSetAttribute(streamer<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, optin));
  CK(cudaFuncSetAttribute(streamer<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, optin));

  const size_t A_BYTES = 246ull << 20;   // hot: 19 experts x 12.98 MB
  const size_t B_BYTES = 39ull << 20;    // cold: 3 experts x 12.98 MB
  const int gridA = 528, gridB = 264;  // allocation upper bound
  int gridA_v = 396, gridB_v = 132;
  const int repsA = argc > 1 ? atoi(argv[1]) : 4;
  const int repsB = repsA;

  float4 *dA, *hB;
  float* sink;
  CK(cudaMalloc(&dA, A_BYTES));
  CK(cudaHostAlloc(&hB, B_BYTES, cudaHostAllocMapped));  // pinned Grace
  CK(cudaMalloc(&sink, gridA * sizeof(float)));
  CK(cudaMemset(dA, 1, A_BYTES));
  for (size_t i = 0; i < B_BYTES / sizeof(float4); ++i) hB[i] = make_float4(1, 1, 1, 1);
  float4* dB = nullptr;
  CK(cudaHostGetDevicePointer((void**)&dB, hB, 0));

  cudaPointerAttributes pa; CK(cudaPointerGetAttributes(&pa, dB));
  printf("device %s SMs %d | A=%zu MB HBM, B=%zu MB pinned-host (type %d)\n",
         prop.name, prop.multiProcessorCount, A_BYTES >> 20, B_BYTES >> 20, (int)pa.type);

  CtaRec *rA, *rB;
  CK(cudaMalloc(&rA, gridA * sizeof(CtaRec)));
  CK(cudaMalloc(&rB, gridB * sizeof(CtaRec)));
  cudaStream_t s1, s2;
  CK(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking));
  CK(cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking));

  const size_t nA = A_BYTES / sizeof(float4), nB = B_BYTES / sizeof(float4);
  std::vector<CtaRec> hRA(gridA), hRB(gridB);

  auto launchA = [&](int smem, bool ef, cudaStream_t s) {
    if (ef) streamer<true><<<gridA_v, 256, smem, s>>>(dA, nA, sink, rA, repsA);
    else    streamer<false><<<gridA_v, 256, smem, s>>>(dA, nA, sink, rA, repsA);
  };
  auto launchB = [&](int smem, bool ef, cudaStream_t s) {
    if (ef) streamer<true><<<gridB_v, 256, smem, s>>>(dB, nB, sink, rB, repsB);
    else    streamer<false><<<gridB_v, 256, smem, s>>>(dB, nB, sink, rB, repsB);
  };

  printf("\n%8s %5s %6s | %8s %8s | %8s %8s | %8s %7s %7s %5s\n",
         "smemA/B", "gridA", "soloA", "GB/s", "soloB", "GB/s",
         "union", "dilA", "vs max", "peak");
  struct Cfg { int smemA, smemB, gA, gB; };
  Cfg cfgs[] = {{76458, 76458, 396, 132},   // production today: hot bps=3, cold bps=3
                {76458, 76458, 264, 132},   // hot 2 CTA/SM + cold 1 CTA/SM, SAME tiles
                {76458, 76458, 264, 264},   // same, cold 2 CTA/SM
                {50000, 32000, 396, 132},   // smaller tiles, hot keeps 3 CTA/SM
                {50000, 32000, 264, 132}};  // smaller tiles, hot 2 CTA/SM
 // hot with headroom
  for (Cfg cfg : cfgs) {
    for (int mode = 0; mode < 1; ++mode) {
      int smem = cfg.smemA;
      bool efA = false, efB = false;
      // warm
      gridA_v = cfg.gA; gridB_v = cfg.gB;
      launchA(cfg.smemA, efA, s1); launchB(cfg.smemB, efB, s2); CK(cudaDeviceSynchronize());

      launchA(cfg.smemA, efA, s1); CK(cudaDeviceSynchronize());
      hRA.resize(gridA_v); CK(cudaMemcpy(hRA.data(), rA, gridA_v * sizeof(CtaRec), cudaMemcpyDeviceToHost));
      double sA = ms(hRA);
      launchB(cfg.smemB, efB, s2); CK(cudaDeviceSynchronize());
      hRB.resize(gridB_v); CK(cudaMemcpy(hRB.data(), rB, gridB_v * sizeof(CtaRec), cudaMemcpyDeviceToHost));
      double sB = ms(hRB);

      launchA(cfg.smemA, efA, s1); launchB(cfg.smemB, efB, s2); CK(cudaDeviceSynchronize());
      hRA.resize(gridA_v); CK(cudaMemcpy(hRA.data(), rA, gridA_v * sizeof(CtaRec), cudaMemcpyDeviceToHost));
      hRB.resize(gridB_v); CK(cudaMemcpy(hRB.data(), rB, gridB_v * sizeof(CtaRec), cudaMemcpyDeviceToHost));
      unsigned long long alo, ahi, blo, bhi;
      span(hRA, &alo, &ahi); span(hRB, &blo, &bhi);
      double uni = (std::max(ahi, bhi) - std::min(alo, blo)) / 1e6;
      double dilA = ((ahi - alo) / 1e6) / sA - 1.0;
      double best = std::max(sA, sB);
      printf("%5d/%-5d %5d | %8.3f %8.1f | %8.3f %8.1f | %8.3f %+6.1f%% %+6.1f%% %5d\n",
             cfg.smemA, cfg.smemB, cfg.gA,
             sA, (double)A_BYTES * repsA / (sA * 1e-3) / 1e9,
             sB, (double)B_BYTES * repsB / (sB * 1e-3) / 1e9,
             uni, dilA * 100.0, (uni / best - 1.0) * 100.0,
             peak_per_sm(hRA, hRB));
    }
  }
  printf("\nsolo/union in ms (CTA globaltimer spans). dilA = A's own span growth when co-run.\n"
         "vs max = union against max(soloA,soloB), the ideal for two independent memory systems.\n"
         "peak = max CTAs resident on one SM across both kernels (2+ => genuine co-residency).\n");
  return 0;
}

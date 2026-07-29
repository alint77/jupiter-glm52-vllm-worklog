// Does Marlin's dynamic-shared-memory request prevent two kernels from ever
// sharing an SM?  Marlin launches with `deviceSharedMemPerSM / blocks_per_sm`
// bytes of dynamic shared memory per CTA (ops.cu:533), not with what the kernel
// actually needs.  If that is what pins occupancy, two concurrent launches can
// never be co-resident and their union is serial by construction.
//
// This probe reproduces the situation with two dummy kernels of identical shape
// and a swept dynamic-shared-memory request.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

#define CK(x)                                                                \
  do {                                                                       \
    cudaError_t e = (x);                                                     \
    if (e != cudaSuccess) {                                                  \
      printf("CUDA error %s at %d: %s\n", #x, __LINE__,                      \
             cudaGetErrorString(e));                                         \
      exit(1);                                                               \
    }                                                                        \
  } while (0)

struct CtaRec {
  unsigned long long start, stop;
  unsigned int smid;
  unsigned int pad;
};

__device__ __forceinline__ unsigned int smid() {
  unsigned int r;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(r));
  return r;
}
__device__ __forceinline__ unsigned long long gtime() {
  unsigned long long r;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(r));
  return r;
}

// Fixed-work kernel; touches its dynamic shared memory so the request is real.
__global__ __launch_bounds__(256) void worker(CtaRec* rec, int iters,
                                              float* sink, int smem_ints) {
  extern __shared__ int sh[];
  unsigned long long t0 = gtime();
  if (threadIdx.x == 0) {
    rec[blockIdx.x].start = t0;
    rec[blockIdx.x].smid = smid();
  }
  for (int i = threadIdx.x; i < smem_ints; i += blockDim.x) sh[i] = i;
  __syncthreads();
  float acc = 0.f;
  int idx = threadIdx.x % (smem_ints > 0 ? smem_ints : 1);
  for (int it = 0; it < iters; ++it) {
    acc += (float)sh[idx];
    idx = (idx * 7 + 13) % (smem_ints > 0 ? smem_ints : 1);
    acc = fmaf(acc, 1.000001f, 1.0f);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    rec[blockIdx.x].stop = gtime();
    if (acc == 12345.678f) sink[blockIdx.x] = acc;  // never true
  }
}

static double span_ms(const std::vector<CtaRec>& r) {
  unsigned long long lo = ~0ull, hi = 0;
  for (auto& c : r) {
    lo = std::min(lo, c.start);
    hi = std::max(hi, c.stop);
  }
  return (hi - lo) / 1e6;
}

// Fraction of A's busy span during which at least one B CTA is resident, and
// the peak number of CTAs resident on any single SM across both kernels.
static void coresidency(const std::vector<CtaRec>& a,
                        const std::vector<CtaRec>& b, double* overlap_frac,
                        int* peak_per_sm) {
  unsigned long long lo = ~0ull, hi = 0;
  for (auto& c : a) { lo = std::min(lo, c.start); hi = std::max(hi, c.stop); }
  unsigned long long blo = ~0ull, bhi = 0;
  for (auto& c : b) { blo = std::min(blo, c.start); bhi = std::max(bhi, c.stop); }
  unsigned long long ov_lo = std::max(lo, blo), ov_hi = std::min(hi, bhi);
  *overlap_frac = ov_hi > ov_lo ? double(ov_hi - ov_lo) / double(hi - lo) : 0.0;

  // peak concurrent CTAs on one SM, sampled on a grid of timestamps
  int peak = 0;
  std::vector<unsigned long long> pts;
  for (auto& c : a) pts.push_back(c.start);
  for (auto& c : b) pts.push_back(c.start);
  std::sort(pts.begin(), pts.end());
  for (auto t : pts) {
    int cnt[256] = {0};
    for (auto& c : a)
      if (c.start <= t && t < c.stop && c.smid < 256) cnt[c.smid]++;
    for (auto& c : b)
      if (c.start <= t && t < c.stop && c.smid < 256) cnt[c.smid]++;
    for (int i = 0; i < 256; ++i) peak = std::max(peak, cnt[i]);
  }
  *peak_per_sm = peak;
}

int main(int argc, char** argv) {
  int dev = 0;
  CK(cudaSetDevice(dev));
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, dev));
  int smem_per_sm = prop.sharedMemPerMultiprocessor;
  int smem_optin = 0;
  CK(cudaDeviceGetAttribute(&smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                            dev));
  printf("device %s, SMs %d, sharedMemPerMultiprocessor %d B, optin/block %d B\n",
         prop.name, prop.multiProcessorCount, smem_per_sm, smem_optin);

  CK(cudaFuncSetAttribute(worker, cudaFuncAttributeMaxDynamicSharedMemorySize,
                          smem_optin));

  const int gridA = 396;  // production hot grid: sms * blocks_per_sm = 132 * 3
  const int gridB = 132;  // a cold-tier-sized grid
  const int iters = argc > 1 ? atoi(argv[1]) : 200000;

  CtaRec *dA, *dB;
  float* sink;
  CK(cudaMalloc(&dA, gridA * sizeof(CtaRec)));
  CK(cudaMalloc(&dB, gridB * sizeof(CtaRec)));
  CK(cudaMalloc(&sink, gridA * sizeof(float)));
  cudaStream_t s1, s2;
  CK(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking));
  CK(cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking));

  // 76800 = 233472/3 - 1024, exactly what Marlin requests at blocks_per_sm=3
  int smems[] = {76800, 57344, 48000, 32768, 16384};
  printf("\n%9s %7s %8s %8s %8s %8s %7s %6s\n", "dynSmem", "occ/SM", "soloA_ms",
         "soloB_ms", "serial", "union_ms", "ov_frac", "peak");
  for (int smem : smems) {
    int smem_ints = smem / 4 - 4;
    int occ = 0;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, worker, 256, smem));

    std::vector<CtaRec> hA(gridA), hB(gridB);
    // warm
    worker<<<gridA, 256, smem, s1>>>(dA, iters / 10, sink, smem_ints);
    CK(cudaDeviceSynchronize());

    // solo A
    worker<<<gridA, 256, smem, s1>>>(dA, iters, sink, smem_ints);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(hA.data(), dA, gridA * sizeof(CtaRec), cudaMemcpyDeviceToHost));
    double soloA = span_ms(hA);

    // solo B
    worker<<<gridB, 256, smem, s2>>>(dB, iters, sink, smem_ints);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(hB.data(), dB, gridB * sizeof(CtaRec), cudaMemcpyDeviceToHost));
    double soloB = span_ms(hB);

    // concurrent, independent launches (no cross-stream waits)
    worker<<<gridA, 256, smem, s1>>>(dA, iters, sink, smem_ints);
    worker<<<gridB, 256, smem, s2>>>(dB, iters, sink, smem_ints);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(hA.data(), dA, gridA * sizeof(CtaRec), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hB.data(), dB, gridB * sizeof(CtaRec), cudaMemcpyDeviceToHost));
    unsigned long long lo = ~0ull, hi = 0;
    for (auto& c : hA) { lo = std::min(lo, c.start); hi = std::max(hi, c.stop); }
    for (auto& c : hB) { lo = std::min(lo, c.start); hi = std::max(hi, c.stop); }
    double uni = (hi - lo) / 1e6;
    double ovf;
    int peak;
    coresidency(hA, hB, &ovf, &peak);
    printf("%9d %7d %8.3f %8.3f %8.3f %8.3f %7.2f %6d\n", smem, occ, soloA,
           soloB, soloA + soloB, uni, ovf, peak);
  }
  printf("\nocc/SM = cudaOccupancyMaxActiveBlocksPerMultiprocessor for ONE kernel.\n"
         "peak   = max CTAs resident on any single SM across BOTH kernels.\n");
  return 0;
}

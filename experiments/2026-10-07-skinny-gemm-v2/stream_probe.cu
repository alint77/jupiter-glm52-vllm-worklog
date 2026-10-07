// How fast can one kernel stream a decode GEMM's weight from HBM?
// tma_stream: each CTA owns a contiguous slice; one thread issues 1D bulk
// copies (cp.async.bulk) of CHUNK bytes into a STAGES-deep shared-memory ring,
// completion on mbarriers; the other warps wait for a stage, read one 16 B
// word per thread from it (so the data is consumed), and release it.
// ldg_stream: every thread streams uint4 loads, UNROLL in flight.
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

__device__ __forceinline__ uint32_t s32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void bar_init(uint64_t* b, int n) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(s32(b)), "r"(n));
}
__device__ __forceinline__ void bar_expect(uint64_t* b, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(s32(b)), "r"(bytes)
               : "memory");
}
__device__ __forceinline__ void bar_arrive(uint64_t* b) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(s32(b)) : "memory");
}
__device__ __forceinline__ void bar_wait(uint64_t* b, int parity) {
  asm volatile(
      "{\n .reg .pred p;\n W_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      " @!p bra W_%=;\n}\n" ::"r"(s32(b)), "r"(parity)
      : "memory");
}
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* b) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
          s32(dst)),
      "l"(src), "r"(bytes), "r"(s32(b))
      : "memory");
}

__global__ void tma_stream(const char* __restrict__ src, int64_t nbytes, int chunk, int stages,
                           uint32_t* sink) {
  extern __shared__ __align__(128) char ring[];
  __shared__ uint64_t full[16], empty[16];
  const int64_t per = (nbytes / gridDim.x + 127) & ~int64_t(127);
  const int64_t lo = per * blockIdx.x, hi = min(nbytes, lo + per);
  const int n = lo < hi ? int((hi - lo + chunk - 1) / chunk) : 0;
  const int consumers = blockDim.x / 32 - 1;
  if (threadIdx.x == 0) {
    for (int s = 0; s < stages; ++s) {
      bar_init(&full[s], 1);
      bar_init(&empty[s], consumers);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  if (warp == 0) {
    if (lane == 0)
      for (int i = 0; i < n; ++i) {
        const int s = i % stages;
        if (i >= stages) bar_wait(&empty[s], ((i / stages) - 1) & 1);
        const int64_t off = lo + int64_t(i) * chunk;
        const int64_t rest = hi - off;
        const uint32_t bytes = uint32_t(rest < chunk ? rest : chunk);
        bar_expect(&full[s], bytes);
        bulk_g2s(ring + size_t(s) * chunk, src + off, bytes, &full[s]);
      }
  } else {
    uint32_t acc = 0;
    for (int i = 0; i < n; ++i) {
      const int s = i % stages;
      bar_wait(&full[s], (i / stages) & 1);
      const uint4 v = reinterpret_cast<const uint4*>(ring + size_t(s) * chunk)[threadIdx.x - 32];
      acc ^= v.x ^ v.w;
      __syncwarp();
      if (lane == 0) bar_arrive(&empty[s]);
    }
    if (acc == 0x12345678u) sink[0] = acc;
  }
}

template <int UNROLL>
__global__ void ldg_stream(const uint4* __restrict__ src, int64_t n16, uint32_t* sink) {
  uint32_t acc = 0;
  const int64_t stride = int64_t(gridDim.x) * blockDim.x;
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n16; i += stride * UNROLL) {
    uint4 v[UNROLL];
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      const int64_t j = i + u * stride;
      v[u] = j < n16 ? __ldcs(src + j) : make_uint4(0, 0, 0, 0);
    }
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) acc ^= v[u].x ^ v[u].w;
  }
  if (acc == 0x12345678u) sink[0] = acc;
}

void tma(torch::Tensor src, torch::Tensor sink, int64_t grid, int64_t chunk, int64_t stages,
         int64_t consumers) {
  const int smem = int(chunk * stages);
  cudaFuncSetAttribute(tma_stream, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  tma_stream<<<grid, (consumers + 1) * 32, smem, c10::cuda::getCurrentCUDAStream()>>>(
      static_cast<const char*>(src.data_ptr()), src.numel() * src.element_size(), int(chunk),
      int(stages), static_cast<uint32_t*>(sink.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void ldg(torch::Tensor src, torch::Tensor sink, int64_t grid, int64_t threads, int64_t unroll) {
  const int64_t n16 = src.numel() * src.element_size() / 16;
  auto st = c10::cuda::getCurrentCUDAStream();
  auto p = static_cast<const uint4*>(src.data_ptr());
  auto k = static_cast<uint32_t*>(sink.data_ptr());
  if (unroll == 4) ldg_stream<4><<<grid, threads, 0, st>>>(p, n16, k);
  else ldg_stream<8><<<grid, threads, 0, st>>>(p, n16, k);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("tma", &tma);
  m.def("ldg", &ldg);
}

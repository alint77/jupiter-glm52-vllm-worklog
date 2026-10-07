// v6 = v4 + programmatic dependent launch: the first weight batch is loaded
// before griddepcontrol.wait (weights do not depend on the previous kernel),
// activations after it; launch_dependents once the main loop is done.
// v4 = v2 with the next batch's loads issued before this batch's MMAs.
// y[M<=8, N] = x[M, K] @ W[N, K]^T, bf16 in, fp32 accumulate, bf16 out.
// The weight streams straight from HBM into registers with 16 B loads (no
// TMA: a TMA ring tops out near 2.4 TB/s here, plain loads reach ~2.9) and
// feeds mma.m16n8k16 with the weight as A (16 rows of N) and x^T as B (the 8
// tokens). Within each 32-wide K step thread (g, t) loads 8 consecutive k of
// rows g and g + 8 and of token g; a fixed permutation of K inside the step
// (the same for A and B, so the dot product is unchanged) makes those three
// 16 B loads exactly the operands of two MMAs. A CTA owns 16 rows; its WARPS
// warps split K and reduce through shared memory.
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

__device__ __forceinline__ void mma(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                    uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
      "{%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

template <int WARPS, int UNROLL, bool PDL>
__global__ void __launch_bounds__(WARPS * 32)
    skinny_v6(const __nv_bfloat16* __restrict__ W, const __nv_bfloat16* __restrict__ x,
              __nv_bfloat16* __restrict__ y, int M, int N, int K) {
  __shared__ float red[WARPS][32][4];
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane >> 2, t = lane & 3;
  const int n0 = blockIdx.x * 16;
  const int kper = K / WARPS;  // multiple of 32 * UNROLL
  const size_t k0 = size_t(warp) * kper;
  const uint4* w0 = reinterpret_cast<const uint4*>(W + size_t(n0 + g) * K + k0) + t;
  const uint4* w1 = reinterpret_cast<const uint4*>(W + size_t(n0 + g + 8) * K + k0) + t;
  const uint4* xr = reinterpret_cast<const uint4*>(x + size_t(g) * K + k0) + t;
  const bool tok = g < M;
  float c[4] = {0.f, 0.f, 0.f, 0.f};
  const int steps = kper / 32;  // multiple of UNROLL
  uint4 a0[UNROLL], b0[UNROLL], v0[UNROLL], a1[UNROLL], b1[UNROLL], v1[UNROLL];
#define LOAD(A, B, V, S)                                                   \
  _Pragma("unroll") for (int u = 0; u < UNROLL; ++u) {                     \
    A[u] = __ldcs(w0 + ((S) + u) * 4);                                     \
    B[u] = __ldcs(w1 + ((S) + u) * 4);                                     \
    V[u] = tok ? __ldg(xr + ((S) + u) * 4) : make_uint4(0, 0, 0, 0);      \
  }
#define MMA(A, B, V)                                                       \
  _Pragma("unroll") for (int u = 0; u < UNROLL; ++u) {                     \
    mma(c, A[u].x, B[u].x, A[u].y, B[u].y, V[u].x, V[u].y);                \
    mma(c, A[u].z, B[u].z, A[u].w, B[u].w, V[u].z, V[u].w);                \
  }
  _Pragma("unroll") for (int u = 0; u < UNROLL; ++u) {
    a0[u] = __ldcs(w0 + u * 4);
    b0[u] = __ldcs(w1 + u * 4);
  }
  if (PDL) asm volatile("griddepcontrol.wait;" ::: "memory");
  _Pragma("unroll") for (int u = 0; u < UNROLL; ++u)
    v0[u] = tok ? __ldg(xr + u * 4) : make_uint4(0, 0, 0, 0);
  for (int s = 0; s < steps; s += 2 * UNROLL) {
    if (s + UNROLL < steps) LOAD(a1, b1, v1, s + UNROLL)
    MMA(a0, b0, v0)
    if (s + 2 * UNROLL < steps) LOAD(a0, b0, v0, s + 2 * UNROLL)
    if (s + UNROLL < steps) MMA(a1, b1, v1)
  }
#undef LOAD
#undef MMA
  if (PDL) asm volatile("griddepcontrol.launch_dependents;");
#pragma unroll
  for (int i = 0; i < 4; ++i) red[warp][lane][i] = c[i];
  __syncthreads();
  if (warp == 0) {
#pragma unroll
    for (int w = 1; w < WARPS; ++w)
#pragma unroll
      for (int i = 0; i < 4; ++i) c[i] += red[w][lane][i];
    // c0, c1: row n0 + g, tokens 2t, 2t + 1; c2, c3: row n0 + g + 8
    if (2 * t < M) {
      y[size_t(2 * t) * N + n0 + g] = __float2bfloat16(c[0]);
      y[size_t(2 * t) * N + n0 + g + 8] = __float2bfloat16(c[2]);
    }
    if (2 * t + 1 < M) {
      y[size_t(2 * t + 1) * N + n0 + g] = __float2bfloat16(c[1]);
      y[size_t(2 * t + 1) * N + n0 + g + 8] = __float2bfloat16(c[3]);
    }
  }
}

template <int WARPS, int UNROLL>
void launch(const torch::Tensor& x, const torch::Tensor& w, torch::Tensor& y, bool pdl) {
  const int M = x.size(0), K = x.size(1), N = w.size(0);
  TORCH_CHECK(M <= 8 && N % 16 == 0 && K % (32 * WARPS * UNROLL) == 0, "shape");
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(N / 16);
  cfg.blockDim = dim3(WARPS * 32);
  cfg.stream = c10::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr;
  attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr.val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = &attr;
  cfg.numAttrs = pdl ? 1 : 0;
  auto W_ = reinterpret_cast<const __nv_bfloat16*>(w.data_ptr());
  auto X_ = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  auto Y_ = reinterpret_cast<__nv_bfloat16*>(y.data_ptr());
  if (pdl)
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, skinny_v6<WARPS, UNROLL, true>, W_, X_, Y_, M, N, K));
  else
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, skinny_v6<WARPS, UNROLL, false>, W_, X_, Y_, M, N, K));
}

torch::Tensor gemm(torch::Tensor x, torch::Tensor w, int64_t warps, int64_t unroll, bool pdl) {
  auto y = torch::empty({x.size(0), w.size(0)}, x.options());
#define CASE(W_, U_) \
  if (warps == W_ && unroll == U_) { launch<W_, U_>(x, w, y, pdl); return y; }
  CASE(2, 4) CASE(2, 8) CASE(4, 2) CASE(4, 4) CASE(4, 8) CASE(8, 2) CASE(8, 4) CASE(16, 2)
#undef CASE
  TORCH_CHECK(false, "unsupported config");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("gemm", &gemm); }

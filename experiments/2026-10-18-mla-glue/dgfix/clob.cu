// M <= 32 bf16 GEMM for the decode/verify step: y[M, N] = x[M, K] @ W[N, K]^T.
// At these M the work is streaming W; cuBLAS reaches ~2.3 TB/s on these
// shapes at M=8, this kernel ~2.8. Weights load straight from HBM into
// registers with 16 B loads (a TMA ring tops out near 2.4 TB/s here) and feed
// mma.m16n8k16 with W as A (16 rows of N) and x^T as B (8 tokens per tile, NT
// tiles; each loaded weight fragment feeds all NT). Within each 32-wide K step
// thread (g, t) loads 8 consecutive k of rows g and g + 8 and of tokens
// g + 8j; a fixed permutation of K inside the step, the same for W and x,
// makes those 16 B loads exactly the operands of two MMAs per tile. A CTA owns
// 16 rows; its WARPS warps split K, double-buffer their loads in registers and
// reduce through shared memory. Programmatic dependent launch: the first weight
// batch is loaded before griddepcontrol.wait, activations after it, with
// volatile loads (__ldg's ld.global.nc may be hoisted above the wait).
// Derivation and measurements:
// agent_space/experiments/2026-10-07-skinny-gemm-v2.
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

__device__ __forceinline__ void mma(float* c, uint32_t a0, uint32_t a1,
                                    uint32_t a2, uint32_t a3, uint32_t b0,
                                    uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, "
      "{%4,%5,%6,%7}, "
      "{%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint4 ldx(const uint4* p) {
  uint4 v;
  asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p)
               : "memory");
  return v;
}

template <int WARPS, int UNROLL, int NT, bool PDL>
__global__ void __launch_bounds__(WARPS * 32)
    decode_gemm_kernel(const __nv_bfloat16* __restrict__ W,
                       const __nv_bfloat16* __restrict__ x,
                       __nv_bfloat16* __restrict__ y, int M, int N, int K) {
  __shared__ float red[WARPS][32][4 * NT];
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane >> 2,
            t = lane & 3;
  const int n0 = blockIdx.x * 16;
  const int kper = K / WARPS;  // multiple of 32 * UNROLL
  const size_t k0 = size_t(warp) * kper;
  const uint4* wr =
      reinterpret_cast<const uint4*>(W + size_t(n0 + g) * K + k0) + t;
  const size_t wstep = size_t(8) * K / 8;  // 8 rows of W, in uint4
  const uint4* xr = reinterpret_cast<const uint4*>(x + size_t(g) * K + k0) + t;
  bool tok[NT];
#pragma unroll
  for (int j = 0; j < NT; ++j) tok[j] = g + 8 * j < M;
  float c[NT][4] = {};
  const int steps = kper / 32;  // multiple of UNROLL
  // weights: rows g, g + 8 of the CTA's 16; x: tokens g + 8j
  uint4 a0[2][UNROLL], v0[NT][UNROLL], a1[2][UNROLL], v1[NT][UNROLL];
#define LOADW(A, S)                                                \
  _Pragma("unroll") for (int i = 0; i < 2; ++i)                    \
      _Pragma("unroll") for (int u = 0; u < UNROLL; ++u) A[i][u] = \
          __ldcs(wr + i * wstep + ((S) + u) * 4);
#define LOADX(V, S)                                                \
  _Pragma("unroll") for (int j = 0; j < NT; ++j)                   \
      _Pragma("unroll") for (int u = 0; u < UNROLL; ++u) V[j][u] = \
          tok[j] ? ldx(xr + j * wstep + ((S) + u) * 4)            \
                 : make_uint4(0, 0, 0, 0);
#define MMA(A, V)                                                    \
  _Pragma("unroll") for (int u = 0; u < UNROLL; ++u)                 \
      _Pragma("unroll") for (int j = 0; j < NT; ++j) {               \
    mma(c[j], A[0][u].x, A[1][u].x, A[0][u].y, A[1][u].y, V[j][u].x, \
        V[j][u].y);                                                  \
    mma(c[j], A[0][u].z, A[1][u].z, A[0][u].w, A[1][u].w, V[j][u].z, \
        V[j][u].w);                                                  \
  }
  LOADW(a0, 0)
  if (PDL) asm volatile("griddepcontrol.wait;" ::: "memory");
  LOADX(v0, 0)
  for (int s = 0; s < steps; s += 2 * UNROLL) {
    if (s + UNROLL < steps) {
      LOADW(a1, s + UNROLL)
      LOADX(v1, s + UNROLL)
    }
    MMA(a0, v0)
    if (s + 2 * UNROLL < steps) {
      LOADW(a0, s + 2 * UNROLL)
      LOADX(v0, s + 2 * UNROLL)
    }
    if (s + UNROLL < steps) MMA(a1, v1)
  }
#undef LOADW
#undef LOADX
#undef MMA
  if (PDL) asm volatile("griddepcontrol.launch_dependents;");
#pragma unroll
  for (int j = 0; j < NT; ++j)
#pragma unroll
    for (int i = 0; i < 4; ++i) red[warp][lane][4 * j + i] = c[j][i];
  __syncthreads();
  if (warp == 0) {
#pragma unroll
    for (int w = 1; w < WARPS; ++w)
#pragma unroll
      for (int j = 0; j < NT; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) c[j][i] += red[w][lane][4 * j + i];
    // tile j: c0, c1 row n0 + g, tokens 8j + 2t, +1; c2, c3 row n0 + g + 8
#pragma unroll
    for (int j = 0; j < NT; ++j) {
      const int m = 8 * j + 2 * t;
      if (m < M) {
        y[size_t(m) * N + n0 + g] = __float2bfloat16(c[j][0]);
        y[size_t(m) * N + n0 + g + 8] = __float2bfloat16(c[j][2]);
      }
      if (m + 1 < M) {
        y[size_t(m + 1) * N + n0 + g] = __float2bfloat16(c[j][1]);
        y[size_t(m + 1) * N + n0 + g + 8] = __float2bfloat16(c[j][3]);
      }
    }
  }
}

template <int WARPS, int UNROLL, int NT>
void launch(const torch::Tensor& x, const torch::Tensor& w, torch::Tensor& y,
            bool pdl) {
  const int M = x.size(0), K = x.size(1), N = w.size(0);
  TORCH_CHECK(M <= 8 * NT && N % 16 == 0 && K % (32 * WARPS * UNROLL) == 0,
              "shape");
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
    C10_CUDA_CHECK(
        cudaLaunchKernelEx(&cfg, decode_gemm_kernel<WARPS, UNROLL, NT, true>,
                           W_, X_, Y_, M, N, K));
  else
    C10_CUDA_CHECK(
        cudaLaunchKernelEx(&cfg, decode_gemm_kernel<WARPS, UNROLL, NT, false>,
                           W_, X_, Y_, M, N, K));
}

// token tiles follow from M; the configs measured best at some M (see
// decode_gemm.config)
torch::Tensor gemm(torch::Tensor x, torch::Tensor w, int64_t warps,
                   int64_t unroll, bool pdl) {
  auto y = torch::empty({x.size(0), w.size(0)}, x.options());
  const int nt = (static_cast<int>(x.size(0)) + 7) / 8;
#define CASE(W_, U_, NT_)                         \
  if (warps == W_ && unroll == U_ && nt == NT_) { \
    launch<W_, U_, NT_>(x, w, y, pdl);            \
    return y;                                     \
  }
#define CASES(NT_) \
  CASE(2, 4, NT_); \
  CASE(2, 8, NT_); \
  CASE(4, 2, NT_); \
  CASE(4, 4, NT_); \
  CASE(8, 2, NT_); \
  CASE(8, 4, NT_);
  CASES(1)
  CASES(2)
  CASES(3)
  CASES(4)
#undef CASES
#undef CASE
  TORCH_CHECK(false, "unsupported config");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("gemm", &gemm); }

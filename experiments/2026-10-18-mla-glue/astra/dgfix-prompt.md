# Review request: decode_gemm PDL fix (do not run commands; your sandbox cannot run here — reason from the material below)

Context: GLM-5.3 decode on GH200 (sm_90a), vLLM. Your earlier race review (race-review.md) ranked candidates; the root cause was then found:
`decode_gemm_kernel<..., PDL=true>` loads its activations x with `__ldg` (ld.global.nc / SASS `LDG.E.128.CONSTANT`) and nvcc hoisted
some of those loads above `asm volatile("griddepcontrol.wait;" ::: "memory")`. The new glue kernel (mla_qkv_glue) issues
`griddepcontrol.launch_dependents` at its entry and writes q_c (the input of the PDL q_b decode_gemm), so q_b read stale q_c.
SASS of the served .so (before the fix): every PDL instantiation had 2-32 `LDG.E.128.CONSTANT` before `ACQBULK`.

## The fix (diff of vllm/model_executor/layers/decode_gemm/decode_gemm.cu)
```diff
--- /e/home/jusers/naeimitabiei1/jupiter/.claude/jobs/14fe1b4a/tmp/dgfix/old/decode_gemm.cu	2026-10-23 14:15:58.430873000 +0200
+++ /e/home/jusers/naeimitabiei1/jupiter/.claude/jobs/14fe1b4a/tmp/dgfix/new/decode_gemm.cu	2026-10-23 14:16:00.104916000 +0200
@@ -9,7 +9,8 @@
 // makes those 16 B loads exactly the operands of two MMAs per tile. A CTA owns
 // 16 rows; its WARPS warps split K, double-buffer their loads in registers and
 // reduce through shared memory. Programmatic dependent launch: the first weight
-// batch is loaded before griddepcontrol.wait, activations after it.
+// batch is loaded before griddepcontrol.wait, activations after it, with
+// volatile loads (__ldg's ld.global.nc may be hoisted above the wait).
 // Derivation and measurements:
 // agent_space/experiments/2026-10-07-skinny-gemm-v2.
 #include <c10/cuda/CUDAStream.h>
@@ -27,6 +28,14 @@
       : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
 }
 
+__device__ __forceinline__ uint4 ldx(const uint4* p) {
+  uint4 v;
+  asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];"
+               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
+               : "l"(p));
+  return v;
+}
+
 template <int WARPS, int UNROLL, int NT, bool PDL>
 __global__ void __launch_bounds__(WARPS * 32)
     decode_gemm_kernel(const __nv_bfloat16* __restrict__ W,
@@ -56,7 +65,7 @@
 #define LOADX(V, S)                                                \
   _Pragma("unroll") for (int j = 0; j < NT; ++j)                   \
       _Pragma("unroll") for (int u = 0; u < UNROLL; ++u) V[j][u] = \
-          tok[j] ? __ldg(xr + j * wstep + ((S) + u) * 4)           \
+          tok[j] ? ldx(xr + j * wstep + ((S) + u) * 4)            \
                  : make_uint4(0, 0, 0, 0);
 #define MMA(A, V)                                                    \
   _Pragma("unroll") for (int u = 0; u < UNROLL; ++u)                 \
```

## Fixed file in full
```cuda
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
               : "l"(p));
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
```

## Race test (login GH200, correctness only): producer.cu
```cuda
// PDL primary for the race test: lets dependents launch at entry, then waits
// `spin` cycles and only then writes x, like a slow upstream kernel.
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

__global__ void produce_kernel(const uint4* src, uint4* dst, size_t n,
                               long long spin) {
  asm volatile("griddepcontrol.launch_dependents;");
  const long long t0 = clock64();
  while (clock64() - t0 < spin) {
  }
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += size_t(gridDim.x) * blockDim.x)
    dst[i] = src[i];
}

void produce(torch::Tensor src, torch::Tensor dst, int64_t spin) {
  const size_t n = src.numel() * src.element_size() / 16;
  produce_kernel<<<4, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const uint4*>(src.data_ptr()),
      reinterpret_cast<uint4*>(dst.data_ptr()), n, spin);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("produce", &produce); }
```
## race_test.py
```python
"""PDL race test for decode_gemm: a primary that triggers dependents at entry
and writes x late, then decode_gemm(pdl=True) on x. Its output must equal the
serialized (pdl=False) result on the same data, bit for bit.

    race_test.py <dir with decode_gemm.cu> [iters]
"""
import sys
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

src_dir = Path(sys.argv[1]).resolve()
iters = int(sys.argv[2]) if len(sys.argv) > 2 else 200
flags = ["-O3", "-gencode=arch=compute_90a,code=sm_90a"]
dg = load(name=f"dg_{src_dir.name}", sources=[str(src_dir / "decode_gemm.cu")],
          extra_cuda_cflags=flags, build_directory=str(src_dir))
pb = Path(__file__).resolve().parent / "pbuild"
pb.mkdir(exist_ok=True)
pr = load(name="dg_producer", sources=[str(pb.parent / "producer.cu")],
          extra_cuda_cflags=flags, build_directory=str(pb))

torch.manual_seed(0)
SHAPES = {"q_b": (4096, 2048), "qkv_a": (2112, 6144)}  # (N, K)
CFGS = [(2, 4), (2, 8), (4, 2), (4, 4), (8, 2), (8, 4)]
SENTINEL = 1e4
bad_total = 0
for name, (n, k) in SHAPES.items():
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02
    for m in (4, 8, 16, 32):
        src = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        x = torch.empty_like(src)
        for cfg in CFGS:
            if k % (32 * cfg[0] * cfg[1]):
                continue
            ref = dg.gemm(src, w, *cfg, False)
            bad = 0
            for _ in range(iters):
                x.fill_(SENTINEL)
                pr.produce(src, x, 20000)
                y = dg.gemm(x, w, *cfg, True)
                bad += int(not torch.equal(y, ref))
            # same under CUDA graph replay
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                x.fill_(SENTINEL)
                pr.produce(src, x, 20000)
                yg = dg.gemm(x, w, *cfg, True)
            gbad = 0
            for _ in range(iters):
                g.replay()
                gbad += int(not torch.equal(yg, ref))
            torch.cuda.synchronize()
            bad_total += bad + gbad
            print(f"{src_dir.name} {name:5s} M={m:2d} cfg={cfg}  eager {bad:4d}/{iters}"
                  f"  graph {gbad:4d}/{iters}", flush=True)
print(f"{src_dir.name}: {bad_total} mismatching runs")
```

## Results
- old build: 9582 of 9600 runs mismatch (every shape q_b 4096x2048 / qkv_a 2112x6144, M = 4/8/16/32, all 6 configs, eager and CUDA graph).
- fixed build: 0 of 9600 mismatch.
- SASS of the fixed build: in all 24 PDL instantiations zero global loads other than the weight `LDG.E.EF.128` precede `ACQBULK`;
  x loads are now `LDG.E.128` (1440 of them), weights `LDG.E.EF.128` (1152).
- Other PDL consumers we own, audited the same way in SASS (any global load before the first ACQBULK): sliced decode layer_kernel,
  route_prep, finalize, mla_glue kernel: 0. tiered_prefill gemm_xres_kernel: 1 `LDG.CONSTANT` before ACQBULK, but there only
  the tail CTA waits by design (grouped launches over disjoint rows; the unit list is written by a non-PDL predecessor).
- Served probes (prod vs prod+fix vs glue+fix) are running now.

## Questions
1. Is the fix correct and sufficient under the PTX memory model? Specifically: does `griddepcontrol.wait` guarantee visibility of the
   primary grid's writes to plain `ld.global` (default .ca on sm_90: may hit L1) issued after it, or should x use `.cg`/`.relaxed.gpu`?
   Can any L1 line for x be stale at that point (other CTAs of this kernel on the same SM only load x after their own wait; the L1 is
   not flushed when a PDL secondary starts early)?
2. Is `asm volatile` without a "memory" clobber enough to keep these loads below the wait (the wait asm has a "memory" clobber and both
   are volatile)? Any risk nvcc/ptxas reorders them anyway (ptxas scheduling of ld vs griddepcontrol.wait)?
3. Performance: x loads were the read-only path before. Any reason the new loads should cost measurable time at M <= 32
   (x is <= 32 x 6144 bf16, re-read by all N/16 CTAs, L2-resident)? I will bench on Booster.
4. Weight loads `__ldcs` before the wait stay: weights are never written by a predecessor. Agree?
5. The test: does it actually exercise the hazard (producer with 4 CTAs that spins 20000 cycles then writes x), and is a bit-exact compare
   against pdl=False on the same kernel valid? Anything missing (e.g. a case where the primary writes x early and the hoisted load
   would see only part)?
6. Anything else in this kernel or the PDL pattern ("launch_dependents at the producer's entry") that I should fix while here?
Be concise; rank issues by severity.

# Review request: fusing GLM-5.3 decode dense GEMMs into their neighbours (plan check)

**Do not run any commands** (your sandbox cannot start on this machine). Reason from the material below; say where you'd need a measurement.

## Setting
GLM-5.3 (DSA/MLA, 78 layers, 75 MoE) served on 4x GH200 (sm_90a, 132 SMs, HBM ~3.6 TB/s measured streaming roof, ~60 MiB L2), TP4 + DCP4, MTP3 speculative decode: verify step M = 8 tokens at c=1, up to M=64 at c=16. All weights bf16 for these GEMMs. Hard rule: **same ops only** — identical math (bf16 rounding points preserved, no algebraic refactoring such as applying 1/rms in the epilogue); fp32 accumulation order changes inside a GEMM are acceptable (configs already differ). Optimizing decode only.

Model dims (per rank, TP4): hidden 6144; q_lora_rank 2048; kv_lora_rank 512; qk_rope 64; qk_nope 192 (qk_head 256); v_head_dim 256; 64 heads -> 16 local; indexer 32 heads x 128 (wq_b replicated: 2048 -> 4096).
Per-rank GEMMs (weight [N, K] bf16):
- fused_qkv_a: N = 2048 + 576 = 2624, K = 6144 (32.5 MB), replicated
- q_b: N = 16*256 = 4096, K = 2048 (16.8 MB)
- indexer wq_b: N = 4096, K = 2048 (16.8 MB), 21 anchor layers only, runs on a side stream
- W_UV: bmm per local head, [16 heads] x (8 x 512) @ (512 x 256) (16*512*256*2 = 4 MB)
- o_proj: N = 6144, K = 16*256 = 4096 (50.3 MB)

## Measured per-layer main-stream sequence (2026-10-08 trace, c=1, M=8, rank 0, CUDA graph; start us rel, dur us)
```
  0.0 14.9 FlashMLA sparse decode
 11.4  6.3 FlashMLA split combine          (since fused with the next one into one DCP-tail kernel)
 18.1  6.3 one_shot::lse_reduce_scatter   (cross-GPU)
 25.0  3.9 nvjet bmm  = W_UV
 29.1 16.5 decode_gemm = o_proj
 45.1  5.8 flashinfer trtllm allreduce_fusion oneshot_lamport (attn AR + residual + RMSNorm)
 ... router / grouped_topk / route_prep / tiered MoE w13 / act / w2 / finalize ... (MoE kernel since replaced by one TP-sliced persistent kernel)
140.4  1.6 triton_poi_fused_add_mul_0
142.4 61.1 flashinfer lamport AR (MoE AR + residual + RMSNorm; includes cross-rank wait)
203.3 12.5 decode_gemm = fused_qkv_a
216.3  1.5 triton_poi_fused_1             (unattributed; likely split / contiguous copy of q_c)
218.1  2.2 triton_red_fused_2             (unattributed; likely q_a + kv_a RMSNorm)
220.4  7.6 decode_gemm = q_b
228.5  2.3 triton fused rope/split/index_select...
231.3  1.9 concat_and_cache_ds_mla x2
235.8  3.6 nvjet NNT (W_UK absorb)
239.9  8.2 one_shot::gather_cat (DCP query all-gather) -> next layer
```
Back-to-back starts show ~0.1-0.2 us gaps (graphs + PDL). decode_gemm in-model SOL: o_proj 83%, qkv_a 70%, q_b 60% of HBM floor. Fitted streaming cost: plain 16 B loads ~3.3 us + MB/3.49 TB/s; TMA bulk 8.0 us + MB/3.65 TB/s. Earlier results: PDL on decode_gemm (first weight batch loaded before griddepcontrol.wait) gained only 0.2-0.5 us/call; split-K across CTAs gained nothing; an L2-prefetch of o_proj weights during the ~12 us DCP-tail/W_UV window made o_proj -5 us but slowed W_UV and the reduce-scatter by the same 5 us (no layer gain); prefetching qkv_a during the MoE slows HBM-bound MoE; "the step follows the slowest rank, which has no all-reduce wait to hide a prefetch in". With the sliced MoE, rank skew at the AR is ~2.1 us/call at c=1. Our decode_gemm fills all SMs, and when used for the side-stream shared expert it delayed overlapping kernels (router +7.7 us, MoE +3.9 us), so those stay on cuBLAS. For M=32-64 most of these GEMMs fall back to cuBLAS (config() below) at 50-63% SOL.

## The plan to review (as I gave it to the user)
1. q-chain (qkv_a -> split -> norm -> q_b; 24.7 us measured vs 13.5 us floor for 49 MB):
   a. Fold the q_a RMSNorm (and the split glue) into q_b's prologue: q_c is 8 x 2048 bf16 = 32 KB; every q_b CTA recomputes the norm for its tokens from L2 (q_b's decode_gemm grid is N/16 = 256 CTAs, each already reads the whole x). One CTA still writes normalized q_c for the indexer (wq_b, side stream). Must reproduce the Triton norm's reduction order for bitwise identity. Claimed ~4 us/layer, ~0.3 ms/step.
   b. qkv_a + q_b in one persistent kernel: dynamic work queue (qkv_a tiles then q_b tiles), last-arriver counter on q_c completion; CTAs that run out of qkv_a work start loading q_b weights while waiting (16.8 MB fits in aggregate smem). Claimed safe against non-co-residency because claims are queue-ordered (same pattern as our MoE kernel: an item is claimed only by a running CTA, and q_b items are reached only after all qkv_a items are claimed). Claimed ~3 us/layer.
2. o-chain (W_UV -> o_proj; 20.6 us vs ~14.4 us floor): fuse W_UV into o_proj's prologue, claiming no grid sync because o_proj's K (heads x 256) aligns with per-head W_UV: "a CTA handling head h's K-slice only needs head h's W_UV output". Claimed ~3.9 us/layer, ~0.25-0.3 ms/step.
3. MoE AR + residual + RMSNorm -> qkv_a: judged not worth it (cross-GPU lamport kernel to own; prefetch-in-the-wait doesn't help the slowest rank; 1/rms-in-epilogue is forbidden by same-ops).
4. Bottom line claimed: 1 + 2 recover ~0.6-0.8 ms/step at c=1 (~3-4% of 18.8 ms), inferred not measured. First step: norm into q_b. At c>=8 these GEMMs are on cuBLAS so fusion needs decode_gemm extended to M=64 first.

## Issue I noticed myself after giving the plan (please assess)
decode_gemm's CTA owns 16 rows of N and its warps split the **whole K** (see source). So for o_proj (384 CTAs) every CTA needs v for all 16 heads; a W_UV prologue would make each CTA recompute all of W_UV (4 MB of W_UV weight reads per CTA from L2 -> ~1.5 GB L2 traffic): infeasible as stated. Making "head h's K-slice" a CTA's unit means split-K over heads (16 partials per output), i.e. a cross-CTA reduction (fp32 red/atomics or workspace + last-arriver), which is what v5 split-K tried without gain. Also for 1a: qkv_a's output row is 2624 wide, so q_c is a strided slice (decode_gemm requires contiguous x; triton_poi_fused_1 may be that copy).

## Questions
1. Is each fusion feasible as described, and are the dependency claims right (especially 1b's deadlock-freedom argument with PDL / side streams / CUDA graphs, and 2's)? For 2, what is the best structure: split-K over heads with an in-kernel reduce, W_UV as a separate phase in a persistent kernel with a per-head-ready flag, or not worth it?
2. Are the gain estimates credible? What is the real mechanism of each saving (removed tiny kernels' durations vs. hidden ramp vs. HBM idle time), and which is double-counted? Note qkv_a->q_b and W_UV->o_proj already start within 0.2 us of each other; is the per-boundary ~3 us ramp actually paid at a boundary where the dependent is PDL-launched and loads weights before griddepcontrol.wait? (Do Triton kernels trigger launch_dependents? If not, does q_b's PDL prologue overlap anything?)
3. Same-ops pitfalls: bitwise identity of a re-implemented RMSNorm vs Triton's (eps, fp32 accumulate, reduction order, bf16 rounding of output before q_b), W_UV output bf16 rounding before o_proj, indexer consumer of normalized q_c, kv_a norm still needed.
4. Cheaper alternatives I missed (e.g., make the two Triton glue kernels PDL-aware or merge them; fold kv_a norm/rope/concat_and_cache into one kernel; concatenate q_b + indexer wq_b along N on anchor layers despite the side stream; fuse W_UK absorb or the DCP gather; extending decode_gemm to M=64 first as the bigger lever).
5. Ranked recommendation: what to build first, what to measure first (on a Booster node; login node GPU is not valid for perf), and what to drop.

Output: ranked findings (severity, claim, reasoning, recommendation), then a recommended plan.

## Source: vllm/model_executor/layers/decode_gemm/__init__.py (dispatch/config)
```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""bf16 linear layers at decode sizes (M <= 32) as a weight stream (see
decode_gemm.cu). Registered as vllm::decode_unquantized_gemm, which falls back
to F.linear for larger M, other dtypes and weights it was not tuned for, so
torch.compile never branches on the batch size. Built on first use for sm_90a,
cached under VLLM_CACHE_ROOT."""

from functools import cache
from pathlib import Path

import torch
import torch.nn.functional as F

import vllm.envs as envs
from vllm.utils.torch_utils import direct_register_custom_op

_HERE = Path(__file__).resolve().parent
MAX_M = 32
# lm_head and larger weights already stream at the HBM floor with cuBLAS
MAX_WEIGHT_BYTES = 128 << 20


@cache
def _extension():
    from torch.utils.cpp_extension import load

    build = Path(envs.VLLM_CACHE_ROOT) / "torch_extensions" / "decode_gemm"
    build.mkdir(parents=True, exist_ok=True)
    return load(
        name="vllm_decode_gemm",
        sources=[str(_HERE / "decode_gemm.cu")],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
        build_directory=str(build),
    )


def config(n: int, k: int, m: int = 8) -> tuple[int, int] | None:
    """(warps per 16-row tile, K steps per load batch) measured best on GH200
    for the GLM-5.3 verify-step shapes at M = m tokens, or None to keep cuBLAS
    (agent_space/experiments/2026-10-08-m32)."""
    # Small weights (N <= 1024 or K <= 512: the shared expert, which runs on a
    # side stream) stay on cuBLAS: this kernel fills every SM and in the
    # model delayed the router / MoE kernels it overlaps by more than it saved.
    if n % 16 or n <= 1024 or k <= 512 or n * k * 2 > MAX_WEIGHT_BYTES:
        return None
    cfg: tuple[int, int] | None
    if m <= 8:
        if k <= 2048 or (n == k and k >= 6144):
            cfg = (2, 8)
        elif k <= 3072:
            cfg = (2, 4)
        else:
            cfg = (4, 4)
    elif k <= 2048 and n != k:  # q_b: cuBLAS is level from 17 tokens
        cfg = (2, 4) if m <= 16 else None
    elif k <= 3072 or (n == k and k >= 6144):  # dense MLP down / gate_up
        cfg = (2, 4)
    elif m <= 16:
        cfg = (4, 4)
    elif k <= 4096:  # o_proj
        cfg = (4, 2) if m <= 24 else (2, 4)
    else:  # fused qkv_a: cuBLAS is level at 32
        cfg = (4, 4) if m <= 24 else None
    return cfg if cfg is not None and k % (32 * cfg[0] * cfg[1]) == 0 else None


def _impl(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    cfg = (
        config(weight.shape[0], weight.shape[1], x.shape[0])
        if weight.dim() == 2
        else None
    )
    if (
        cfg is None
        or x.dim() != 2
        or x.shape[0] > MAX_M
        or x.shape[0] == 0
        or x.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or not x.is_contiguous()
        or not weight.is_contiguous()
    ):
        return F.linear(x, weight, bias)
    y = _extension().gemm(x, weight, cfg[0], cfg[1], True)
    return y if bias is None else y + bias


def _fake(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


direct_register_custom_op(
    op_name="decode_unquantized_gemm", op_func=_impl, fake_impl=_fake
)


def decode_unquantized_gemm(
    layer: torch.nn.Module,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.ops.vllm.decode_unquantized_gemm(x, weight, bias)
```

## Source: vllm/model_executor/layers/decode_gemm/decode_gemm.cu
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
// batch is loaded before griddepcontrol.wait, activations after it.
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
          tok[j] ? __ldg(xr + j * wstep + ((S) + u) * 4)           \
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

## Source: vllm/model_executor/layers/mla.py:136-194 (MLA forward)
```python
        if self.q_lora_rank is not None:
            assert self.fused_qkv_a_proj is not None, (
                "fused_qkv_a_proj is required when q_lora_rank is not None"
            )
            assert self.q_a_layernorm is not None, (
                "q_a_layernorm is required when q_lora_rank is not None"
            )
            assert self.q_b_proj is not None, (
                "q_b_proj is required when q_lora_rank is not None"
            )

            qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
            q_c, kv_lora = qkv_lora.split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                dim=-1,
            )
            q_c = self.q_a_layernorm(q_c)
            q = self.q_b_proj(q_c)[0]
        else:
            assert self.kv_a_proj_with_mqa is not None, (
                "kv_a_proj_with_mqa is required when q_lora_rank is None"
            )
            assert self.q_proj is not None, (
                "q_proj is required when q_lora_rank is None"
            )
            kv_lora = self.kv_a_proj_with_mqa(hidden_states)[0]
            q = self.q_proj(hidden_states)[0]

        kv_c, k_pe = kv_lora.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_c_normed = self.kv_a_layernorm(kv_c)

        q = q.view(-1, self.num_heads, self.qk_head_dim)
        # Add head dim of 1 to k_pe
        k_pe = k_pe.unsqueeze(1)

        if self.rotary_emb is not None:
            q[..., self.qk_nope_head_dim :], k_pe = self.rotary_emb(
                positions, q[..., self.qk_nope_head_dim :], k_pe
            )

        if self.cold_prefetch_layer is not None:
            torch.ops.vllm.tiered_moe_cold_join(q)

        if self.indexer and self.is_sparse and not self.skip_topk:
            self.indexer(hidden_states, q_c, positions, self.indexer_rope_emb)

        if llama_4_scaling is not None:
            q *= llama_4_scaling

        attn_out = self.mla_attn(
            q,
            kv_c_normed,
            k_pe,
            output_shape=(hidden_states.shape[0], self.num_heads * self.v_head_dim),
        )

        if self.cold_prefetch_layer is not None:
            torch.ops.vllm.tiered_moe_cold_fork(attn_out, self.cold_prefetch_layer)
        return self.o_proj(attn_out)[0]
```

## Source: mla_attention.py:1084-1106 (_v_up_proj = W_UV)
```python
    def _v_up_proj(self, x: torch.Tensor, out: torch.Tensor):
        # Convert from (B, N, L) to (N, B, L)
        x = x.view(-1, self.num_heads, self.kv_lora_rank).transpose(0, 1)
        out = out.view(-1, self.num_heads, self.v_head_dim)
        if self.is_aiter_triton_fp4_bmm_enabled:
            out = rocm_aiter_ops.batched_gemm_a16wfp4(
                x,
                self.W_V,
                self.W_V_scale,
                out,
                transpose_bm=True,
                prequant=True,
                y_scale=None,
            )
            x = out.view(-1, self.num_heads * self.v_head_dim)
        elif self.is_aiter_triton_fp8_bmm_enabled:
            # Multiply + Transpose (N, B, L) x (N, L, V)->(N, B, V)->(B, N, V)
            x = rocm_aiter_ops.triton_fp8_bmm(
                x, self.W_V, self.W_V_scale, group_size=128, transpose_bm=True, YQ=out
            )
        else:
            # Multiply + Transpose (N, B, L) x (N, L, V)->(N, B, V)->(B, N, V)
            torch.bmm(x, self.W_UV, out=out.transpose(0, 1))
```

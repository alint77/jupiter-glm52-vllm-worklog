# Review request: glue2, a fused decode glue kernel for GLM-5.3 MLA (prototype + integration plan)

**Do not run any commands** (your sandbox cannot start on this machine). Reason from the material below; say where a measurement is needed.

## Context
GLM-5.3 on 4x GH200 (sm_90a), TP4 + DCP4, MTP3 decode: M = 4 tokens per in-flight request, target M = 4..32 (c=1..8). Hard rule: same ops only (identical math; fp32 accumulation-order changes inside reductions/GEMMs have been tolerated; bf16 rounding points must be preserved). Per layer, the served verify CUDA graph runs (nsys, M=4, ~50K ctx, rank 0, us, "gap" = start - previous end):

```
AR+residual+RMSNorm 7.5 -> decode_gemm fused_qkv_a 12.4 (PDL) -> triton_poi_fused_0 (k_pe RoPE, 1.4, gap .4)
-> triton_red_fused_1 (kv_a + q_a RMSNorm combo, 2.8, gap .4) -> decode_gemm q_b 7.6 (PDL, gap .2)
-> triton q RoPE/copy 1.4 -> concat_and_cache_ds_mla (main cache) 1.8 -> concat_and_cache_ds_mla (skip-KV stage) 1.7
-> nvjet W_UK 3.7 -> one_shot gather_cat (DCP) 6.5 -> FlashMLA ...
```
The second concat_and_cache runs only on the 57 "skip" layers (78 layers; 21 anchor layers carry the indexer). decode_gemm (our kernel) loads its first weight batch before `griddepcontrol.wait` and calls `launch_dependents` after its main loop. The Triton kernels don't use PDL.

glue2 replaces the k RoPE, the norm combo and both cache writes with one kernel that calls `griddepcontrol.launch_dependents` at entry (so q_b can launch and preload weights) then `griddepcontrol.wait` before reading qkv_a's output. Since kv_c_normed / k_pe feed only the cache writes in decode, the writes move ahead of q_b.

## Measured (prototype, microbench)
`bench_glue2.py`: one CUDA graph of 64 chains over 8 rotating weight sets (cold L2), decode_gemm with served configs (cuBLAS F.linear at M=32 where decode_gemm falls back), both KV writes in every chain. Two GH200 nodes agree to 0.06 us. us per chain:

| M | served (inductor glue + q_b + 2x concat_and_cache) | glue (v1: norms+RoPE, PDL) + q_b + 2x concat | glue2 | floor (qkv_a -> q_b on stale input) |
|--:|--:|--:|--:|--:|
| 4 | 25.25 | 23.68 | 20.66 | 17.92 |
| 8 | 25.76 | 23.91 | 20.81 | 18.18 |
| 16 | 27.42 | 25.33 | 22.34 | 19.68 |
| 32 | 29.33 | 27.94 | 25.04 | 22.84 |

Correctness vs torch.compile'd reference + `ops.concat_and_cache_mla(..., "fp8_ds_mla", k_scale=1)`: rows 4..32 bitwise for q_c normed and all bytes of both caches (one padded slot -1 per test). 4096 random rows with varied scale: 26/8.4M q_c elements 1 bf16 ulp off, 1+1 cache bytes of 5.4M. The served Triton kernel source (from the inductor cache) is included below; my torch.compile reference is a re-creation, not the served artifact.

Claimed step gain: ~0.30-0.36 ms/step at M=4..32 (78 layers minus one concat on the 21 anchor layers), ~1.8% at M=4.

## Integration plan (not done)
1. In the MLA forward decode path (M <= 32, CUDA-graph decode), call a custom op wrapping glue2 instead of split + q_a_layernorm + kv_a_layernorm + k RoPE; it gets kv_cache and slot_mapping via `get_attention_context(layer_name)` (as vllm's mla_rope_kvcache_cat_fusion pass does) and the skip-KV stager's buffer + reserved_map for skip layers.
2. Suppress the later `do_kv_cache_update` / `write_new` cache writes for those steps; the q RoPE stays where it is (needs q_b output). Prefill path unchanged.
3. Tests (bitwise vs current path over shapes, padding, DCP slot -1, anchor vs skip layers), then served same-node A/B at c=2/c=4.

## Questions
1. Correctness of glue2 vs the two reference kernels: arithmetic order, rounding points (kv normed bf16 then quantized; scale from bf16 normed values; division x/scale and SATFINITE; FLT_MIN floor; 448 divisor), entry layout (bytes 0..511 fp8, 512..527 fp32 scales, 528..655 bf16 RoPE), shared-memory reduction/syncthreads correctness (two block_sum calls + tmax), alignment assumptions (qkv row stride 2624 bf16 = 5248 B: is the uint4 load of q_c / 32-bit loads valid for every row? cache entry stride 656 B and the uint16/float/uint32 stores' alignment), slot -1 handling, int64 index math.
2. PDL semantics: is `launch_dependents` at entry + `griddepcontrol.wait` before reading correct and safe here (q_b's griddepcontrol.wait waits for glue2 grid completion and memory flush? does early trigger risk q_b reading qn before it's written? any issue with glue2 itself being PDL-launched after qkv_a which triggers after its main loop)? Inside CUDA graph capture?
3. Benchmark validity: does the microbench fairly represent the served chain (inductor glue re-created by torch.compile with combo_kernels; KV writes on every layer; cache in device memory -- is the served main cache possibly in host/UVA memory?; L2 rotation; CUDA-event timing over graph replays)? What could make the in-model gain smaller?
4. Integration risks: moving the KV write before q_b / the indexer / attention (anything reading the cache slot in between? DCP ranks writing only owned slots via slot=-1? the skip-KV stager's `stage()` on anchor layers and its side stream/event ordering -- write_new happens on skip layers; is moving it earlier within the same layer safe?), torch.compile/piecewise graph interplay (splitting ops include the kv cache update?), CUDA graph capture of a custom op that reads forward context, MTP drafter layers (they have their own KV), prefill/mixed batches, padded tokens in captured graphs (slot -1).
5. What else should go into the same kernel or what's a better boundary (e.g., q RoPE into q_b's epilogue, indexer-side work, W_UK)? Ranked recommendations.

Output: ranked findings (severity, claim, reasoning, recommendation), then a go/no-go on integration with the minimum test set.

## glue.cu (prototype; glue = v1, glue2 = this review's subject)
```cuda
// Prototype: the glue between fused_qkv_a and q_b in one PDL-aware kernel.
// Replaces inductor's triton_poi_fused_0 (k_pe interleaved RoPE) and
// triton_red_fused_1 (kv_a + q_a RMSNorm) for the GLM-5.3 MLA layout:
// qkv[M, 2624] = [q_c 2048 | kv_c 512 | k_pe 64]. One CTA per token. It lets
// its dependent (q_b's decode_gemm) launch at once, so q_b's first weight
// batch loads while this runs, then waits for qkv_a before reading.
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cfloat>
#include <torch/extension.h>

constexpr int QL = 2048, KVL = 512, PE = 64, LD = QL + KVL + PE;

template <int N>
__device__ __forceinline__ float block_sum(float v, float* sh) {
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
  if (threadIdx.x % 32 == 0) sh[threadIdx.x / 32] = v;
  __syncthreads();
  float s = 0.f;
#pragma unroll
  for (int w = 0; w < N; ++w) s += sh[w];
  return s;
}

__global__ void __launch_bounds__(256)
    qkv_glue_kernel(const __nv_bfloat16* __restrict__ qkv,
                    const int64_t* __restrict__ pos,
                    const __nv_bfloat16* __restrict__ cos_sin,
                    const __nv_bfloat16* __restrict__ wq,
                    const __nv_bfloat16* __restrict__ wkv,
                    __nv_bfloat16* __restrict__ qn,
                    __nv_bfloat16* __restrict__ kvn,
                    __nv_bfloat16* __restrict__ kpe, float eps) {
  __shared__ float shq[8], shk[8];
  asm volatile("griddepcontrol.launch_dependents;");
  asm volatile("griddepcontrol.wait;" ::: "memory");
  const int m = blockIdx.x, t = threadIdx.x;
  const __nv_bfloat16* row = qkv + size_t(m) * LD;
  // q_c: 8 per thread, kv_c: 2 per thread
  const uint4 qv = reinterpret_cast<const uint4*>(row)[t];
  const __nv_bfloat162 kv2 = reinterpret_cast<const __nv_bfloat162*>(row + QL)[t];
  const __nv_bfloat162* q2 = reinterpret_cast<const __nv_bfloat162*>(&qv);
  float q[8];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    q[2 * i] = __low2float(q2[i]);
    q[2 * i + 1] = __high2float(q2[i]);
  }
  const float k0 = __low2float(kv2), k1 = __high2float(kv2);
  float sq = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) sq += q[i] * q[i];
  const float sk = k0 * k0 + k1 * k1;
  const float rq = rsqrtf(block_sum<8>(sq, shq) / float(QL) + eps);
  const float rk = rsqrtf(block_sum<8>(sk, shk) / float(KVL) + eps);
  const uint4 wqv = reinterpret_cast<const uint4*>(wq)[t];
  const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(&wqv);
  uint4 out;
  __nv_bfloat162* o2 = reinterpret_cast<__nv_bfloat162*>(&out);
#pragma unroll
  for (int i = 0; i < 4; ++i)
    o2[i] = __floats2bfloat162_rn(q[2 * i] * rq * __low2float(w2[i]),
                                  q[2 * i + 1] * rq * __high2float(w2[i]));
  reinterpret_cast<uint4*>(qn + size_t(m) * QL)[t] = out;
  const __nv_bfloat162 wk = reinterpret_cast<const __nv_bfloat162*>(wkv)[t];
  reinterpret_cast<__nv_bfloat162*>(kvn + size_t(m) * KVL)[t] =
      __floats2bfloat162_rn(k0 * rk * __low2float(wk), k1 * rk * __high2float(wk));
  if (t < PE / 2) {
    const __nv_bfloat16* cs = cos_sin + pos[m] * PE;
    const float c = __bfloat162float(cs[t]), s = __bfloat162float(cs[PE / 2 + t]);
    const float x0 = __bfloat162float(row[QL + KVL + 2 * t]);
    const float x1 = __bfloat162float(row[QL + KVL + 2 * t + 1]);
    reinterpret_cast<__nv_bfloat162*>(kpe + size_t(m) * PE)[t] =
        __floats2bfloat162_rn(x0 * c - x1 * s, x1 * c + x0 * s);
  }
}

// glue2 = glue + the fp8_ds_mla KV write (concat_and_cache_ds_mla_kernel's
// entry: 512 e4m3 bytes, 4 fp32 tile scales, 64 bf16 RoPE values) into the
// main cache and, when map2 is given, the skip-KV stage. The entry is built
// once from the normalized, bf16-rounded kv_c and the rotated k_pe.
struct Cache {
  uint8_t* base;
  const int64_t* slots;
  int64_t block_stride, entry_stride;
  int block_size;
};

__device__ __forceinline__ uint8_t* entry(const Cache& c, int m) {
  const int64_t s = c.slots[m];
  if (s < 0) return nullptr;
  return c.base + (s / c.block_size) * c.block_stride +
         (s % c.block_size) * c.entry_stride;
}

__global__ void __launch_bounds__(256)
    qkv_glue2_kernel(const __nv_bfloat16* __restrict__ qkv,
                     const int64_t* __restrict__ pos,
                     const __nv_bfloat16* __restrict__ cos_sin,
                     const __nv_bfloat16* __restrict__ wq,
                     const __nv_bfloat16* __restrict__ wkv,
                     __nv_bfloat16* __restrict__ qn, Cache c1, Cache c2,
                     bool has2, float eps) {
  __shared__ float shq[8], shk[8], tmax[8];
  asm volatile("griddepcontrol.launch_dependents;");
  asm volatile("griddepcontrol.wait;" ::: "memory");
  const int m = blockIdx.x, t = threadIdx.x, lane = t % 32, warp = t / 32;
  const __nv_bfloat16* row = qkv + size_t(m) * LD;
  const uint4 qv = reinterpret_cast<const uint4*>(row)[t];
  const __nv_bfloat162 kv2 = reinterpret_cast<const __nv_bfloat162*>(row + QL)[t];
  const __nv_bfloat162* q2 = reinterpret_cast<const __nv_bfloat162*>(&qv);
  float q[8];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    q[2 * i] = __low2float(q2[i]);
    q[2 * i + 1] = __high2float(q2[i]);
  }
  const float k0 = __low2float(kv2), k1 = __high2float(kv2);
  float sq = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) sq += q[i] * q[i];
  const float rq = rsqrtf(block_sum<8>(sq, shq) / float(QL) + eps);
  const float rk = rsqrtf(block_sum<8>(k0 * k0 + k1 * k1, shk) / float(KVL) + eps);
  const uint4 wqv = reinterpret_cast<const uint4*>(wq)[t];
  const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(&wqv);
  uint4 out;
  __nv_bfloat162* o2 = reinterpret_cast<__nv_bfloat162*>(&out);
#pragma unroll
  for (int i = 0; i < 4; ++i)
    o2[i] = __floats2bfloat162_rn(q[2 * i] * rq * __low2float(w2[i]),
                                  q[2 * i + 1] * rq * __high2float(w2[i]));
  reinterpret_cast<uint4*>(qn + size_t(m) * QL)[t] = out;

  // kv_c: elements 2t, 2t + 1; tile = 128 elements = 64 threads = 2 warps
  const __nv_bfloat162 wk = reinterpret_cast<const __nv_bfloat162*>(wkv)[t];
  const __nv_bfloat162 kn =
      __floats2bfloat162_rn(k0 * rk * __low2float(wk), k1 * rk * __high2float(wk));
  float mx = fmaxf(fabsf(__low2float(kn)), fabsf(__high2float(kn)));
  for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
  if (lane == 0) tmax[warp] = mx;
  __syncthreads();
  const int tile = t / 64;
  const float scale = fmaxf(fmaxf(tmax[2 * tile], tmax[2 * tile + 1]) / 448.f, FLT_MIN);
  const uint16_t fp8 =
      uint16_t(__nv_cvt_float_to_fp8(__low2float(kn) / scale, __NV_SATFINITE, __NV_E4M3)) |
      uint16_t(__nv_cvt_float_to_fp8(__high2float(kn) / scale, __NV_SATFINITE, __NV_E4M3)) << 8;
  uint32_t pe = 0;
  if (t < PE / 2) {
    const __nv_bfloat16* cs = cos_sin + pos[m] * PE;
    const float c = __bfloat162float(cs[t]), s = __bfloat162float(cs[PE / 2 + t]);
    const float x0 = __bfloat162float(row[QL + KVL + 2 * t]);
    const float x1 = __bfloat162float(row[QL + KVL + 2 * t + 1]);
    const __nv_bfloat162 r = __floats2bfloat162_rn(x0 * c - x1 * s, x1 * c + x0 * s);
    pe = *reinterpret_cast<const uint32_t*>(&r);
  }
#pragma unroll
  for (int k = 0; k < 2; ++k) {
    if (k == 1 && !has2) break;
    uint8_t* e = entry(k ? c2 : c1, m);
    if (e == nullptr) continue;
    reinterpret_cast<uint16_t*>(e)[t] = fp8;
    if (t % 64 == 0) reinterpret_cast<float*>(e + KVL)[tile] = scale;
    if (t < PE / 2) reinterpret_cast<uint32_t*>(e + KVL + 16)[t] = pe;
  }
}

Cache cache_of(const torch::Tensor& cache, const torch::Tensor& slots) {
  TORCH_CHECK(cache.dim() == 3 && cache.size(2) == KVL + 16 + 2 * PE &&
              cache.scalar_type() == torch::kUInt8 && cache.stride(2) == 1);
  return {cache.data_ptr<uint8_t>(), slots.data_ptr<int64_t>(), cache.stride(0),
          cache.stride(1), static_cast<int>(cache.size(1))};
}

void glue2(torch::Tensor qkv, torch::Tensor pos, torch::Tensor cos_sin,
           torch::Tensor wq, torch::Tensor wkv, torch::Tensor qn,
           torch::Tensor cache1, torch::Tensor slots1,
           std::optional<torch::Tensor> cache2,
           std::optional<torch::Tensor> slots2, double eps, bool pdl) {
  TORCH_CHECK(qkv.size(1) == LD && qkv.stride(0) == LD && cos_sin.size(1) == PE);
  const Cache c1 = cache_of(cache1, slots1);
  const bool has2 = cache2.has_value();
  const Cache c2 = has2 ? cache_of(*cache2, *slots2) : c1;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(qkv.size(0));
  cfg.blockDim = dim3(256);
  cfg.stream = c10::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr;
  attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr.val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = &attr;
  cfg.numAttrs = pdl ? 1 : 0;
  auto B = [](const torch::Tensor& x) {
    return reinterpret_cast<__nv_bfloat16*>(x.data_ptr());
  };
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, qkv_glue2_kernel, B(qkv),
                                    pos.data_ptr<int64_t>(), B(cos_sin), B(wq),
                                    B(wkv), B(qn), c1, c2, has2, float(eps)));
}

void glue(torch::Tensor qkv, torch::Tensor pos, torch::Tensor cos_sin,
          torch::Tensor wq, torch::Tensor wkv, torch::Tensor qn,
          torch::Tensor kvn, torch::Tensor kpe, double eps, bool pdl) {
  TORCH_CHECK(qkv.size(1) == LD && qkv.stride(0) == LD && cos_sin.size(1) == PE);
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(qkv.size(0));
  cfg.blockDim = dim3(256);
  cfg.stream = c10::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr;
  attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr.val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = &attr;
  cfg.numAttrs = pdl ? 1 : 0;
  auto B = [](const torch::Tensor& x) {
    return reinterpret_cast<__nv_bfloat16*>(x.data_ptr());
  };
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, qkv_glue_kernel, B(qkv),
                                    pos.data_ptr<int64_t>(), B(cos_sin), B(wq),
                                    B(wkv), B(qn), B(kvn), B(kpe), float(eps)));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("glue", &glue);
  m.def("glue2", &glue2);
}
```

## bench_glue2.py
```python
"""qkv_a -> glue -> q_b plus the layer's two fp8_ds_mla KV writes: inductor
glue + 2x concat_and_cache_mla (as served) vs glue.cu's glue2 (one kernel).

Correctness: glue2's normalized q_c and both cache entries against the
inductor reference + ops.concat_and_cache_mla (mismatched elements / bytes),
with a padded slot (-1). Timing: as bench_qchain.py, one CUDA graph of REPS
chains over NW rotating weight sets. Variants:
  served   qkv_a -> inductor RoPE/norms -> q_b -> concat_and_cache x2
  glue     qkv_a -> glue (PDL) -> q_b -> concat_and_cache x2
  glue2    qkv_a -> glue2 (norms, RoPE, both KV writes; PDL) -> q_b
  floor    qkv_a -> q_b on a stale input
    bench_glue2.py [--m 4 8 16 32] [--reps 64] [--nw 8]
"""

import argparse
from pathlib import Path

import torch
import torch._inductor.config as ic
from torch.utils.cpp_extension import load

from vllm import _custom_ops as ops
from vllm.model_executor.layers import decode_gemm as dg

ap = argparse.ArgumentParser()
ap.add_argument("--m", type=int, nargs="+", default=[4, 8, 16, 32])
ap.add_argument("--reps", type=int, default=64)
ap.add_argument("--nw", type=int, default=8)
ap.add_argument("--iters", type=int, default=20)
a = ap.parse_args()

here = Path(__file__).resolve().parent
build = Path("/e/fscratch/profound") / Path.home().name / "caches/torch_extensions/qkv_glue"
build.mkdir(parents=True, exist_ok=True)
ext = load(name="qkv_glue", sources=[str(here / "glue.cu")],
           extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
           build_directory=str(build))
_gemm = dg._extension().gemm


def gemm(x, w, *cfg):
    """decode_gemm with the served config, or cuBLAS where it falls back."""
    return _gemm(x, w, *cfg) if cfg[0] is not None else torch.nn.functional.linear(x, w)
ic.combo_kernels = True
dev = torch.device("cuda:0")
torch.manual_seed(0)
EPS = 1e-5
QL, KVL, PE, H = 2048, 512, 64, 6144
NQKV, NQB, ROW = QL + KVL + PE, 4096, 656
bf = dict(device=dev, dtype=torch.bfloat16)


def rms(x, w):
    x = x.float()
    r = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)
    return ((x * r) * w.float()).to(torch.bfloat16)


def ref(qkv, pos, cos_sin, wq, wkv):
    q_c, kv = qkv.split([QL, KVL + PE], -1)
    kv_c, k_pe = kv.split([KVL, PE], -1)
    cos, sin = cos_sin[pos].float().chunk(2, -1)
    x = k_pe.float().unflatten(-1, (PE // 2, 2))
    x0, x1 = x[..., 0], x[..., 1]
    kpe = torch.stack((x0 * cos - x1 * sin, x1 * cos + x0 * sin), -1).flatten(-2)
    return rms(q_c, wq), rms(kv_c, wkv), kpe.to(torch.bfloat16)


ref_c = torch.compile(ref, dynamic=False)
k_scale = torch.ones(1, device=dev, dtype=torch.float32)


def served_kv(kvn, kpe, c1, s1, c2, s2):
    ops.concat_and_cache_mla(kvn, kpe, c1, s1, "fp8_ds_mla", k_scale)
    ops.concat_and_cache_mla(kvn, kpe, c2, s2, "fp8_ds_mla", k_scale)


def glue(qkv, pos, cos_sin, wq, wkv):
    m = qkv.shape[0]
    qn = torch.empty(m, QL, **bf)
    kvn = torch.empty(m, KVL, **bf)
    kpe = torch.empty(m, PE, **bf)
    ext.glue(qkv, pos, cos_sin, wq, wkv, qn, kvn, kpe, EPS, True)
    return qn, kvn, kpe


def glue2(qkv, pos, cos_sin, wq, wkv, c1, s1, c2, s2):
    qn = torch.empty(qkv.shape[0], QL, **bf)
    ext.glue2(qkv, pos, cos_sin, wq, wkv, qn, c1, s1, c2, s2, EPS, True)
    return qn


wqkv = [torch.randn(NQKV, H, **bf) * 0.02 for _ in range(a.nw)]
wqb = [torch.randn(NQB, QL, **bf) * 0.02 for _ in range(a.nw)]
wq = torch.rand(QL, **bf) + 0.5
wkv = torch.rand(KVL, **bf) + 0.5
P = 1 << 20
inv = 1.0 / (10000 ** (torch.arange(0, PE, 2, device=dev).float() / PE))
fr = torch.arange(P, device=dev).float()[:, None] * inv[None]
cos_sin = torch.cat((fr.cos(), fr.sin()), -1).to(torch.bfloat16)


def caches(m):
    c1 = torch.zeros(4096, 64, ROW, device=dev, dtype=torch.uint8)
    nb2 = max(16, (m + 63) // 64)
    c2 = torch.zeros(nb2, 64, ROW, device=dev, dtype=torch.uint8)
    s1 = torch.randperm(4096 * 64, device=dev)[:m]
    s1[m // 2] = -1  # a padded token
    s2 = torch.randperm(nb2 * 64, device=dev)[:m]
    return c1, s1, c2, s2


def check(rows, scale):
    qkv = torch.randn(rows, NQKV, **bf) * torch.rand(rows, 1, **bf) * scale
    pos = torch.randint(0, P, (rows,), device=dev)
    c1, s1, c2, s2 = caches(rows)
    r1, r2 = c1.clone(), c2.clone()
    qr, kvr, kper = ref_c(qkv, pos, cos_sin, wq, wkv)
    served_kv(kvr, kper, r1, s1, r2, s2)
    qn = glue2(qkv, pos, cos_sin, wq, wkv, c1, s1, c2, s2)
    print(f"  rows {rows:>4}: q_c mismatches {int((qn != qr).sum())}/{qn.numel()}, "
          f"cache bytes {int((c1 != r1).sum())} + {int((c2 != r2).sum())} "
          f"of {2 * (rows - 1) * ROW} written")


print("correctness vs inductor + concat_and_cache_mla:")
for rows in (4, 8, 16, 32):
    check(rows, 4)
check(4096, 4)

for m in a.m:
    x = torch.randn(m, H, **bf)
    pos = torch.randint(0, 400_000, (m,), device=dev)
    cq, cb = dg.config(NQKV, H, m), dg.config(NQB, QL, m)
    print(f"M={m}: decode_gemm cfg qkv_a {cq} q_b {cb}")
    c1, s1, c2, s2 = caches(m)
    stale = ref_c(gemm(x, wqkv[0], *(cq or (None, None)), True), pos, cos_sin, wq, wkv)[0]

    def chain(kind, i):
        qkv = gemm(x, wqkv[i % a.nw], *(cq or (None, None)), True)
        if kind == "glue2":
            qn = glue2(qkv, pos, cos_sin, wq, wkv, c1, s1, c2, s2)
            return gemm(qn, wqb[i % a.nw], *(cb or (None, None)), True)
        if kind == "floor":
            return gemm(stale, wqb[i % a.nw], *(cb or (None, None)), True)
        qn, kvn, kpe = (ref_c if kind == "served" else glue)(qkv, pos, cos_sin, wq, wkv)
        y = gemm(qn, wqb[i % a.nw], *(cb or (None, None)), True)
        served_kv(kvn, kpe, c1, s1, c2, s2)
        return y

    res = {}
    for kind in ("served", "glue", "glue2", "floor"):
        for i in range(3):
            chain(kind, i)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for i in range(a.reps):
                chain(kind, i)
        g.replay()
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        ts = []
        for _ in range(5):
            e0.record()
            for _ in range(a.iters):
                g.replay()
            e1.record()
            torch.cuda.synchronize()
            ts.append(e0.elapsed_time(e1) * 1e3 / (a.iters * a.reps))
        res[kind] = sorted(ts)[2]
        print(f"  {kind:<7} {res[kind]:6.2f} us/chain (median of 5: "
              f"{' '.join(f'{t:.2f}' for t in sorted(ts))})")
    print(f"  glue2 saves {res['served'] - res['glue2']:.2f} us/layer vs served "
          f"({res['glue'] - res['glue2']:.2f} vs glue); "
          f"{res['glue2'] - res['floor']:.2f} over the floor")
```

## Served inductor kernels (strings of the compile artifact, rank 0): k_pe RoPE and the norm combo
```python
def triton_poi_fused_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 2)
    x1 = ((xindex // 2) % 32)
    x2 = xindex // 64
    x3 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (2560 + 2*x1 + 2624*x2), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp6 = tl.load(in_ptr1 + (x2), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.full([XBLOCK], 1048576, tl.int32)
    tmp8 = tmp6 + tmp7
    tmp9 = tmp6 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp6)
    tl.device_assert(((0 <= tl.broadcast_to(tmp10, [XBLOCK])) & (tl.broadcast_to(tmp10, [XBLOCK]) < 1048576)) | ~(tmp4 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp10, [XBLOCK]) < 1048576")
    tmp12 = tl.load(in_ptr2 + (x1 + 64*tmp10), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp13 = tmp5 * tmp12
    tmp14 = tl.load(in_ptr0 + (2561 + 2*x1 + 2624*x2), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp15 = tl.load(in_ptr2 + (32 + x1 + 64*tmp10), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp16 = tmp14 * tmp15
    tmp17 = tmp13 - tmp16
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp4, tmp17, tmp18)
    tmp20 = tmp0 >= tmp3
    tmp21 = tl.full([1], 2, tl.int64)
    tmp22 = tmp0 < tmp21
    tmp23 = tl.load(in_ptr0 + (2561 + 2*x1 + 2624*x2), tmp20 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp24 = tl.load(in_ptr1 + (x2), tmp20 & xmask, eviction_policy='evict_last', other=0.0)
    tmp25 = tl.full([XBLOCK], 1048576, tl.int32)
    tmp26 = tmp24 + tmp25
    tmp27 = tmp24 < 0
    tmp28 = tl.where(tmp27, tmp26, tmp24)
    tl.device_assert(((0 <= tl.broadcast_to(tmp28, [XBLOCK])) & (tl.broadcast_to(tmp28, [XBLOCK]) < 1048576)) | ~(tmp20 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp28, [XBLOCK]) < 1048576")
    tmp30 = tl.load(in_ptr2 + (x1 + 64*tmp28), tmp20 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp31 = tmp23 * tmp30
    tmp32 = tl.load(in_ptr0 + (2560 + 2*x1 + 2624*x2), tmp20 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp33 = tl.load(in_ptr2 + (32 + x1 + 64*tmp28), tmp20 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp34 = tmp32 * tmp33
    tmp35 = tmp31 + tmp34
    tmp36 = tl.full(tmp35.shape, 0.0, tmp35.dtype)
    tmp37 = tl.where(tmp20, tmp35, tmp36)
    tmp38 = tl.where(tmp4, tmp19, tmp37)
    tl.store(out_ptr0 + (x3), tmp38, xmask)
''', device_str='cuda')
# kernel path: /e/fscratch/profound/naeimitabiei1/caches/marlin/vllm-cache-glm53-nsol-c1-2272145/torch_compile_cache/torch_aot_compile/402a0a7dd913d04ae88029e11d7537be5a7f60aa117f65a78ade79627a81363f/inductor_cache/py/cpyxjhwsrjtlor7mcczzbemxahzhjxa4pdt7caamrmlhdezcauiy.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_red_fused_1 = async_compile.triton('triton_red_fused_1', '''
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch
@triton_heuristics.reduction(
    size_hints={'x': 4096, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr1': '*bf16', 'out_ptr3': '*bf16', 'xnumel_0': 'i32', 'xnumel_1': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'enable_fp_fusion': True, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
def triton_red_fused_1(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr3, xnumel_0, xnumel_1, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    pid = tl.program_id(0)
    num_xblocks_0 = tl.cdiv(xnumel_0, XBLOCK)
    num_xblocks_1 = num_xblocks_0 + tl.cdiv(xnumel_1, XBLOCK)
    if pid < num_xblocks_0:
        pid_offset = pid
        r0_numel = 512
        R0_BLOCK_0: tl.constexpr = 512
        rnumel = r0_numel
        RBLOCK: tl.constexpr = R0_BLOCK_0
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
        xmask = xindex < xnumel_0
        r0_index = tl.arange(0, R0_BLOCK_0)[None, :]
        r0_offset = 0
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        x0 = xindex
        tmp0 = tl.load(in_ptr0 + (2048 + r0_1 + 2624*x0), r0_mask & xmask, other=0.0).to(tl.float32)
        tmp14 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tmp1 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK_0])
        tmp5 = tl.where(r0_mask & xmask, tmp3, 0)
        tmp6 = tl.sum(tmp5, 1)[:, None].to(tl.float32)
        tmp7 = tl.full([1, 1], 512.0, tl.float32)
        tmp8 = (tmp6 / tmp7)
        tmp9 = tl.full([1, 1], 1e-05, tl.float32)
        tmp10 = tmp8 + tmp9
        tmp11 = libdevice.rsqrt(tmp10)
        tmp12 = tmp1 * tmp11
        tmp13 = tmp12.to(tl.float32)
        tmp15 = tmp13 * tmp14
        tl.store(out_ptr1 + (r0_1 + 512*x0), tmp15, r0_mask & xmask)
    elif pid < num_xblocks_1:
        pid_offset = pid - num_xblocks_0
        r0_numel = 2048
        rnumel = r0_numel
        RBLOCK: tl.constexpr = R0_BLOCK
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
        xmask = xindex < xnumel_1
        r0_base = tl.arange(0, R0_BLOCK)[None, :]
        rbase = r0_base
        x2 = xindex
        _tmp20 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_3 = r0_index
            tmp16 = tl.load(in_ptr0 + (r0_3 + 2624*x2), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
            tmp17 = tmp16.to(tl.float32)
            tmp18 = tmp17 * tmp17
            tmp19 = tl.broadcast_to(tmp18, [XBLOCK, R0_BLOCK])
            tmp21 = _tmp20 + tmp19
            _tmp20 = tl.where(r0_mask & xmask, tmp21, _tmp20)
        tmp20 = tl.sum(_tmp20, 1)[:, None]
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_3 = r0_index
            tmp22 = tl.load(in_ptr0 + (r0_3 + 2624*x2), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
            tmp31 = tl.load(in_ptr2 + (r0_3), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
            tmp23 = tmp22.to(tl.float32)
            tmp24 = tl.full([1, 1], 2048.0, tl.float32)
            tmp25 = (tmp20 / tmp24)
            tmp26 = tl.full([1, 1], 1e-05, tl.float32)
            tmp27 = tmp25 + tmp26
            tmp28 = libdevice.rsqrt(tmp27)
            tmp29 = tmp23 * tmp28
            tmp30 = tmp29.to(tl.float32)
            tmp32 = tmp30 * tmp31
            tl.store(out_ptr3 + (r0_3 + 2048*x2), tmp32, r0_mask & xmask)
    else:
        pass
def get_args():
    arg_0 = rand_strided((4096, 2624), (2624, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.bfloat16)
    arg_2 = rand_strided((2048,), (1,), device='cuda:0', dtype=torch.bfloat16)
    arg_3 = rand_strided((4096, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_4 = rand_strided((4096, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    return arg_0, arg_1, arg_2, arg_3, arg_4, 4096, 4096,
```

## csrc/libtorch_stable/cache_kernels.cu:446-552 (concat_and_cache_ds_mla_kernel; kFp8ScaleDivisor = 448 on CUDA)
```cuda
template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt>
__global__ void concat_and_cache_ds_mla_kernel(
    const scalar_t* __restrict__ kv_c,  // [num_tokens, kv_lora_rank]
    const scalar_t* __restrict__ k_pe,  // [num_tokens, pe_dim]
    cache_t* __restrict__ kv_cache,  // [num_blocks, block_size, (kv_lora_rank
                                     // + pe_dim)]
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int block_stride,                    //
    const int entry_stride,                    //
    const int kv_c_stride,                     //
    const int k_pe_stride,                     //
    const int kv_lora_rank,                    //
    const int pe_dim,                          //
    const int block_size,                      //
    const float* scale                         //
) {
  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];
  // NOTE: slot_idx can be -1 if the token is padded
  if (slot_idx < 0) {
    return;
  }
  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;
  const int64_t dst_idx_start =
      block_idx * block_stride + block_offset * entry_stride;

  // For the NoPE part, each tile of 128 elements is handled by half of one warp
  // (16 threads). There are 4 total tiles, so 2 warps (64 threads).
  // Lanes 0 and 16 of each warp write the scale values for that warp's tiles.
  // The RoPE part (last 64 elements) is handled by another 1 warp (32 threads).
  // So in total, we use 3 warps (96 threads) per block.

  // Cast kv_cache to 16_bit for RoPE values
  scalar_t* kv_cache_16bit =
      reinterpret_cast<scalar_t*>(&kv_cache[dst_idx_start]);

  // The last warp handles the RoPE part
  if (threadIdx.x >= 64) {
    // Each thread handles two elements of RoPE
    const int8_t pe_idx_start = (threadIdx.x - 64) * 2;
    const int64_t src_idx = token_idx * k_pe_stride + pe_idx_start;
    // Vectorized load of two 16-bit values, performed as one 32-bit load
    const int32_t vals = *reinterpret_cast<const int32_t*>(&k_pe[src_idx]);
    // RoPE values start after the packed 8-bit NoPE values and the
    // 32-bit scales
    const int64_t dst_idx = kv_lora_rank / 2 + 8 + pe_idx_start;
    // Vectorized store of two 16-bit values, performed as one 32-bit store
    *reinterpret_cast<int32_t*>(&kv_cache_16bit[dst_idx]) = vals;
    return;
  }

  // The first two warps handle the NoPE part
  const int8_t warp_idx = threadIdx.x >> 5;
  const int8_t lane_idx = threadIdx.x & 31;
  const int8_t tile_idx = warp_idx * 2 + (lane_idx >> 4);

  // Each thread handles 8 elements of NoPE
  // Load the NoPE elements for this thread into registers
  const int64_t src_idx_start = token_idx * kv_c_stride + (threadIdx.x * 8);
  // Vectorized load of eight 16-bit values, performed as an int4 load
  const int4 vals_i4 = *reinterpret_cast<const int4*>(&kv_c[src_idx_start]);
  const scalar_t* vals = reinterpret_cast<const scalar_t*>(&vals_i4);

  // Max absolute value of this thread's elements
  float max_abs = fmaxf(fmaxf(fmaxf(fabsf(vals[0]), fabsf(vals[1])),
                              fmaxf(fabsf(vals[2]), fabsf(vals[3]))),
                        fmaxf(fmaxf(fabsf(vals[4]), fabsf(vals[5])),
                              fmaxf(fabsf(vals[6]), fabsf(vals[7]))));

  // Warp-level reduction to find the max absolute value in each half-warp
#pragma unroll
  for (int offset = 8; offset > 0; offset /= 2) {
    max_abs = fmaxf(max_abs, VLLM_SHFL_XOR_SYNC_WIDTH(max_abs, offset, 16));
  }

  // Compute the scale for the tile
  float tile_scale = fmaxf(max_abs / kFp8ScaleDivisor, FLT_MIN);

  // The first lane of each half-warp writes the scale to kv_cache
  if ((lane_idx == 0) || (lane_idx == 16)) {
    float* kv_cache_32bit = reinterpret_cast<float*>(&kv_cache[dst_idx_start]);
    const uint64_t dst_idx = kv_lora_rank / 4 + tile_idx;
    kv_cache_32bit[dst_idx] = tile_scale;
  }

  // Now all threads in the block scale and write their elements
  // NoPE data is packed in the first kv_lora_rank/2 bytes (first 256 bytes)
  const int64_t dst_idx_base = dst_idx_start + (threadIdx.x * 8);

  uint8_t result[8];
#pragma unroll
  for (int i = 0; i < 8; i++) {
    result[i] =
        fp8::scaled_convert<uint8_t, scalar_t, Fp8KVCacheDataType::kFp8E4M3>(
            vals[i], tile_scale);
  }

  // Store as aligned 64-bit writes
  *reinterpret_cast<uint64_t*>(&kv_cache[dst_idx_base]) =
      *reinterpret_cast<const uint64_t*>(result);
}

template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt>
__global__ void indexer_k_quant_and_cache_kernel(
    const scalar_t* __restrict__ k,  // [num_tokens, head_dim]
    cache_t* __restrict__ kv_cache,  // [num_blocks, block_size, cache_stride]
```

## csrc/quantization/w8a8/fp8/nvidia/quant_utils.cuh:482-493
```cuda
__inline__ __device__ uint8_t scaled_vec_conversion<uint8_t, __nv_bfloat16>(
    const __nv_bfloat16& a, const float scale,
    const __nv_fp8_interpretation_t fp8_type) {
    #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  assert(false);
    #else
  __nv_fp8_storage_t res = __nv_cvt_float_to_fp8(__bfloat162float(a) / scale,
                                                 __NV_SATFINITE, fp8_type);
  return (uint8_t)res;
    #endif
  __builtin_unreachable();  // Suppress missing return statement warning
}
```

## vllm/model_executor/layers/mla.py:136-194 (MLA forward)
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

## vllm/v1/attention/backend.py:1025-1046 (do_kv_cache_update)
```python
    def do_kv_cache_update(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
    ) -> None:
        if kv_cache.numel() == 0:
            return
        from vllm import _custom_ops as ops

        ops.concat_and_cache_mla(
            kv_c_normed,
            k_pe.squeeze(1),
            kv_cache,
            slot_mapping.flatten(),
            kv_cache_dtype=kv_cache_dtype,
            scale=k_scale,
        )

```

## vllm/v1/attention/backends/mla/skip_kv_stage.py:176-205, 295-325 (stager buffers, write_new)
```python

    def __init__(self, device: torch.device, layers: dict[int, Any]):
        self.max_queries, self.columns = limits()
        self.gather_rows = gather_rows()
        rows = self.gather_rows + RESERVED_ROWS
        self.buffers = torch.zeros(
            STAGE_BUFFERS, rows // 64, 64, ROW_BYTES, dtype=torch.uint8, device=device
        )
        self.slots = torch.zeros(self.gather_rows, dtype=torch.int32, device=device)
        self.count = torch.zeros(1, dtype=torch.int32, device=device)
        self.rows = torch.full(
            (self.max_queries, self.columns), -1, dtype=torch.int32, device=device
        )
        self.stream = torch.cuda.Stream(device)
        # anchor cache -> (its skip layers' caches, ready event, group number);
        # skip cache -> buffer position
        self.groups: dict[int, tuple[list[torch.Tensor], torch.cuda.Event, int]]
        self.groups = {}
        self.position: dict[int, int] = {}
        self._ready: torch.cuda.Event | None = None
        self._width = self.columns  # the anchor's index width, set by stage()
        self._reserved = self.gather_rows + torch.arange(
            self.max_queries, dtype=torch.int64, device=device
        )
        self.reserved_map = torch.full_like(self._reserved, -1)
        num_slots = 0
        for index, layer in layers.items():
            if layer.impl._reuses_topk:
                continue
            group: list[torch.Tensor] = []
...
            return None
        assert self._ready is not None
        torch.cuda.current_stream().wait_event(self._ready)
        rows = self.rows.view(-1)[: num_queries * self._width]
        return self.buffers[p], rows.view(num_queries, self._width)

    def write_new(
        self,
        cache: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        slot_mapping: torch.Tensor,
        k_scale: torch.Tensor,
    ) -> None:
        """Skip layer's KV write: this step's tokens into its reserved rows
        (the mapping is built by the anchor's ``stage``)."""
        p = self.position.get(cache.data_ptr())
        if p is None:
            return
        from vllm import _custom_ops as ops

        ops.concat_and_cache_mla(
            kv_c_normed,
            k_pe.squeeze(1),
            self.buffers[p],
            self.reserved_map[: slot_mapping.numel()],
            kv_cache_dtype="fp8_ds_mla",
            scale=k_scale,
        )


```

## vllm/model_executor/layers/decode_gemm/decode_gemm.cu (q_b / qkv_a kernel, PDL)
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

## vllm/v1/attention/backends/mla/flashmla_sparse.py:985-1015 (the sparse backend override: main write + skip-KV write_new)
```python
            valid_counts,
            self.softmax_scale,
            mask_empty_lse=not self.dcp_combine_drops_inf_lse,
            out=out,
        )

    def do_kv_cache_update(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
    ) -> None:
        super().do_kv_cache_update(
            kv_c_normed, k_pe, kv_cache, slot_mapping, kv_cache_dtype, k_scale
        )
        if (
            self._skip_kv
            and self._reuses_topk
            and slot_mapping.numel() <= skip_kv_stage.limits()[0]
        ):
            skip_kv_stage.get_stager(kv_cache.device).write_new(
                kv_cache, kv_c_normed, k_pe, slot_mapping, k_scale
            )

    def _dcp_decode_sched(self, q: torch.Tensor) -> bool:
        """Whether this decode call can run on dcp_decode_sched_meta's schedule
        (FlashMLA sm90 fp8 sparse decode, no head padding)."""
        return (
```

## vllm/model_executor/layers/attention/mla_attention.py:600-625 and 1109-1140 (call sites)
```python
            attn_metadata_raw = forward_context.attn_metadata
            attn_metadata: MLACommonMetadata
            if isinstance(attn_metadata_raw, dict):
                attn_metadata = attn_metadata_raw[self.layer_name]  # type: ignore[assignment]
            elif isinstance(attn_metadata_raw, list):
                # list[dict[str, AttentionMetadata]]: used in speculative decoding
                # where [0] is the base-model (non-speculative) metadata dict.
                attn_metadata = attn_metadata_raw[0][self.layer_name]  # type: ignore[assignment]
            else:
                attn_metadata = attn_metadata_raw
            self_kv_cache = self.kv_cache
            slot_mapping = forward_context.slot_mapping

            assert isinstance(slot_mapping, dict), (
                f"Expected slot_mapping to be a dict, got {type(slot_mapping)}. "
            )
            self.impl.do_kv_cache_update(  # type: ignore[attr-defined]
                kv_c_normed,
                k_pe,
                self_kv_cache,
                slot_mapping.get(self.layer_name),
                self.kv_cache_dtype,
                self._k_scale,
            )
            output = torch.empty(output_shape, dtype=q.dtype, device=q.device)
            self.forward_impl(
...
def unified_mla_kv_cache_update(
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    layer_name: LayerNameType,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
) -> torch.Tensor:
    """
    Returns a dummy that is passed to unified_attention to signal a side effect and
    the data dependency between them to ensure torch.compile preserves ordering.
    """
    layer_name = _resolve_layer_name(layer_name)
    _, attn_layer, kv_cache, layer_slot_mapping = get_attention_context(layer_name)
    if layer_slot_mapping is not None:
        attn_layer.impl.do_kv_cache_update(  # type: ignore[attr-defined]
            kv_c_normed,
            k_pe,
            kv_cache,
            layer_slot_mapping,
            kv_cache_dtype,
            k_scale,
        )

    return torch.empty(0, device=kv_c_normed.device, dtype=kv_c_normed.dtype)


def unified_mla_kv_cache_update_fake(
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    layer_name: LayerNameType,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
```

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Decode-step MoE for TP-sliced tiered INT4 experts on sm_90a (GLM-5.3 W4A16:
// symmetric INT4, bf16 group-32 scales, 6144 hidden, top-8).
//
// Every GPU holds a 512-wide slice of every routed expert's intermediate
// dimension (gate / up rows [512 r, 512 r + 512), the matching down columns):
// hot slices in HBM, cold slices in its own Grace memory, both in Marlin's
// layout. The all-reduce after the MoE sums the four partial outputs.
//
// One persistent kernel per layer, one CTA per SM, warp specialized: a
// producer warp claims groups of 128-row x 512-K units from a dynamic
// per-tier queue (shared-expert w13, routed w13, shared-expert w2, routed w2)
// and streams them with TMA into a 4-stage smem ring; eight consumer warps
// decode INT4 to exact f16 (one shift, four lop3, four hfma2 per word) and run
// mma.sync m16n8k16 with fp32 accumulation. The last T + 1 CTAs first build
// the step's expert lists and activation rows; the shared expert's w13 needs
// neither and starts at once. Routed w13 partials go to y13 through fp32 reds;
// the CTA that completes an entry's w13 applies silu * up and publishes the
// entry, whose w2 units then run. The warp that completes a 128-row output
// tile writes it as bf16. The output is routed_scale * routed + shared, the
// MoE's final partial sum. Numerics match Marlin's up to fp32 summation order.
//
// C ABI (no torch headers): td_workspace_bytes(), td_forward(...).

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>

#include <cstddef>
#include <cstring>
#include <cstdint>

namespace tiered_decode {

// The 4-way intermediate slice of every expert: INTER = 512 per GPU. w13 and
// w2 units are both 128 rows x 512 of K (w13: one K chunk, the 12 chunks of a
// tile accumulate in registers; w2: all of K): 32 KB of weights and 4 KB of
// bf16 scales each.
#ifndef TD_STAGES
  #define TD_STAGES 4  // 5 fits but measured slower
#endif
#ifndef TD_COLD_CTAS
  #define TD_COLD_CTAS 16
#endif
constexpr int HIDDEN = 6144, INTER = 512, TOPK = 8;
constexpr int MAX_TOK = 8, MAX_TOKENS = 32, MAX_ROUTES = MAX_TOKENS * TOPK,
              MAX_LIST = MAX_ROUTES;
constexpr int R0 = 128, R1 = 128;                // rows per w13 / w2 unit
constexpr int CK0 = 512;                         // w13 K chunk
constexpr int KT0 = CK0 / 16, KT1 = INTER / 16;  // k16 rows per unit
constexpr int G0 = CK0 / 32, G1 = INTER / 32;    // scale groups per unit
constexpr int TILES0 = 2 * INTER / R0, CHUNKS0 = HIDDEN / CK0;
constexpr int UNITS0 = TILES0 * CHUNKS0;  // w13 units per entry
constexpr int TILES1 = HIDDEN / R1;       // w2 units per entry
constexpr int W_BYTES = R0 * CK0 / 2, S_BYTES = G0 * R0 * 2;
static_assert(W_BYTES == R1 * INTER / 2 && S_BYTES == G1 * R1 * 2,
              "w13 and w2 units share the stage layout");
enum Fmt : int { MXFP4 = 0, INT4 = 1 };
constexpr int XROW_BYTES0 = CK0 * 2, XROW_BYTES1 = INTER * 2;
constexpr int XROW_STRIDE =
    XROW_BYTES0 + 64;  // token rows g, g+1 on disjoint banks
static_assert(XROW_BYTES0 == XROW_BYTES1, "w13 and w2 rows are the same size");
// an x2 row in the workspace: INTER halves, then its fp32 scale (xs2), so one
// bulk copy brings both and the producer has no dependent xs2 load
constexpr int X2_LD = INTER + 8, X2_COPY = XROW_BYTES1 + 16;
static_assert(X2_COPY <= XROW_STRIDE, "x2 row + scale fit a stage row");
// shared expert (bf16, this GPU's 512-wide TP slice of it): units of 256 rows
// x 64 of K, 128 B swizzled, for all T tokens
constexpr int RS = 256, CKS = 64;
constexpr int TILES_S0 = 2 * INTER / RS, CHUNKS_S0 = HIDDEN / CKS;
constexpr int TILES_S1 = HIDDEN / RS, CHUNKS_S1 = INTER / CKS;
constexpr int UNITS_S0 = TILES_S0 * CHUNKS_S0, UNITS_S1 = TILES_S1 * CHUNKS_S1;
static_assert(RS * CKS * 2 == W_BYTES, "a shared unit fills the weight slot");
constexpr int XS_BYTES =
    CKS * 2;  // one token's activation slice per shared unit
// shared units' token rows sit 144 B apart: rows g = 0..7 on disjoint banks
constexpr int XS_STRIDE = XS_BYTES + 16;
// stages start on 1 KB boundaries (128 B swizzle)
constexpr int STAGE_BYTES =
    (W_BYTES + S_BYTES + MAX_TOK * XROW_STRIDE + 1023) / 1024 * 1024;
static_assert(MAX_TOKENS * XS_STRIDE <= STAGE_BYTES - W_BYTES,
              "shared rows fit");
constexpr int STAGES = TD_STAGES;
constexpr int CONSUMER_WARPS = 8;
constexpr int THREADS = (CONSUMER_WARPS + 1) * 32;
constexpr int SMEM_HEAD = 128;
static_assert(STAGES <= 8, "barrier head holds 8 stages");
constexpr int SMEM_BYTES = SMEM_HEAD + 1024 + STAGES * STAGE_BYTES;
static_assert(SMEM_BYTES <= 227 * 1024, "ring exceeds shared memory");
constexpr int PLACEMENT_EXP = 14;  // decoded weights are value * 2^-14
constexpr int GRID = 132;
static_assert(
    CONSUMER_WARPS == 8,
    "consumer warps are 4 row blocks x 2 K halves (w13), 8 row blocks (w2)");


struct Expert {
  int local;  // index into the tier's tensors
  int ntok;
  int tok[MAX_TOK];    // token rows
  int route[MAX_TOK];  // token * TOPK + k
  float wt[MAX_TOK];   // router weight * routed_scale
};

// Per tier and projection: the weight and scale tensor maps. A weight map is
// [E][K/16][N*2] int32 with a {128, 64, 1} box: one 64-row tile's 64 k16 rows,
// 512 B each, landing contiguous in shared memory. A scale map is [E][K/32][N]
// bytes with a {64, 32, 1} box.
struct alignas(64) Tier {
  CUtensorMap w[2];
  CUtensorMap s[2];
};

// One call's counters. Calls alternate between two sets, and each call zeroes
// the other set for the next, so nothing has to run between calls.
struct Counters {
  int next[2];                // per tier: next group to claim (hot, cold)
  int done13[2][MAX_LIST];    // w13 chunks flushed per entry
  int done_s;                 // shared w13 chunks flushed
  int prep;                   // prep CTAs finished
  int tiles[HIDDEN / 128];    // flushes into each 128-row output tile
};

// Device workspace, zeroed once at allocation; y13, y13s and y are re-zeroed
// by their last reader so every call finds them clean.
struct Workspace {
  Expert lists[2][MAX_LIST];  // hot, cold
  int counts[2];
  float xs13[MAX_TOKENS];
  float xs2[MAX_ROUTES];
  alignas(128)
      __half x13[MAX_TOKENS * HIDDEN];  // TMA sources: 16 B aligned at least
  alignas(128) __half x2[MAX_ROUTES * X2_LD];
  alignas(128) float y13[MAX_ROUTES * 2 * INTER];
  alignas(128) float y[MAX_TOKENS * HIDDEN];
  int ready[2][MAX_LIST];  // 1 once the entry's activation rows are written
                           // (zeroed by the list CTA before the entry exists)
  alignas(128) __nv_bfloat16 x2s[MAX_TOKENS * INTER];  // shared activation
  alignas(128) float y13s[MAX_TOKENS * 2 * INTER];
  int ready_s;                // the call's epoch once x2s is written
  unsigned long long arrive;  // CTAs started, over all calls
  Counters c[2];
};
static_assert(offsetof(Workspace, x13) % 16 == 0 &&
                  offsetof(Workspace, x2) % 16 == 0 &&
                  offsetof(Workspace, x2s) % 16 == 0,
              "TMA alignment");

struct Params {
  Tier tier[2];
  CUtensorMap
      sw[2];  // shared expert w13 [2 * INTER][HIDDEN], w2 [HIDDEN][INTER]
  Workspace* ws;
  const __nv_bfloat16* x;  // [T][HIDDEN]
  __nv_bfloat16* out;      // [T][HIDDEN]
  const int* ids;          // [T][TOPK] global expert ids, -1: none
  const float* wt;         // [T][TOPK]
  const bool* padding;     // [T] or null: a padded token's routes are dropped
  const int* hot_map;      // [E] global id -> hot slot, else -1
  const int* cold_map;     // [E] global id -> cold slot, else -1
  int T, hot_size, cold_size;
  float shared_scale, routed_scale;
  int has_shared;
};

// ---------------------------------------------------------------- PTX helpers
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)),
               "r"(count));
}
#ifdef TD_COMPUTE_ONLY
// timing probe: no loads; each stage is released with a plain arrive
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(bar))
               : "memory");
}
#else
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(
                   smem_u32(bar)),
               "r"(bytes)
               : "memory");
}
#endif
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(bar))
               : "memory");
}
__device__ __forceinline__ uint32_t lds_u32(uint32_t a) {
  uint32_t v;
  asm volatile("ld.shared.b32 %0, [%1];" : "=r"(v) : "r"(a));
  return v;
}
__device__ __forceinline__ uint4 lds_v4(uint32_t a) {
  uint4 v;
  asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "r"(a));
  return v;
}
__device__ __forceinline__ void mbar_wait_a(uint32_t bar, uint32_t parity) {
  asm volatile(
      "{\n .reg .pred p;\n WAITA_%=:\n "
      "mbarrier.try_wait.parity.shared::cta.b64 "
      "p, [%0], %1;\n"
      " @!p bra WAITA_%=;\n}\n" ::"r"(bar),
      "r"(parity)
      : "memory");
}
__device__ __forceinline__ void mbar_arrive_a(uint32_t bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(bar)
               : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
  asm volatile(
      "{\n .reg .pred p;\n WAIT_%=:\n mbarrier.try_wait.parity.shared::cta.b64 "
      "p, [%0], %1;\n"
      " @!p bra WAIT_%=;\n}\n" ::"r"(smem_u32(bar)),
      "r"(parity)
      : "memory");
}
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src,
                                         uint32_t bytes, uint64_t* bar) {
#ifndef TD_COMPUTE_ONLY
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], "
      "[%1], %2, [%3];" ::"r"(smem_u32(dst)),
      "l"(src), "r"(bytes), "r"(smem_u32(bar))
      : "memory");
#endif
}
__device__ __forceinline__ void tma_3d(void* dst, const CUtensorMap* map,
                                       int c0, int c1, int c2, uint64_t* bar) {
#ifndef TD_COMPUTE_ONLY
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::"
      "bytes [%0], [%1, {%2, %3, %4}], [%5];" ::"r"(smem_u32(dst)),
      "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(c2),
      "r"(smem_u32(bar))
      : "memory");
#endif
}
__device__ __forceinline__ void tma_2d(void* dst, const CUtensorMap* map,
                                       int c0, int c1, uint64_t* bar) {
#ifndef TD_COMPUTE_ONLY
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::"
      "bytes [%0], [%1, {%2, %3}], [%4];" ::"r"(smem_u32(dst)),
      "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(smem_u32(bar))
      : "memory");
#endif
}
__device__ __forceinline__ void ldmatrix_x4(uint32_t* a, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
               : "r"(smem_u32(p)));
}
__device__ __forceinline__ void mma_bf16(float* d, const uint32_t* a,
                                         uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, "
      "{%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_f16(float* d, const uint32_t* a,
                                        uint32_t b0, uint32_t b1,
                                        const float* c) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "f"(c[0]),
        "f"(c[1]), "f"(c[2]), "f"(c[3]));
}
__device__ __forceinline__ uint32_t prmt0(uint32_t a, uint32_t sel) {
  uint32_t out;
  asm("prmt.b32 %0, %1, 0, %2;" : "=r"(out) : "r"(a), "r"(sel));
  return out;
}
// One Marlin word -> the four f16x2 A registers. Nibbles, low to high:
// (g,k0) (g,k8) (g+8,k0) (g+8,k8) (g,k1) (g,k9) (g+8,k1) (g+8,k9). Low and high
// nibbles become e5m2 bytes (sign at 7, exponent at 3..2, mantissa at 1), and
// an e5m2 byte is the high byte of the f16 it equals.
__device__ __forceinline__ void decode(uint32_t w, uint32_t* a) {
  const uint32_t lo = ((w << 4) & 0x80808080u) | ((w << 1) & 0x0E0E0E0Eu);
  const uint32_t hi = (w & 0x80808080u) | ((w >> 3) & 0x0E0E0E0Eu);
  a[0] = prmt0(lo, 0x2404);  // row g,   k0 k1
  a[1] = prmt0(lo, 0x3414);  // row g+8, k0 k1
  a[2] = prmt0(hi, 0x2404);  // row g,   k8 k9
  a[3] = prmt0(hi, 0x3414);  // row g+8, k8 k9
}
// The same word as symmetric INT4 (code - 8), in the same fragment order: each
// nibble pair lands under 0x6400 as 1024 + code, and one exact fma gives
// (code - 8) * 2^-14, the scale the MXFP4 decode leaves its values at too.
__device__ __forceinline__ void decode_int4(uint32_t w, uint32_t* a) {
  const __half2 unit = __halves2half2(__ushort_as_half(0x0400),
                                      __ushort_as_half(0x0400));  // 2^-14
  const __half2 bias =
      __halves2half2(__ushort_as_half(0xAC08),
                     __ushort_as_half(0xAC08));  // -1032 * 2^-14
  const int shift[4] = {0, 8, 4, 12};  // rows g, g+8 at k0 k1; then at k8 k9
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const uint32_t t = ((w >> shift[i]) & 0x000F000Fu) | 0x64006400u;
    const __half2 v =
        __hfma2(*reinterpret_cast<const __half2*>(&t), unit, bias);
    a[i] = *reinterpret_cast<const uint32_t*>(&v);
  }
}
// The same fragment order with one shift and four lop3 per word: nibbles
// (0,4) / (2,6) under 0x6400 give 1024 + code, nibbles (1,5) / (3,7) give
// 1024 + 16 * code; one exact hfma2 each brings both to (code - 8) * 2^-14.
__device__ __forceinline__ uint32_t lop3_and_or(uint32_t a, uint32_t mask,
                                                uint32_t magic) {
  uint32_t d;
  asm("lop3.b32 %0, %1, %2, %3, 0xEA;"
      : "=r"(d)
      : "r"(a), "r"(mask), "r"(magic));
  return d;  // (a & mask) | magic
}
__device__ __forceinline__ void decode_int4_fast(uint32_t w, uint32_t* a) {
  const uint32_t magic = 0x64006400u;
  const uint32_t w8 = w >> 8;
  const uint32_t lo0 = lop3_and_or(w, 0x000F000Fu, magic);   // rows g,   k0 k1
  const uint32_t lo1 = lop3_and_or(w8, 0x000F000Fu, magic);  // rows g+8, k0 k1
  const uint32_t hi0 = lop3_and_or(w, 0x00F000F0u, magic);   // rows g,   k8 k9
  const uint32_t hi1 = lop3_and_or(w8, 0x00F000F0u, magic);  // rows g+8, k8 k9
  const __half2 unit =
      __halves2half2(__ushort_as_half(0x0400), __ushort_as_half(0x0400));
  const __half2 bias =
      __halves2half2(__ushort_as_half(0xAC08), __ushort_as_half(0xAC08));
  // (1024 + 16 c) * 2^-18 - 72 * 2^-14 = (c - 8) * 2^-14
  const __half2 unit16 =
      __halves2half2(__ushort_as_half(0x0040), __ushort_as_half(0x0040));
  const __half2 bias16 =
      __halves2half2(__ushort_as_half(0x9C80), __ushort_as_half(0x9C80));
  __half2 v;
  v = __hfma2(*reinterpret_cast<const __half2*>(&lo0), unit, bias);
  a[0] = *reinterpret_cast<const uint32_t*>(&v);
  v = __hfma2(*reinterpret_cast<const __half2*>(&lo1), unit, bias);
  a[1] = *reinterpret_cast<const uint32_t*>(&v);
  v = __hfma2(*reinterpret_cast<const __half2*>(&hi0), unit16, bias16);
  a[2] = *reinterpret_cast<const uint32_t*>(&v);
  v = __hfma2(*reinterpret_cast<const __half2*>(&hi1), unit16, bias16);
  a[3] = *reinterpret_cast<const uint32_t*>(&v);
}
// Programmatic dependent launch: every kernel of a layer is launched early and
// waits here before touching what its predecessor writes; it releases its own
// successor right after, so each launch and prologue hides behind the previous
// kernel instead of following it.
__device__ __forceinline__ void pdl_wait() {
  asm volatile("griddepcontrol.wait;" ::: "memory");
}
__device__ __forceinline__ void pdl_release() {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}
__device__ __forceinline__ float e8m0(uint32_t byte) {
  return __uint_as_float(byte << 23);
}

// Element k of an activation row -> its slot in the f16 B-fragment layout:
// [k/32 group][tq][k16 block within group][b0 lo, b0 hi, b1 lo, b1 hi].
__device__ __forceinline__ int frag_slot(int k) {
  const int kb = k / 16, r = k % 16, tq = (r % 8) / 2, reg = r / 8, h = r % 2;
  return (((kb / 2) * 4 + tq) * 2 + kb % 2) * 4 + reg * 2 + h;
}

// Scale a row so its largest magnitude is 2^13 or below and store it in f16;
// the power of two (times the weights' 2^14) goes to *scale.
__device__ __forceinline__ float row_scale(float m, float* scale) {
  const int t = m > 0.f ? static_cast<int>(ceilf(log2f(m))) - 13 : 0;
  *scale = exp2f(static_cast<float>(t + PLACEMENT_EXP));
  return exp2f(static_cast<float>(-t));
}

__device__ __forceinline__ float block_max(float m, float* red) {
  for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
  if (threadIdx.x % 32 == 0) red[threadIdx.x / 32] = m;
  __syncthreads();
  if (threadIdx.x < 32) {
    m = threadIdx.x < blockDim.x / 32 ? red[threadIdx.x] : 0.f;
    for (int o = 16; o; o >>= 1)
      m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if (threadIdx.x == 0) red[0] = m;
  }
  __syncthreads();
  m = red[0];
  __syncthreads();
  return m;
}


// ---------------------------------------------------------------- 1: prep
// The last T + 1 CTAs of each call, before their own share of the layer.
// CTA GRID - 1 - T + t: token row t -> scaled f16 fragments (x13, xs13).
__device__ void prep_token(Workspace* ws, const __nv_bfloat16* x, int t,
                           float* red) {
  // every element loads before any is used: this is a latency chain
  constexpr int PER = (HIDDEN + THREADS - 1) / THREADS;
  const __nv_bfloat16* __restrict__ xr = x + static_cast<size_t>(t) * HIDDEN;
  float v[PER], m = 0.f;
#pragma unroll
  for (int i = 0; i < PER; ++i) {
    const int k = threadIdx.x + i * THREADS;
    v[i] = k < HIDDEN ? __bfloat162float(xr[k]) : 0.f;
  }
#pragma unroll
  for (int i = 0; i < PER; ++i) m = fmaxf(m, fabsf(v[i]));
  const float inv = row_scale(block_max(m, red), &ws->xs13[t]);
  __half* __restrict__ out = ws->x13 + static_cast<size_t>(t) * HIDDEN;
#pragma unroll
  for (int i = 0; i < PER; ++i) {
    const int k = threadIdx.x + i * THREADS;
    if (k < HIDDEN) out[frag_slot(k)] = __float2half_rn(v[i] * inv);
  }
}

// CTA GRID - 1: this GPU's per-tier expert lists. Each route's position among
// its expert's; the first route then opens ceil(n / MAX_TOK) consecutive
// entries, MAX_TOK routes each. Also zeroes the next call's counters and this
// call's entries' ready flags.
__device__ void prep_lists(const Params& p, Counters& next_call, int* n_of,
                           int* count) {
  Workspace* ws = p.ws;
  const int slots = p.hot_size + p.cold_size, routes = p.T * TOPK;
  const int r = threadIdx.x;
  int* base_of = n_of + slots;
  // every global read is issued up front: this block is a latency chain
  int e = -1;
  float wt = 0.f;
  if (r < routes) {
    e = p.ids[r];
    if (p.padding != nullptr && p.padding[r / TOPK]) e = -1;
    wt = p.wt[r];
  }
  int tier = -1, local = -1;
  if (e >= 0) {
    const int h = p.hot_map[e], c = p.cold_map[e];
    if (h >= 0) {
      tier = 0;
      local = h;
    } else if (c >= 0) {
      tier = 1;
      local = c;
    }
  }
  for (int i = threadIdx.x; i < slots; i += blockDim.x) n_of[i] = 0;
  if (threadIdx.x < 2) count[threadIdx.x] = 0;
  int* zero = reinterpret_cast<int*>(&next_call);
  for (int i = threadIdx.x; i < static_cast<int>(sizeof(Counters) / 4);
       i += blockDim.x)
    zero[i] = 0;
  __syncthreads();
  const int idx = tier * p.hot_size + local;
  int pos = -1;
  if (tier >= 0) pos = atomicAdd(&n_of[idx], 1);
  __syncthreads();
  if (pos == 0) {
    const int n = n_of[idx], entries = (n + MAX_TOK - 1) / MAX_TOK;
    const int base = atomicAdd(&count[tier], entries);
    base_of[idx] = base;
    for (int q = 0; q < entries; ++q) {
      Expert& ex = ws->lists[tier][base + q];
      ex.local = local;
      ex.ntok = min(MAX_TOK, n - q * MAX_TOK);
    }
  }
  __syncthreads();
  if (tier >= 0) {
    Expert& ex = ws->lists[tier][base_of[idx] + pos / MAX_TOK];
    const int i = pos % MAX_TOK;
    ex.tok[i] = r / TOPK;
    ex.route[i] = r;
    ex.wt[i] = wt * p.routed_scale;
  }
  for (int q = 0; q < 2; ++q)
    for (int i = threadIdx.x; i < count[q]; i += blockDim.x)
      ws->ready[q][i] = 0;
  if (threadIdx.x < 2) ws->counts[threadIdx.x] = count[threadIdx.x];
  // nothing contributes to the output: it is zero
  if (count[0] + count[1] == 0 && !p.has_shared)
    for (int i = threadIdx.x; i < p.T * HIDDEN; i += blockDim.x)
      p.out[i] = __float2bfloat16_rn(0.f);
}

// ---------------------------------------------------------------- 2: the layer
__device__ __forceinline__ int ld_acquire(const int* p) {
  int v;
  asm volatile("ld.acquire.gpu.global.b32 %0, [%1];"
               : "=r"(v)
               : "l"(p)
               : "memory");
  return v;
}
__device__ __forceinline__ void st_release(int* p, int v) {
  asm volatile("st.release.gpu.global.b32 [%0], %1;" ::"l"(p), "r"(v)
               : "memory");
}
__device__ __forceinline__ void fence_proxy_async() {
  asm volatile("fence.proxy.async.global;" ::: "memory");
}
// predicated fire-and-forget add: no branch per element
__device__ __forceinline__ void red_add_if(float* a, float v, bool p) {
  asm volatile(
      "{\n .reg .pred q;\n setp.ne.b32 q, %2, 0;\n @q red.global.add.f32 [%0], "
      "%1;\n}" ::"l"(a),
      "f"(v), "r"(static_cast<int>(p))
      : "memory");
}
__device__ __forceinline__ void red_add_v4(float* a, float4 v) {
  asm volatile("red.global.v4.f32.add [%0], {%1, %2, %3, %4};" ::"l"(a),
               "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w)
               : "memory");
}
__device__ __forceinline__ void sts_f32(uint32_t a, float v) {
  asm volatile("st.shared.f32 [%0], %1;" ::"r"(a), "f"(v));
}
__device__ __forceinline__ float4 lds_f4(uint32_t a) {
  float4 v;
  asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3}, [%4];"
               : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
               : "r"(a));
  return v;
}
// One warp's 64 rows x ntok tokens of a routed unit (acc * per-token scale)
// added into base + row(token) * stride: staged through the warp's smem
// scratch [token][68] (conflict-free), then vector reds over contiguous rows.
constexpr int SCR_LD = 68;
__device__ __forceinline__ void flush_rows(float (*acc)[4], const float* fs,
                                           uint32_t scr, float* base,
                                           const int* rows4, int stride,
                                           int ntok, int g, int tq, int lane) {
#pragma unroll
  for (int mb = 0; mb < 4; ++mb)
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      sts_f32(
          scr + ((2 * tq + (i & 1)) * SCR_LD + mb * 16 + g + (i >> 1) * 8) * 4,
          acc[mb][i] * fs[i & 1]);
      acc[mb][i] = 0.f;
    }
  __syncwarp();
#pragma unroll
  for (int j = 0; j < MAX_TOK / 2; ++j) {  // token c = lane / 16 + 2 j
    const int k = lane + 32 * j;
    if (k < ntok * 16) {
      const int c = k >> 4, r4 = k & 15;
      const float4 v = lds_f4(scr + (c * SCR_LD + r4 * 4) * 4);
      red_add_v4(base + static_cast<size_t>(rows4[j]) * stride + r4 * 4, v);
    }
  }
  __syncwarp();
}
__device__ __forceinline__ void fence_acq_rel_gpu() {
  asm volatile("fence.acq_rel.gpu;" ::: "memory");
}
// w13 completion handoff: consumer warps 1..7 arrive, warp 0 syncs
__device__ __forceinline__ void handoff_arrive() {
  asm volatile("bar.arrive 2, %0;" ::"n"(CONSUMER_WARPS * 32) : "memory");
}
__device__ __forceinline__ void handoff_sync() {
  asm volatile("bar.sync 2, %0;" ::"n"(CONSUMER_WARPS * 32) : "memory");
}
__device__ __forceinline__ void consumer_sync() {
  asm volatile("bar.sync 1, %0;" ::"n"(CONSUMER_WARPS * 32) : "memory");
}

// silu(gate) * up for one entry's routes, one warp per route: f16 fragments
// of x2 with the row's power-of-two scale; the y13 rows are zeroed for the
// next call.
__device__ __forceinline__ void activate_route(Workspace* ws, int r, int lane) {
  float* __restrict__ yr = ws->y13 + static_cast<size_t>(r) * 2 * INTER;
  float a[INTER / 32], m = 0.f;
#pragma unroll
  for (int q = 0; q < INTER / 32; ++q) {
    const float g = __ldcg(yr + q * 32 + lane),
                u = __ldcg(yr + INTER + q * 32 + lane);
    a[q] = __fdividef(g, 1.f + __expf(-g)) * u;
    m = fmaxf(m, fabsf(a[q]));
  }
  for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
  float scale;
  const float inv = row_scale(m, &scale);
  if (lane == 0) ws->xs2[r] = scale;
  __half* __restrict__ out = ws->x2 + static_cast<size_t>(r) * X2_LD;
  if (lane == 0) *reinterpret_cast<float*>(out + INTER) = scale;
#pragma unroll
  for (int q = 0; q < INTER / 32; ++q) {
    out[frag_slot(q * 32 + lane)] = __float2half_rn(a[q] * inv);
    yr[q * 32 + lane] = 0.f;
    yr[INTER + q * 32 + lane] = 0.f;
  }
}

// The shared expert's silu * up for all T tokens (plain bf16 rows of x2s);
// y13s is zeroed for the next call.
__device__ __forceinline__ void activate_shared(Workspace* ws, int T, int warp,
                                                int lane) {
  for (int tok = warp; tok < T; tok += CONSUMER_WARPS) {
    float* __restrict__ yr = ws->y13s + static_cast<size_t>(tok) * 2 * INTER;
    __nv_bfloat16* __restrict__ out =
        ws->x2s + static_cast<size_t>(tok) * INTER;
#pragma unroll
    for (int q = 0; q < INTER / 32; ++q) {
      const int k = q * 32 + lane;
      const float g = __ldcg(yr + k), u = __ldcg(yr + INTER + k);
      out[k] =
          __float2bfloat16_rn(__fdividef(g, 1.f + __expf(-g)) * u);
      yr[k] = 0.f;
      yr[INTER + k] = 0.f;
    }
  }
}

// One routed unit's 16 group steps for this warp. PH 0 (w13): 4 row blocks x
// 2 K halves of a 64-row x 1024 box; PH 1 (w2): 8 row blocks of a 128-row x
// 512 box. All smem offsets are immediates off two per-warp bases.
// One routed unit for this warp: all 4 row blocks of a 64-row tile over 4
// group steps of K, so each lane's 16 B of a Marlin k16 row (its 4 row blocks)
// is one conflict-free LDS.128, and each activation fragment feeds 4 MMAs.
// w13 unit (64 rows x 1024): warp w takes K steps [4w, 4w + 4); w2 unit (128
// rows x 512): warp w takes 64-row half w / 4, K steps [4 (w % 4), + 4).
template <int PH>
__device__ __forceinline__ void consume_routed(uint32_t wb, uint32_t sb,
                                               uint32_t xb, float (*acc)[4]) {
  constexpr int KROW = PH ? 1024 : 512;  // bytes per k16 row of the box
  constexpr int SROW = PH ? 256 : 128;   // bytes per scale group row
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const uint4 w0 = lds_v4(wb + (2 * j) * KROW);
    const uint4 w1 = lds_v4(wb + (2 * j + 1) * KROW);
    const uint4 sw = lds_v4(sb + j * SROW);
    const uint4 xv = lds_v4(xb + j * 64);
    const uint32_t w0s[4] = {w0.x, w0.y, w0.z, w0.w},
                   w1s[4] = {w1.x, w1.y, w1.z, w1.w};
    const uint32_t sws[4] = {sw.x, sw.y, sw.z, sw.w};
#pragma unroll
    for (int mb = 0; mb < 4; ++mb) {
      const float zero[4] = {0.f, 0.f, 0.f, 0.f};
      float d[4];
      uint32_t a[4];
      decode_int4_fast(w0s[mb], a);
      mma_f16(d, a, xv.x, xv.y, zero);
      decode_int4_fast(w1s[mb], a);
      mma_f16(d, a, xv.z, xv.w, d);
      const float s0 = __uint_as_float(sws[mb] << 16),
                  s1 = __uint_as_float(sws[mb] & 0xFFFF0000u);
      acc[mb][0] = fmaf(s0, d[0], acc[mb][0]);
      acc[mb][1] = fmaf(s0, d[1], acc[mb][1]);
      acc[mb][2] = fmaf(s1, d[2], acc[mb][2]);
      acc[mb][3] = fmaf(s1, d[3], acc[mb][3]);
    }
  }
}

// Work queues. Each tier's work is a list of groups, claimed in order with
// one atomic by whichever CTA's producer is free: the shared expert's w13
// (tile x 12 chunks), the routed w13 (entry x tile, 6 chunks), the shared w2
// (tile, 8 chunks), the routed w2 (entry x 128-row unit). An entry's w13
// groups go out early and to many CTAs at once, so its w2 units rarely wait;
// the queue ends on 1-unit groups, so CTAs finish within a unit of each
// other. A CTA whose own tier's queue is empty takes the other tier's.
enum Kind : int { K_S0 = 0, K_R0 = 1, K_S1 = 2, K_R1 = 3, K_END = 15 };
constexpr int SCH0 = 12;               // shared w13 chunks per group
constexpr int SG0 = CHUNKS_S0 / SCH0;  // groups per shared w13 tile
static_assert(CHUNKS_S0 % SCH0 == 0, "shared w13 groups tile K");
struct Group {
  int kind, x, c0,
      nch;  // x: R0 entry * TILES0 + tile, R1 entry * TILES1 + unit, S tile
};
#ifndef TD_GR1
  #define TD_GR1 1  // routed w2 units per claim; 2-4 within noise
#endif
constexpr int GR1 =
    TD_GR1;  // routed w2 units per group (consecutive tiles, one entry)
static_assert(TILES1 % GR1 == 0, "w2 groups tile an entry");
// A routed w13 tile's 12 K chunks go out as R0S adjacent groups, so several
// CTAs share a late tile and its entry's activation is not stuck behind one
// CTA's 12 serial units (the w2 producers spin on it).
#ifndef TD_R0S
  #define TD_R0S 1  // 2-4 measured slower: more w13 flushes
#endif
constexpr int R0S = TD_R0S, R0CH = CHUNKS0 / R0S;
static_assert(CHUNKS0 % R0S == 0, "w13 groups split the K chunks evenly");
// warps 1..7 run at most STAGES units ahead of warp 0 (the ring), so with
// more units than that between w13 flushes no warp can arrive at the handoff
// barrier for the next flush before warp 0 has synced on this one
static_assert(R0CH > STAGES, "w13 handoff barrier generations cannot overlap");
__device__ __forceinline__ int queue_len(int n, bool sh) {
  return (sh ? TILES_S0 * SG0 + TILES_S1 : 0) +
         n * (TILES0 * R0S + TILES1 / GR1);
}
__device__ __forceinline__ Group group_at(int gi, int n, bool sh) {
  const int ns0 = sh ? TILES_S0 * SG0 : 0, nr0 = n * TILES0 * R0S,
            ns1 = sh ? TILES_S1 : 0;
  if (gi < ns0) return {K_S0, gi / SG0, (gi % SG0) * SCH0, SCH0};
  gi -= ns0;
  if (gi < nr0) return {K_R0, gi / R0S, (gi % R0S) * R0CH, R0CH};
  gi -= nr0;
  if (gi < ns1) return {K_S1, gi, 0, CHUNKS_S1};
  gi -= ns1;
  return {K_R1, gi, 0, GR1};
}

// The warp that completes a 128-row output tile (every flush into it counted)
// writes the tile's T rows as bf16 and zeroes them in y for the next call.
__device__ __forceinline__ void tile_done(Workspace* ws, Counters& cur,
                                          int tile, int expect, int T,
                                          __nv_bfloat16* __restrict__ out,
                                          int lane) {
#ifdef TD_ABL_NOTILE
  return;  // timing probe: no counting, no output
#endif
  __syncwarp();  // every lane's reds into the tile, before lane 0's count
  int n = 0;
  if (lane == 0) {
    __threadfence();
    n = atomicAdd(&cur.tiles[tile], 1) + 1;
  }
  if (__shfl_sync(0xffffffffu, n, 0) != expect) return;
  __threadfence();
  __syncwarp();  // lane 0's acquire (the count) before every lane's reads
  constexpr int Q = 128 / 4;  // float4 per tile row
#pragma unroll 4
  for (int tok = 0; tok < T; ++tok) {
    const size_t i = static_cast<size_t>(tok) * HIDDEN + tile * 128 + lane * 4;
    float4* y = reinterpret_cast<float4*>(ws->y + i);
    const float4 v = __ldcg(y);
    __stcg(y, make_float4(0.f, 0.f, 0.f, 0.f));
    __nv_bfloat162* o = reinterpret_cast<__nv_bfloat162*>(out + i);
    o[0] = __floats2bfloat162_rn(v.x, v.y);
    o[1] = __floats2bfloat162_rn(v.z, v.w);
  }
  static_assert(Q == 32, "one float4 of a tile row per lane");
}

__global__ void __launch_bounds__(THREADS, 1)
    layer_kernel(const __grid_constant__ Params p) {
  extern __shared__ __align__(128) unsigned char smem[];
  uint64_t* full = reinterpret_cast<uint64_t*>(smem);
  uint64_t* empty = full + STAGES;
  // 1 KB aligned (128 B swizzle) by an offset into smem, so every load off it
  // stays a shared-space LDS rather than a generic LD
  unsigned char* ring =
      smem + SMEM_HEAD +
      ((1024u - ((smem_u32(smem) + SMEM_HEAD) & 1023u)) & 1023u);
  __shared__ int4 desc[STAGES];
  // per stage, routed units: each token's destination row (w13: its route's
  // y13 row; w2: its token's y row) and the scale its flush applies
  // (w13: xs13[token]; w2: the route weight, times the row's xs2 at the
  // flush), written by the producer
  __shared__ int sd_row[STAGES][MAX_TOK];
  __shared__ float sd_f[STAGES][MAX_TOK];
  __shared__ int s_last;
  __shared__ int s_expect;  // flushes per output tile, once the lists exist
  __shared__ unsigned long long s_ticket;
  __shared__ float red[32];
  __shared__ int s_count[2];
  __shared__ int s_slots[4 * MAX_LIST];  // prep_lists: n_of, base_of
  __shared__ __align__(16) float scr_all[CONSUMER_WARPS][MAX_TOK * SCR_LD];
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  Workspace* ws = p.ws;
  if (threadIdx.x == 0) {
    for (int s = 0; s < STAGES; ++s) {
      mbar_init(&full[s], 1);
      mbar_init(&empty[s], CONSUMER_WARPS);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    // the previous call's CTAs all counted themselves before this call began
    s_ticket = atomicAdd(&ws->arrive, 1ull);
  }
  __syncthreads();
  const unsigned long long call = s_ticket / GRID;
  Counters& cur = ws->c[call & 1];
  const int epoch = static_cast<int>(call & 0x3fffffff) + 1;
  const int T = p.T;
  const bool sh = p.has_shared;
  pdl_wait();  // the router's ids and weights (and x, written before them)
  pdl_release();
  const int prep0 = GRID - 1 - T;
  if (static_cast<int>(blockIdx.x) >= prep0) {
    if (blockIdx.x == GRID - 1)
      prep_lists(p, ws->c[(call + 1) & 1], s_slots, s_count);
    else
      prep_token(ws, p.x, blockIdx.x - prep0, red);
    fence_proxy_async();  // x13 is read by bulk copies
    __syncthreads();
    if (threadIdx.x == 0) {
      __threadfence();
      atomicAdd(&cur.prep, 1);
    }
  }

  if (warp == CONSUMER_WARPS) {  // producer warp
    if (lane == 0) {
      for (int q = 0; q < 2; ++q)
        for (int k = 0; k < 2; ++k) {
          asm volatile("prefetch.tensormap [%0];" ::"l"(
                           reinterpret_cast<uint64_t>(&p.tier[q].w[k]))
                       : "memory");
          asm volatile("prefetch.tensormap [%0];" ::"l"(
                           reinterpret_cast<uint64_t>(&p.tier[q].s[k]))
                       : "memory");
        }
      if (sh)
        for (int k = 0; k < 2; ++k)
          asm volatile("prefetch.tensormap [%0];" ::"l"(
                           reinterpret_cast<uint64_t>(&p.sw[k]))
                       : "memory");
    }
    float xs13r = 0.f;  // token lane's x13 scale, once prep is in
    int last_ready = -1, it = 0;
    // one group's units into the ring
    auto issue = [&](int q, int gi, int n) {
      const Group gr = group_at(gi, n, q == 0 && sh);
      const Tier& tr = p.tier[q];
      // a routed group's entry record, once per group
      const bool routed = gr.kind == K_R0 || gr.kind == K_R1;
      int ei = 0, t0 = 0, ntok = 0, local = 0, tokl = 0, routel = 0;
      float fl = 0.f;
      if (gr.kind == K_R0) {
        ei = gr.x / TILES0;
        t0 = gr.x - ei * TILES0;
      } else if (gr.kind == K_R1) {
        ei = gr.x / (TILES1 / GR1);
        t0 = (gr.x - ei * (TILES1 / GR1)) * GR1;
      }
      if (routed) {
        // one round trip: every field at once (lanes past ntok read a valid
        // slot and are masked later)
        const Expert& e = ws->lists[q][ei];
        const int l7 = lane & (MAX_TOK - 1);
        ntok = e.ntok;
        local = e.local;
        tokl = e.tok[l7];
        routel = e.route[l7];
        const float wtl = e.wt[l7];
        const float xs = __shfl_sync(0xffffffffu, xs13r, tokl & 31);
        // w13: the token's x13 scale; w2: the route weight (consumers apply
        // the x2 row's scale, which arrives with the row)
        fl = gr.kind == K_R0 ? xs : wtl;
      }
      for (int ci = 0; ci < gr.nch; ++ci, ++it) {
        const int s = it % STAGES, c = gr.c0 + ci;
        unsigned char* dst = ring + static_cast<size_t>(s) * STAGE_BYTES;
        if (lane == 0 && it >= STAGES)
          mbar_wait(&empty[s], ((it / STAGES) - 1) & 1);
        __syncwarp();
        const int hdr = gr.kind | q << 4 | (ci == 0) << 8 |
                        (ci == gr.nch - 1) << 9 | gr.nch << 16;
        if (gr.kind == K_R0) {
          if (lane < ntok) {
            sd_row[s][lane] = routel;
            sd_f[s][lane] = fl;
          }
          __syncwarp();
          if (lane == 0) {
            desc[s] = make_int4(hdr, ei, t0, ntok);
            mbar_expect_tx(&full[s], W_BYTES + S_BYTES + ntok * XROW_BYTES0);
            tma_3d(dst, &tr.w[0], t0 * 2 * R0, c * KT0, local, &full[s]);
            tma_3d(dst + W_BYTES, &tr.s[0], t0 * R0, c * G0, local, &full[s]);
          }
          __syncwarp();
          if (lane < ntok)
            bulk_g2s(dst + W_BYTES + S_BYTES + lane * XROW_STRIDE,
                     ws->x13 + static_cast<size_t>(tokl) * HIDDEN + c * CK0,
                     XROW_BYTES0, &full[s]);
        } else if (gr.kind == K_R1) {
          const int t = t0 + ci;
          if (lane < ntok) {
            sd_row[s][lane] = tokl;
            sd_f[s][lane] = fl;
          }
          __syncwarp();
          if (lane == 0) {
            desc[s] = make_int4(hdr, ei, t, ntok);
            mbar_expect_tx(&full[s], W_BYTES + S_BYTES + ntok * X2_COPY);
            tma_3d(dst, &tr.w[1], t * 256, 0, local, &full[s]);
            tma_3d(dst + W_BYTES, &tr.s[1], t * 128, 0, local, &full[s]);
          }
          // weights are in flight; only the activation rows wait for the
          // entry's ready (every copying lane acquires for itself)
          if (ei != last_ready) {
            while (ld_acquire(&ws->ready[q][ei]) != 1) __nanosleep(32);
            fence_proxy_async();
            last_ready = ei;
          }
          __syncwarp();
          if (lane < ntok)
            bulk_g2s(dst + W_BYTES + S_BYTES + lane * XROW_STRIDE,
                     ws->x2 + static_cast<size_t>(routel) * X2_LD, X2_COPY,
                     &full[s]);
        } else {  // shared expert: w13 reads x itself, w2 the activation
          if (lane == 0) {
            desc[s] = make_int4(hdr, gr.x, c, 0);
            mbar_expect_tx(&full[s], W_BYTES + T * XS_BYTES);
            tma_2d(dst, &p.sw[gr.kind == K_S0 ? 0 : 1], c * CKS, gr.x * RS,
                   &full[s]);
          }
          if (gr.kind == K_S1 &&
              last_ready != -2) {  // every copying lane acquires
            while (ld_acquire(&ws->ready_s) != epoch) __nanosleep(32);
            fence_proxy_async();
          }
          if (gr.kind == K_S1) last_ready = -2;
          __syncwarp();
          if (lane < T)
            bulk_g2s(dst + W_BYTES + lane * XS_STRIDE,
                     (gr.kind == K_S0
                          ? p.x + static_cast<size_t>(lane) * HIDDEN
                          : ws->x2s + static_cast<size_t>(lane) * INTER) +
                         c * CKS,
                     XS_BYTES, &full[s]);
        }
      }
    };
    auto claim = [&](int q) {
      int g = 0;
      if (lane == 0) g = atomicAdd(&cur.next[q], 1);
      return __shfl_sync(0xffffffffu, g, 0);
    };
    // Before the lists exist only the shared expert's w13 can go out: the
    // hot queue starts with it. The first claim past it is held.
    const int ns0 = sh ? TILES_S0 * SG0 : 0;
    int gi = claim(0);
    while (gi < ns0) {
      const int gn = claim(0);  // its round trip overlaps this group's issue
      issue(0, gi, 0);
      gi = gn;
    }
    while (ld_acquire(&cur.prep) != T + 1) __nanosleep(32);
    fence_proxy_async();
    const int n_tier[2] = {ws->counts[0], ws->counts[1]};
    xs13r = lane < T ? ws->xs13[lane] : 0.f;
    if (lane == 0) s_expect = 8 * (n_tier[0] + n_tier[1]) + (sh ? 4 : 0);
    const int len[2] = {queue_len(n_tier[0], sh), queue_len(n_tier[1], false)};
    // CTAs [0, cold_ctas) run the cold tier first, the rest the hot tier;
    // whoever runs out takes the other tier's queue
    const int cold_ctas = len[1] == 0 ? 0 : len[0] == 0 ? GRID : TD_COLD_CTAS;
    int q = 0;
    if (static_cast<int>(blockIdx.x) < cold_ctas) {
      if (gi < len[0]) issue(0, gi, n_tier[0]);
      q = 1;
      last_ready = -1;  // entry numbers restart in the other tier
      gi = claim(1);
    }
    bool stolen = false;
    while (true) {
      if (gi >= len[q]) {
#ifdef TD_NO_STEAL
        break;
#endif
        if (stolen) break;
        stolen = true;
        q ^= 1;
        last_ready = -1;
        gi = claim(q);
        continue;
      }
      const int gn = claim(q);
      issue(q, gi, n_tier[q]);
      gi = gn;
    }
    if (lane == 0) {  // end of work
      const int s = it % STAGES;
      if (it >= STAGES) mbar_wait(&empty[s], ((it / STAGES) - 1) & 1);
      desc[s] = make_int4(K_END, 0, 0, 0);
      mbar_arrive(&full[s]);
    }
    return;
  }

  // consumer warps
  const int g = lane / 4, tq = lane % 4;
  const int nt_n = (T + 7) / 8;
  float acc[4][4] = {};
  float accs[2][4][4] = {};
  // per-warp smem offsets within a stage, computed once
  const uint32_t ring_u = smem_u32(ring), full_u = smem_u32(full),
                 empty_u = smem_u32(empty);
  const uint32_t desc_u = smem_u32(desc), sdf_u = smem_u32(&sd_f[0][2 * tq]);
  const uint32_t scr_u = smem_u32(scr_all[warp]),
                 sdr0_u = smem_u32(&sd_row[0][0]);
  static_assert(CONSUMER_WARPS == 8,
                "8 K slices (w13), 2 halves x 4 K slices (w2)");
  const int h1 = warp / 4, k1 = warp % 4;
  const uint32_t wo1 = (k1 * 8) * 1024 + h1 * 512 + lane * 16;
  const uint32_t so1 = W_BYTES + (k1 * 4) * 256 + h1 * 128 + 16 * g;
  const uint32_t xo1 =
      W_BYTES + S_BYTES + g * XROW_STRIDE + (k1 * 16 + tq) * 16;
  int s = 0;
  uint32_t ph = 0;
  for (int it = 0;; ++it) {
    mbar_wait_a(full_u + 8 * s, ph);
    const uint4 du = lds_v4(desc_u + 16 * s);
    const int4 d = make_int4(du.x, du.y, du.z, du.w);
    const int kind = d.x & 15;
    if (kind == K_END) break;
    const int q = (d.x >> 4) & 1, last = (d.x >> 9) & 1, nch = d.x >> 16;
    const unsigned char* st = ring + static_cast<size_t>(s) * STAGE_BYTES;
    const uint32_t st_u = ring_u + s * STAGE_BYTES;
    const uint32_t empty_s = empty_u + 8 * s;
    const Expert* experts = ws->lists[q];
    // this lane's two tokens of a routed unit: destination rows and scales
    const int ntok = d.w;
    // read unconditionally (stale past ntok, masked at the flush)
    float fs[2] = {__uint_as_float(lds_u32(sdf_u + s * MAX_TOK * 4)),
                   __uint_as_float(lds_u32(sdf_u + s * MAX_TOK * 4 + 4))};
    if (kind == K_R1) {  // * the x2 row's scale, which arrived with the row
      const uint32_t xr = ring_u + s * STAGE_BYTES + W_BYTES + S_BYTES +
                          2 * tq * XROW_STRIDE + XROW_BYTES1;
      fs[0] *= __uint_as_float(lds_u32(xr));
      fs[1] *= __uint_as_float(lds_u32(xr + XROW_STRIDE));
    }
    // the flush's destination rows (tokens lane / 16 + 2 j), read before the
    // stage is released: the producer rewrites sd_row[s] for the next unit
    int rows4[MAX_TOK / 2];
#pragma unroll
    for (int j = 0; j < MAX_TOK / 2; ++j)
      rows4[j] = static_cast<int>(
          lds_u32(sdr0_u + (s * MAX_TOK + (lane >> 4) + 2 * j) * 4));

    if (kind == K_R0) {
      consume_routed<1>(st_u + wo1, st_u + so1, st_u + xo1, acc);
      __syncwarp();
      if (lane == 0) mbar_arrive_a(empty_s);
      if (last) {
        const int ei = d.y, t = d.z;
        flush_rows(acc, fs, scr_u, ws->y13 + t * R0 + h1 * 64, rows4, 2 * INTER,
                   ntok, g, tq, lane);
        // warps 1..7 hand their reds to warp 0 and go on; warp 0 counts the
        // chunks and, if this CTA completed the entry, activates its routes
        // and publishes ready. Nobody else waits on the count's round trip.
        if (warp != 0) {
          handoff_arrive();
        } else {
          handoff_sync();
          int done = 0;
          if (lane == 0) {
            __threadfence();
            done = atomicAdd(&cur.done13[q][ei], nch) + nch == UNITS0;
          }
          if (__shfl_sync(0xffffffffu, done, 0)) {
            __threadfence();
            __syncwarp();  // lane 0's acquire (the count) ordered before every
                           // lane's y13 reads
            const Expert& e = experts[ei];
            const int n = e.ntok;
            for (int r = 0; r < n; ++r) activate_route(ws, e.route[r], lane);
            fence_proxy_async();
            __threadfence();
            __syncwarp();
            if (lane == 0) st_release(&ws->ready[q][ei], 1);
          }
        }
      }
    } else if (kind == K_R1) {
      consume_routed<1>(st_u + wo1, st_u + so1, st_u + xo1, acc);
      __syncwarp();
      if (lane == 0) mbar_arrive_a(empty_s);
      const int t = d.z;
      flush_rows(acc, fs, scr_u, ws->y + t * R1 + h1 * 64, rows4, HIDDEN, ntok,
                 g, tq, lane);
      tile_done(ws, cur, t, s_expect, T, p.out, lane);
    } else {  // shared expert: 32 rows per warp, 4 k16 steps, nt_n token tiles
      const unsigned char* xa = st + W_BYTES;
#pragma unroll
      for (int ks = 0; ks < CKS / 16; ++ks) {
        uint32_t af[2][4];
#pragma unroll
        for (int rb = 0; rb < 2; ++rb) {
          const int row =
              warp * 32 + rb * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
          const int ck = ks * 2 + (lane >> 4);
          ldmatrix_x4(af[rb], st + row * 128 + ((ck ^ (row & 7)) << 4));
        }
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
          if (nt < nt_n) {
            // token row nt * 8 + g, k = 16 ks + 2 tq (+ 8): plain bf16 rows
            const unsigned char* xr =
                xa + (nt * 8 + g) * XS_STRIDE + ks * 32 + tq * 4;
            const uint32_t b0 = *reinterpret_cast<const uint32_t*>(xr),
                           b1 = *reinterpret_cast<const uint32_t*>(xr + 16);
            mma_bf16(accs[0][nt], af[0], b0, b1);
            mma_bf16(accs[1][nt], af[1], b0, b1);
          }
        }
      }
      __syncwarp();
      if (lane == 0) mbar_arrive_a(empty_s);
      if (last) {
        const int t = d.y;
#pragma unroll
        for (int rb = 0; rb < 2; ++rb)
#pragma unroll
          for (int nt = 0; nt < 4; ++nt)
#pragma unroll
            for (int i = 0; i < 4; ++i) {
              const int tok = nt * 8 + 2 * tq + (i & 1);
              const int row = t * RS + warp * 32 + rb * 16 + g + (i >> 1) * 8;
              if (nt < nt_n && tok < T) {
                if (kind == K_S0)
                  atomicAdd(
                      &ws->y13s[static_cast<size_t>(tok) * 2 * INTER + row],
                      accs[rb][nt][i]);
                else
                  atomicAdd(&ws->y[static_cast<size_t>(tok) * HIDDEN + row],
                            p.shared_scale * accs[rb][nt][i]);
              }
              accs[rb][nt][i] = 0.f;
            }
        if (kind == K_S0) {
          consumer_sync();
          if (threadIdx.x == 0) {
            __threadfence();
            s_last = atomicAdd(&cur.done_s, nch) + nch == UNITS_S0;
          }
          consumer_sync();
          if (s_last) {
            __threadfence();
            activate_shared(ws, T, warp, lane);
            fence_proxy_async();
            __threadfence();
            consumer_sync();
            if (threadIdx.x == 0) st_release(&ws->ready_s, epoch);
          }
        } else {
          tile_done(ws, cur, 2 * t + warp / 4, s_expect, T, p.out, lane);
        }
      }
    }
    if (++s == STAGES) {
      s = 0;
      ph ^= 1;
    }
  }
}

// ---------------------------------------------------------------- host entry
// Plain C ABI (no torch headers: the build is the kernel alone), called from
// Python through ctypes with raw device pointers and the current stream.
#define TD_REQUIRE(c, msg)                              \
  do {                                                  \
    if (!(c)) {                                         \
      std::fprintf(stderr, "tiered_decode: %s\n", msg); \
      return -1;                                        \
    }                                                   \
  } while (0)

// [n2][n1][n0] contiguous elements of elem bytes; box {b0, b1, 1}
static int make_map3(CUtensorMap* map, const void* p, uint64_t n0, uint64_t n1,
                     uint64_t n2, int elem, CUtensorMapDataType type,
                     uint32_t b0, uint32_t b1) {
  const cuuint64_t dims[3] = {n0, n1, n2};
  const cuuint64_t strides[2] = {n0 * elem, n0 * n1 * elem};
  const cuuint32_t box[3] = {b0, b1, 1}, unit[3] = {1, 1, 1};
  return cuTensorMapEncodeTiled(
             map, type, 3, const_cast<void*>(p), dims, strides, box, unit,
             CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
             CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
             CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS
             ? 0
             : -1;
}
// [rows][cols] bf16 with a {b0, b1} box, 128 B swizzled
static int make_map_2d_sw128(CUtensorMap* map, const void* p, uint64_t cols,
                             uint64_t rows, uint32_t b0, uint32_t b1) {
  const cuuint64_t dims[2] = {cols, rows};
  const cuuint64_t strides[1] = {cols * 2};
  const cuuint32_t box[2] = {b0, b1}, unit[2] = {1, 1};
  return cuTensorMapEncodeTiled(
             map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(p),
             dims, strides, box, unit, CU_TENSOR_MAP_INTERLEAVE_NONE,
             CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
             CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS
             ? 0
             : -1;
}
static int fill_tier(Tier& tr, const void* w13, const void* s13, const void* w2,
                     const void* s2, int e) {
  std::memset(&tr, 0, sizeof(tr));
  if (e == 0) return 0;
  int r = make_map3(&tr.w[0], w13, 2 * INTER * 2, HIDDEN / 16, e, 4,
                    CU_TENSOR_MAP_DATA_TYPE_UINT32, 2 * R0, KT0);
  r |= make_map3(&tr.w[1], w2, HIDDEN * 2, INTER / 16, e, 4,
                 CU_TENSOR_MAP_DATA_TYPE_UINT32, 2 * R1, KT1);
  r |= make_map3(&tr.s[0], s13, 2 * INTER, HIDDEN / 32, e, 2,
                 CU_TENSOR_MAP_DATA_TYPE_UINT16, R0, G0);
  r |= make_map3(&tr.s[1], s2, HIDDEN, INTER / 32, e, 2,
                 CU_TENSOR_MAP_DATA_TYPE_UINT16, R1, G1);
  return r;
}

}  // namespace tiered_decode

using namespace tiered_decode;

extern "C" long long td_workspace_bytes() { return sizeof(Workspace); }

extern "C" int td_forward(void* out, const void* x, const int* ids,
                          const float* wt, const int* hot_map,
                          const int* cold_map, int T, const void* hw13,
                          const void* hs13, const void* hw2, const void* hs2,
                          int hot_size, const void* cw13, const void* cs13,
                          const void* cw2, const void* cs2, int cold_size,
                          void* workspace, int pdl_launch, const bool* padding,
                          const void* sw13, const void* sw2, float shared_scale,
                          float routed_scale, void* stream_ptr) {
  TD_REQUIRE(T >= 1 && T <= MAX_TOKENS, "1..32 tokens");
  TD_REQUIRE(hot_size + cold_size <= 2 * MAX_LIST, "at most 512 tier slots");
  Params p{};
  TD_REQUIRE(fill_tier(p.tier[0], hw13, hs13, hw2, hs2, hot_size) == 0,
             "hot tier maps");
  TD_REQUIRE(fill_tier(p.tier[1], cw13, cs13, cw2, cs2, cold_size) == 0,
             "cold tier maps");
  p.ws = reinterpret_cast<Workspace*>(workspace);
  p.x = reinterpret_cast<const __nv_bfloat16*>(x);
  p.out = reinterpret_cast<__nv_bfloat16*>(out);
  p.ids = ids;
  p.wt = wt;
  p.padding = padding;
  p.hot_map = hot_map;
  p.cold_map = cold_map;
  p.T = T;
  p.hot_size = hot_size;
  p.cold_size = cold_size;
  p.has_shared = sw13 != nullptr;
  p.shared_scale = shared_scale;
  p.routed_scale = routed_scale;
  if (p.has_shared) {
    TD_REQUIRE(
        make_map_2d_sw128(&p.sw[0], sw13, HIDDEN, 2 * INTER, CKS, RS) == 0,
        "sw13 map");
    TD_REQUIRE(make_map_2d_sw128(&p.sw[1], sw2, INTER, HIDDEN, CKS, RS) == 0,
               "sw2 map");
  }
  static bool attrs = false;
  if (!attrs) {
    cudaFuncSetAttribute(
        layer_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
    attrs = true;
  }
  cudaLaunchAttribute pdl[1];
  pdl[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  pdl[0].val.programmaticStreamSerializationAllowed = 1;
  cudaLaunchConfig_t c = {};
  c.gridDim = dim3(GRID);
  c.blockDim = dim3(THREADS);
  c.dynamicSmemBytes = SMEM_BYTES;
  c.stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  c.attrs = pdl;
  c.numAttrs = pdl_launch ? 1 : 0;
  return cudaLaunchKernelEx(&c, layer_kernel, p) == cudaSuccess ? 0 : -3;
}

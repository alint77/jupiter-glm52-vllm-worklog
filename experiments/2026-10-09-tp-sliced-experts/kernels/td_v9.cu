// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Decode-step MoE for tiered 4-bit experts on sm_90a: MXFP4 (MiMo-V2) or
// symmetric INT4 with bf16 group-32 scales (GLM-5.3 W4A16); both have
// 6144 x 2048 experts and top-8 routing.
//
// One persistent GEMM kernel serves both tiers of a layer: its first CTAs
// stream the cold tier's experts from pinned Grace memory over C2C, the rest
// the hot tier's from HBM, each through a warp-specialized ring of 1D TMA bulk
// copies (3D tensor maps for the strided weight / scale blocks). It reads the
// tier tensors Marlin already uses in place:
//   weights [E][K/16][N*2] int32 in gptq_marlin_repack order: the word at
//     (k_tile * N/64 + n_tile) * 128 + lane * 4 + row_block holds rows g, g+8
//     of its 16-row block at k 2tq, 2tq+1, 2tq+8, 2tq+9 (g = lane/4,
//     tq = lane%4), exactly one mma A fragment;
//   scales  [E][K/32][N] e8m0, permuted so a lane's rows g and g+8 of all four
//     row blocks sit in 8 bytes at n_tile * 64 + 8g; or bf16 in
//     marlin_permute_scales order, the same rows in 16 bytes.
// Weights decode to exact f16 (value * 2^-14): MXFP4 through e5m2 bytes and
// PRMT, INT4 through 0x6400 and one fma. Scales apply per 32-group in fp32;
// activations are f16 under a power-of-two per-row scale (exact for bf16),
// products are exact and accumulate in fp32, so numerics match Marlin's up to
// summation order. Partial sums leave through fp32 red.global.add, which lets
// CTAs split an expert's (tile, chunk) units evenly (stream-K).
//
// A layer is five launches: route/prep (which also picks the GPU for each
// replicated cold expert), w13, silu*up, w2, finalize. Tokens are
// the mma N dimension, so a list entry takes at most 8 of an expert's tokens;
// route_prep gives an expert with more (up to 32 tokens per call) consecutive
// entries, each streaming the expert's weights for its own 8.

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

#include <cstddef>
#include <cstring>
#include <cstdint>

namespace tiered_decode {

// The 4-way intermediate slice of every expert: INTER = 512 per GPU. w13
// units are 64 rows x a 1024 chunk of K (chunks innermost, accumulated in
// registers); w2 units are 128 rows x all 512 of K. Both are 32 KB of weights
// and 4 KB of bf16 scales, and give each consumer warp 16 group steps.
#ifndef TD_STAGES
  #define TD_STAGES 4
#endif
#ifndef TD_COLD_CTAS
  #define TD_COLD_CTAS 16
#endif
constexpr int HIDDEN = 6144, INTER = 512, TOPK = 8;
constexpr int MAX_TOK = 8, MAX_TOKENS = 32, MAX_ROUTES = MAX_TOKENS * TOPK,
              MAX_LIST = MAX_ROUTES;
constexpr int R0 = 64, R1 = 128;                 // rows per w13 / w2 unit
constexpr int CK0 = 1024;                        // w13 K chunk
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
constexpr int XROW_STRIDE = XROW_BYTES0 + 64;  // token rows g, g+1 on disjoint banks
// shared expert (bf16, this GPU's 512-wide TP slice of it): units of 256 rows
// x 64 of K, 128 B swizzled, for all T tokens
constexpr int RS = 256, CKS = 64;
constexpr int TILES_S0 = 2 * INTER / RS, CHUNKS_S0 = HIDDEN / CKS;
constexpr int TILES_S1 = HIDDEN / RS, CHUNKS_S1 = INTER / CKS;
constexpr int UNITS_S0 = TILES_S0 * CHUNKS_S0, UNITS_S1 = TILES_S1 * CHUNKS_S1;
static_assert(RS * CKS * 2 == W_BYTES, "a shared unit fills the weight slot");
constexpr int XS_BYTES = CKS * 2;  // one token's activation slice per shared unit
// stages start on 1 KB boundaries (128 B swizzle)
constexpr int STAGE_BYTES =
    (W_BYTES + S_BYTES + MAX_TOK * XROW_STRIDE + 1023) / 1024 * 1024;
static_assert(MAX_TOKENS * XS_BYTES <= STAGE_BYTES - W_BYTES, "shared rows fit");
constexpr int STAGES = TD_STAGES;
#ifndef TD_CWARPS
  #define TD_CWARPS 16
#endif
constexpr int CONSUMER_WARPS = TD_CWARPS;
constexpr int THREADS = (CONSUMER_WARPS + 1) * 32;
constexpr int SMEM_HEAD = 128;
static_assert(STAGES <= 8, "barrier head holds 8 stages");
constexpr int SMEM_BYTES = SMEM_HEAD + 1024 + STAGES * STAGE_BYTES;
static_assert(SMEM_BYTES <= 227 * 1024, "ring exceeds shared memory");
constexpr int PLACEMENT_EXP = 14;  // decoded weights are value * 2^-14
constexpr int GRID = 132;
constexpr int PREP_THREADS =
    1024;  // route_prep block: 6 hidden elements per thread
static_assert(CONSUMER_WARPS == 8 || CONSUMER_WARPS == 16,
              "w13: 4 row blocks x CW/4 K parts; w2: 8 row blocks x CW/8 K parts");
constexpr int KP0 = CONSUMER_WARPS / 4, KP1 = CONSUMER_WARPS / 8;  // K parts per unit
constexpr int STEPS = 32 / KP0;  // group steps per warp per routed unit
static_assert(16 / KP1 == STEPS, "w13 and w2 warps run the same number of steps");
constexpr int SROWS = RS / CONSUMER_WARPS;  // shared expert rows per warp
constexpr int SRB = SROWS / 16;              // ... in 16-row blocks

struct Expert {
  int local;  // index into the tier's tensors
  int ntok;
  int tok[MAX_TOK];    // token rows
  int route[MAX_TOK];  // token * TOPK + k
  float wt[MAX_TOK];
};

// Per tier and projection: the weight and scale tensor maps. A weight map is
// [E][K/16][N*2] int32 with a {128, 64, 1} box: one 64-row tile's 64 k16 rows,
// 512 B each, landing contiguous in shared memory. A scale map is [E][K/32][N]
// bytes with a {64, 32, 1} box.
struct alignas(64) Tier {
  CUtensorMap w[2];
  CUtensorMap s[2];
};

// Device workspace, zeroed once at allocation; y13 and y are re-zeroed by
// their last reader so every call finds them clean.
struct Workspace {
  Expert lists[2][MAX_LIST];  // hot, cold
  int counts[2];
  int live[MAX_ROUTES];  // route runs an expert on this GPU
  float xs13[MAX_TOKENS];
  float xs2[MAX_ROUTES];
  alignas(128)
      __half x13[MAX_TOKENS * HIDDEN];  // TMA sources: 16 B aligned at least
  alignas(128) __half x2[MAX_ROUTES * INTER];
  alignas(128) float y13[MAX_ROUTES * 2 * INTER];
  alignas(128) float y[MAX_TOKENS * HIDDEN];
  int done13[2][MAX_LIST];  // w13 chunks flushed per entry, zeroed by route_prep
  int ready[2][MAX_LIST];   // epoch once the entry's activation rows are written
  int epoch;                // bumped by route_prep every call
  alignas(128) __nv_bfloat16 x13b[MAX_TOKENS * HIDDEN];  // shared expert input
  alignas(128) __nv_bfloat16 x2s[MAX_TOKENS * INTER];    // its activation
  alignas(128) float y13s[MAX_TOKENS * 2 * INTER];
  int next[2];  // per tier: next group to claim (hot, cold), zeroed by route_prep
  int done_s;   // shared w13 chunks flushed
  int ready_s;  // epoch once x2s is written
  int T;
};
static_assert(offsetof(Workspace, x13) % 16 == 0 &&
                  offsetof(Workspace, x2) % 16 == 0,
              "TMA alignment");

struct Params {
  Tier tier[2];
  CUtensorMap sw[2];  // shared expert w13 [2 * INTER][HIDDEN], w2 [HIDDEN][INTER]
  Workspace* ws;
  float shared_scale;
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
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(
                   smem_u32(bar)),
               "r"(bytes)
               : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(bar))
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
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], "
      "[%1], %2, [%3];" ::"r"(smem_u32(dst)),
      "l"(src), "r"(bytes), "r"(smem_u32(bar))
      : "memory");
}
__device__ __forceinline__ void tma_3d(void* dst, const CUtensorMap* map,
                                       int c0, int c1, int c2, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::"
      "bytes [%0], [%1, {%2, %3, %4}], [%5];" ::"r"(smem_u32(dst)),
      "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(c2),
      "r"(smem_u32(bar))
      : "memory");
}
__device__ __forceinline__ void tma_2d(void* dst, const CUtensorMap* map,
                                       int c0, int c1, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::"
      "bytes [%0], [%1, {%2, %3}], [%4];" ::"r"(smem_u32(dst)),
      "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(smem_u32(bar))
      : "memory");
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
  asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(d) : "r"(a), "r"(mask), "r"(magic));
  return d;  // (a & mask) | magic
}
__device__ __forceinline__ void decode_int4_fast(uint32_t w, uint32_t* a) {
  const uint32_t magic = 0x64006400u;
  const uint32_t w8 = w >> 8;
  const uint32_t lo0 = lop3_and_or(w, 0x000F000Fu, magic);   // rows g,   k0 k1
  const uint32_t lo1 = lop3_and_or(w8, 0x000F000Fu, magic);  // rows g+8, k0 k1
  const uint32_t hi0 = lop3_and_or(w, 0x00F000F0u, magic);   // rows g,   k8 k9
  const uint32_t hi1 = lop3_and_or(w8, 0x00F000F0u, magic);  // rows g+8, k8 k9
  const __half2 unit = __halves2half2(__ushort_as_half(0x0400), __ushort_as_half(0x0400));
  const __half2 bias = __halves2half2(__ushort_as_half(0xAC08), __ushort_as_half(0xAC08));
  // (1024 + 16 c) * 2^-18 - 72 * 2^-14 = (c - 8) * 2^-14
  const __half2 unit16 = __halves2half2(__ushort_as_half(0x0040), __ushort_as_half(0x0040));
  const __half2 bias16 = __halves2half2(__ushort_as_half(0x9C80), __ushort_as_half(0x9C80));
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
#ifdef TD_CTA_TRACE
// Probe build only: one record per gemm CTA {phase, block, SM; entry, ready
// (past its PDL wait), exit} in %globaltimer ns, plus stamps from td_stamp.
// Phases: 0/1 w13/w2 CTA, 10/11 CTA without units; 50 route_prep, 53/63 act
// (live/unrouted route), 54 finalize {entry, past PDL wait, exit}; 60/61
// producer {first issue, past PDL wait (w2), last issue}; 70/71 consumer warp 0
// {first stage full, last stage full, exit}.
__device__ unsigned long long td_trace[1 << 18][4];
__device__ unsigned int td_trace_n;
__device__ __forceinline__ uint64_t td_now() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
// word 0: phase | block << 8 | SM << 16 | n_hot << 32 | n_cold << 48
__device__ __forceinline__ void td_record(int ph, uint64_t t0, uint64_t t1,
                                          int n_hot, int n_cold) {
  uint32_t sm;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
  const unsigned i = atomicAdd(&td_trace_n, 1u) & ((1u << 18) - 1);
  td_trace[i][0] = uint64_t(ph & 0xFF) | (uint64_t(blockIdx.x & 0xFF) << 8) |
                   (uint64_t(sm & 0xFFFF) << 16) |
                   (uint64_t(n_hot & 0xFFFF) << 32) |
                   (uint64_t(n_cold & 0xFFFF) << 48);
  td_trace[i][1] = t0;
  td_trace[i][2] = t1;
  td_trace[i][3] = td_now();
}
__global__ void td_stamp_kernel(int tag) {
  const uint64_t t = td_now();
  td_record(100 + tag, t, t, 0, 0);
}
  #define TD_T0 const uint64_t td_t0 = td_now();
  #define TD_T1(v) v = td_now();
  #define TD_REC(ph, t1) td_record(ph, td_t0, t1, n_hot, n_cold);
  // small kernels: one record per block {entry, past PDL wait, exit}
  #define TD_K0                      \
    const uint64_t td_k0 = td_now(); \
    uint64_t td_k1 = 0;
  #define TD_K1 td_k1 = td_now();
  #define TD_KREC(ph) \
    if (threadIdx.x == 0) td_record(ph, td_k0, td_k1, 0, 0);
  #define TD_V(...) __VA_ARGS__
  #define TD_PREP_BOUNDS __launch_bounds__(PREP_THREADS)
#else
  #define TD_T0
  #define TD_T1(v)
  #define TD_REC(ph, t1)
  #define TD_K0
  #define TD_K1
  #define TD_KREC(ph)
  #define TD_V(...)
  #define TD_PREP_BOUNDS
#endif
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

// ---------------------------------------------------------------- 1: route +
// prep Blocks [0, T): token rows -> f16 fragments. Block T: which GPU runs
// each active expert (below), then this GPU's per-tier expert lists.
//
// Replica assignment. A hot expert runs on its primary GPU; an active cold
// expert with a replica may run on either holder, and every GPU derives the
// same choice from the same router output. The choice balances predicted
// layer time: COST_US is this path's measured graph-replay time per layer
// (GH200, MiMo-V2 shapes) with h hot and c cold experts on a GPU, made
// non-decreasing. Path reversal moves one cold expert at a time off the
// slowest GPU, along at most three replica hops (each hop is a different GPU
// pair, so the flexible experts are tracked as counts per pair), to the end
// that stays fastest, while that end stays below the source's time.
constexpr int EP = 4, PAIRS = EP * (EP - 1) / 2, MAX_EXPERTS = 512,
              MAX_REVERSALS = 64, COST_HOT = 25, COST_COLD = 7;
__constant__ unsigned short COST_US[COST_HOT][COST_COLD] = {
    {0, 62, 111, 161, 226, 269, 305},    {20, 64, 113, 163, 228, 271, 307},
    {30, 65, 115, 165, 229, 272, 308},   {39, 67, 115, 165, 229, 273, 308},
    {45, 67, 115, 165, 230, 273, 309},   {52, 67, 115, 165, 230, 273, 309},
    {58, 69, 116, 165, 230, 273, 309},   {66, 74, 116, 166, 230, 273, 309},
    {73, 84, 116, 166, 231, 273, 309},   {82, 90, 117, 166, 231, 273, 309},
    {87, 102, 118, 166, 231, 273, 309},  {92, 112, 126, 167, 231, 274, 309},
    {100, 120, 128, 167, 231, 274, 309}, {108, 128, 136, 169, 231, 274, 309},
    {116, 139, 151, 169, 231, 274, 309}, {126, 147, 161, 175, 231, 274, 309},
    {136, 155, 171, 180, 232, 275, 309}, {142, 159, 180, 187, 232, 275, 309},
    {147, 164, 189, 193, 233, 275, 309}, {156, 173, 198, 204, 233, 276, 310},
    {164, 181, 207, 215, 233, 277, 310}, {172, 192, 217, 225, 236, 277, 311},
    {180, 203, 226, 235, 239, 278, 312}, {188, 213, 236, 245, 245, 279, 313},
    {196, 224, 246, 254, 254, 279, 314},
};

struct Placement {
  const int* primary;      // [E] GPU holding the expert's own copy
  const int* secondary;    // [E] GPU holding a cold replica, or -1
  const int* primary_hot;  // [E] 1 when the own copy is in the hot tier
  int num_experts;         // 0: the slot maps already say what runs here
  int ep_rank;
  bool schedule;  // false: every expert runs on its primary
};

__device__ __forceinline__ int pair_index(int lo, int hi) {
  return lo * EP - lo * (lo + 1) / 2 + (hi - lo - 1);
}

__device__ __forceinline__ int cost(const unsigned short (*tab)[COST_COLD],
                                    int h, int c) {
  return tab[min(h, COST_HOT - 1)][min(c, COST_COLD - 1)] +
         8 * max(h - (COST_HOT - 1), 0) + 40 * max(c - (COST_COLD - 1), 0);
}

// One warp. at_low[k] of the total[k] flexible experts of pair k sit on the
// pair's lower GPU; on return (lane 0) at_low holds the balanced split. Each
// iteration, lane l < 15 tries the l-th path from the slowest GPU in the
// reference order (length, then ascending GPU ids): its target, and the pairs
// it needs an edge on the lower (need_lo) or upper (need_hi) GPU of.
__device__ void reverse_paths(const unsigned short (*tab)[COST_COLD],
                              const int* hot, const int* fixed,
                              const int* total, int* at_low) {
  const int lane = threadIdx.x % 32;
  int split[PAIRS], tot[PAIRS];
#pragma unroll
  for (int k = 0; k < PAIRS; ++k) {
    split[k] = at_low[k];
    tot[k] = total[k];
  }
  int h[EP], f[EP];
#pragma unroll
  for (int r = 0; r < EP; ++r) {
    h[r] = hot[r];
    f[r] = fixed[r];
  }
  for (int step = 0; step < MAX_REVERSALS; ++step) {
    int now[EP], after[EP];
#pragma unroll
    for (int r = 0; r < EP; ++r) {
      int c = f[r];
#pragma unroll
      for (int o = 0; o < EP; ++o)
        if (o != r) {
          const int k = pair_index(min(r, o), max(r, o));
          c += r < o ? split[k] : tot[k] - split[k];
        }
      now[r] = cost(tab, h[r], c);
      after[r] = cost(tab, h[r], c + 1);
    }
    int src = 0;
#pragma unroll
    for (int r = 1; r < EP; ++r)
      if (now[r] > now[src]) src = r;
    // this lane's path; the i-th GPU other than src, ascending, is other(i)
    auto other = [src](int i) { return i < src ? i : i + 1; };
    int hop1 = -1, hop2 = -1, hop3 = -1, len = 0;
    if (lane < 3) {
      len = 1;
      hop1 = other(lane);
    } else if (lane < 15) {
      const int m = lane < 9 ? lane - 3 : lane - 9;
      const int first = m / 2, second = m % 2 < first ? m % 2 : m % 2 + 1;
      len = lane < 9 ? 2 : 3;
      hop1 = other(first);
      hop2 = other(second);
      hop3 = other(3 - first - second);
    }
    const int target = len == 1 ? hop1 : len == 2 ? hop2 : hop3;
    unsigned need_lo = 0, need_hi = 0;
    auto need = [&](int u, int v) {
      const unsigned bit = 1u << pair_index(min(u, v), max(u, v));
      if (u < v)
        need_lo |= bit;
      else
        need_hi |= bit;
    };
    if (len >= 1) need(src, hop1);
    if (len >= 2) need(hop1, hop2);
    if (len >= 3) need(hop2, hop3);
    unsigned have_lo = 0, have_hi = 0;
#pragma unroll
    for (int k = 0; k < PAIRS; ++k) {
      have_lo |= (split[k] > 0 ? 1u : 0u) << k;
      have_hi |= (tot[k] - split[k] > 0 ? 1u : 0u) << k;
    }
    int end_v = 0, src_v = 0;
#pragma unroll
    for (int r = 0; r < EP; ++r) {
      if (r == target) end_v = after[r];
      if (r == src) src_v = now[r];
    }
    const bool usable = len > 0 && (need_lo & ~have_lo) == 0 &&
                        (need_hi & ~have_hi) == 0 && end_v < src_v;
    const unsigned key =
        usable ? static_cast<unsigned>(end_v) << 5 | lane : 0xffffffffu;
    const unsigned best = __reduce_min_sync(0xffffffffu, key);
    if (best == 0xffffffffu) break;
    const int from = best & 31;
    need_lo = __shfl_sync(0xffffffffu, need_lo, from);
    need_hi = __shfl_sync(0xffffffffu, need_hi, from);
#pragma unroll
    for (int k = 0; k < PAIRS; ++k)
      split[k] += ((need_hi >> k) & 1) - ((need_lo >> k) & 1);
  }
  if (lane == 0)
#pragma unroll
    for (int k = 0; k < PAIRS; ++k) at_low[k] = split[k];
}

template <typename IdT>
__global__ void TD_PREP_BOUNDS
route_prep_kernel(Workspace* ws, const __nv_bfloat16* x, const IdT* topk_ids,
                  const bool* padding, const float* topk_weights,
                  const int* hot_map, const int* cold_map, Placement pl,
                  int num_tokens, int hot_size, int cold_size) {
  // [hot_size + cold_size] each: tier slot -> its routes here, its first entry
  extern __shared__ int n_of[];
  int* base_of = n_of + hot_size + cold_size;
  __shared__ float red[32];
  TD_K0
  pdl_wait();
  TD_K1
  pdl_release();
  if (blockIdx.x < num_tokens) {
    // every element loads before any is used: these small kernels are latency
    // bound
    constexpr int PER = HIDDEN / PREP_THREADS;
    const int t = blockIdx.x;
    const __nv_bfloat16* __restrict__ xr = x + static_cast<size_t>(t) * HIDDEN;
    float v[PER], m = 0.f;
#pragma unroll
    for (int i = 0; i < PER; ++i)
      v[i] = __bfloat162float(xr[threadIdx.x + i * PREP_THREADS]);
#pragma unroll
    for (int i = 0; i < PER; ++i) m = fmaxf(m, fabsf(v[i]));
    const float inv = row_scale(block_max(m, red), &ws->xs13[t]);
    __half* __restrict__ out = ws->x13 + static_cast<size_t>(t) * HIDDEN;
#pragma unroll
    for (int i = 0; i < PER; ++i)
      out[frag_slot(threadIdx.x + i * PREP_THREADS)] =
          __float2half_rn(v[i] * inv);
    __nv_bfloat16* __restrict__ outb = ws->x13b + static_cast<size_t>(t) * HIDDEN;
#pragma unroll
    for (int i = 0; i < PER; ++i)
      outb[frag_slot(threadIdx.x + i * PREP_THREADS)] =
          xr[threadIdx.x + i * PREP_THREADS];
    TD_KREC(50)
    return;
  }
  __shared__ int count[2];
  for (int i = threadIdx.x; i < 2 * MAX_LIST; i += blockDim.x)
    (&ws->done13[0][0])[i] = 0;
  // per expert: 1 when routed, then (tier << 16 | slot) where it runs here
  __shared__ int state[MAX_EXPERTS];
  __shared__ int hot_n[EP], fixed_n[EP], total[PAIRS], at_low[PAIRS];
  __shared__ int warp_n[PREP_THREADS / 32][PAIRS];
  __shared__ unsigned short tab[COST_HOT][COST_COLD];
  const int E = pl.num_experts, routes = num_tokens * TOPK, r = threadIdx.x;
  for (int i = threadIdx.x; i < hot_size + cold_size; i += blockDim.x)
    n_of[i] = 0;
  if (threadIdx.x < 2) count[threadIdx.x] = 0;
  if (r < MAX_ROUTES) ws->live[r] = 0;
  // every global read is issued up front: this block is a latency chain
  int e = -1;
  float wt = 0.f;
  if (r < routes) {
    e = static_cast<int>(topk_ids[r]);
    // a padded token's routes are dropped, as if its ids were -1
    if (padding != nullptr && padding[r / TOPK]) e = -1;
    wt = topk_weights[r];
    if (e >= 0 && E > 0 && e >= E) e = -1;
  }
  int tier = -1, local = -1, slot = -1;
  if (E > 0) {
    const int x_e = threadIdx.x;
    int p = -1, s = -1, hm = -1, cm = -1;
    bool hot = false;
    if (x_e < E) {
      p = pl.primary[x_e];
      hot = pl.primary_hot[x_e] != 0;
      s = pl.secondary[x_e];
      hm = hot_map[x_e];
      cm = cold_map[x_e];
      state[x_e] = 0;
    }
    if (x_e < EP) hot_n[x_e] = fixed_n[x_e] = 0;
    if (x_e < PAIRS) total[x_e] = at_low[x_e] = 0;
    if (x_e < COST_HOT * COST_COLD)
      tab[x_e / COST_COLD][x_e % COST_COLD] =
          COST_US[x_e / COST_COLD][x_e % COST_COLD];
    __syncthreads();
    if (e >= 0) state[e] = 1;
    __syncthreads();
    int k = -1;
    if (x_e < E && state[x_e]) {
      if (hot)
        atomicAdd(&hot_n[p], 1);
      else if (s < 0)
        atomicAdd(&fixed_n[p], 1);
      else {
        k = pair_index(min(p, s), max(p, s));
        atomicAdd(&total[k], 1);
        if (p < s) atomicAdd(&at_low[k], 1);
      }
    } else {
      p = -1;
    }
    // a flexible expert's position among its pair's, in ascending expert id
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    int pos = 0;
#pragma unroll
    for (int j = 0; j < PAIRS; ++j) {
      const unsigned ballot = __ballot_sync(0xffffffffu, k == j);
      if (k == j) pos = __popc(ballot & ((1u << lane) - 1));
      if (lane == 0) warp_n[warp][j] = __popc(ballot);
    }
    __syncthreads();
    if (k >= 0)
      for (int w = 0; w < warp; ++w) pos += warp_n[w][k];
    if (threadIdx.x < 32 && pl.schedule)
      reverse_paths(tab, hot_n, fixed_n, total, at_low);
    __syncthreads();
    if (p >= 0) {
      const int runs_on = k < 0 ? p : pos < at_low[k] ? min(p, s) : max(p, s);
      int code = -1;
      if (runs_on == pl.ep_rank && hot && hm >= 0)
        code = hm;
      else if (runs_on == pl.ep_rank && !hot && cm >= 0)
        code = 1 << 16 | cm;
      state[x_e] = code;
    }
    __syncthreads();
    if (e >= 0 && state[e] >= 0) {
      tier = state[e] >> 16;
      local = state[e] & 0xffff;
    }
  } else {
    __syncthreads();
    if (e >= 0) {
      const int h = hot_map[e], c = cold_map[e];
      if (h >= 0) {
        tier = 0;
        local = h;
      } else if (c >= 0) {
        tier = 1;
        local = c;
      }
    }
  }
  // each route's position among its expert's; the first route then opens
  // ceil(n / MAX_TOK) consecutive entries, MAX_TOK routes each
  const int idx = tier * hot_size + local;
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
    slot = base_of[idx] + pos / MAX_TOK;
    Expert& ex = ws->lists[tier][slot];
    const int i = pos % MAX_TOK;
    ex.tok[i] = r / TOPK;
    ex.route[i] = r;
    ex.wt[i] = wt;
    ws->live[r] = 1;
  }
  __syncthreads();
  if (threadIdx.x < 2) ws->counts[threadIdx.x] = count[threadIdx.x];
  if (threadIdx.x == 0) {
    ws->done_s = 0;
    ws->next[0] = ws->next[1] = 0;
    ws->T = num_tokens;
    ws->epoch += 1;
  }
  TD_KREC(50)
}

// ---------------------------------------------------------------- 2: the
// layer, one persistent CTA per SM. CTAs [0, cold_ctas) run the cold tier,
// the rest the hot tier. Each CTA runs its share of its tier's w13 units, then
// its share of the w2 units. An entry's w13 chunks are counted as they flush;
// the CTA whose flush completes an entry runs silu * up for its routes and
// releases ready[entry]. A w2 unit's producer issues the weights first and
// waits on ready only before loading the activation rows.
__device__ __forceinline__ int cold_ctas_for(int n_hot, int n_cold) {
  if (n_cold == 0) return 0;
  return n_hot == 0 ? GRID : TD_COLD_CTAS;
}
__device__ __forceinline__ int ld_acquire(const int* p) {
  int v;
  asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_release(int* p, int v) {
  asm volatile("st.release.gpu.global.b32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ void fence_proxy_async() {
  asm volatile("fence.proxy.async.global;" ::: "memory");
}
__device__ __forceinline__ void consumer_sync() {
  asm volatile("bar.sync 1, %0;" ::"n"(CONSUMER_WARPS * 32) : "memory");
}

// silu(gate) * up for one entry's routes, one warp per route: f16 fragments
// of x2 with the row's power-of-two scale; the y13 rows are zeroed for the
// next call.
__device__ __forceinline__ void activate(Workspace* ws, const Expert& e) {
  __shared__ float red[MAX_TOK][CONSUMER_WARPS];
  static_assert(INTER == CONSUMER_WARPS * 32 || INTER == 2 * CONSUMER_WARPS * 32, "elements per thread");
  constexpr int PER = INTER / (CONSUMER_WARPS * 32);
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, n = e.ntok;
  float a[MAX_TOK][PER];
#pragma unroll
  for (int i = 0; i < MAX_TOK; ++i)
    if (i < n) {
      const float* yr = ws->y13 + static_cast<size_t>(e.route[i]) * 2 * INTER;
#pragma unroll
      for (int q = 0; q < PER; ++q) {
        const int k = threadIdx.x + q * CONSUMER_WARPS * 32;
        const float g = __ldcg(yr + k), u = __ldcg(yr + INTER + k);
        a[i][q] = __fdividef(g, 1.f + __expf(-g)) * u;
      }
    }
#pragma unroll
  for (int i = 0; i < MAX_TOK; ++i)
    if (i < n) {
      float m = 0.f;
#pragma unroll
      for (int q = 0; q < PER; ++q) m = fmaxf(m, fabsf(a[i][q]));
      for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
      if (lane == 0) red[i][warp] = m;
    }
  consumer_sync();
#pragma unroll
  for (int i = 0; i < MAX_TOK; ++i)
    if (i < n) {
      float m = 0.f;
#pragma unroll
      for (int w = 0; w < CONSUMER_WARPS; ++w) m = fmaxf(m, red[i][w]);
      float scale;
      const float inv = row_scale(m, &scale);
      const int r = e.route[i];
      if (threadIdx.x == 0) ws->xs2[r] = scale;
      float* yr = ws->y13 + static_cast<size_t>(r) * 2 * INTER;
      __half* out = ws->x2 + static_cast<size_t>(r) * INTER;
#pragma unroll
      for (int q = 0; q < PER; ++q) {
        const int k = threadIdx.x + q * CONSUMER_WARPS * 32;
        out[frag_slot(k)] = __float2half_rn(a[i][q] * inv);
        yr[k] = 0.f;
        yr[INTER + k] = 0.f;
      }
    }
}

// The shared expert's silu * up for all T tokens (bf16 fragments of x2s), one
// element per consumer thread; y13s is zeroed for the next call.
__device__ __forceinline__ void activate_shared(Workspace* ws, int T) {
  constexpr int PER = INTER / (CONSUMER_WARPS * 32);
  for (int tok = 0; tok < T; ++tok) {
    float* __restrict__ yr = ws->y13s + static_cast<size_t>(tok) * 2 * INTER;
    __nv_bfloat16* __restrict__ out = ws->x2s + static_cast<size_t>(tok) * INTER;
#pragma unroll
    for (int q = 0; q < PER; ++q) {
      const int k = threadIdx.x + q * CONSUMER_WARPS * 32;
      const float g = __ldcg(yr + k), u = __ldcg(yr + INTER + k);
      out[frag_slot(k)] = __float2bfloat16_rn(__fdividef(g, 1.f + __expf(-g)) * u);
      yr[k] = 0.f;
      yr[INTER + k] = 0.f;
    }
  }
}

// One routed unit's 16 group steps for this warp. PH 0 (w13): 4 row blocks x
// 2 K halves of a 64-row x 1024 box; PH 1 (w2): 8 row blocks of a 128-row x
// 512 box. All smem offsets are immediates off two per-warp bases.
template <int PH>
__device__ __forceinline__ void consume_routed(const unsigned char* st, int warp,
                                               int lane, float* acc) {
  constexpr int KROW = PH ? 1024 : 512;  // bytes per k16 row of the box
  constexpr int SROW = PH ? 256 : 128;   // bytes per scale group row
  const int g = lane / 4, tq = lane % 4;
  const int mb = PH ? warp % 8 : warp % 4, kp = PH ? warp / 8 : warp / 4;
  const int wrow = PH ? (mb >> 2) * 512 + 4 * (mb & 3) : 4 * mb;
  const int sblk = PH ? (mb >> 2) * 128 + 4 * (mb & 3) : 4 * mb;
  const unsigned char* wb = st + (kp * STEPS * 2) * KROW + lane * 16 + wrow;
  const unsigned char* sb = st + W_BYTES + (kp * STEPS) * SROW + 16 * g + sblk;
  const unsigned char* xb = st + W_BYTES + S_BYTES + g * XROW_STRIDE + (kp * STEPS * 4 + tq) * 16;
#pragma unroll
  for (int j = 0; j < STEPS; ++j) {
    const uint32_t sw = *reinterpret_cast<const uint32_t*>(sb + j * SROW);
    const uint32_t w0 = *reinterpret_cast<const uint32_t*>(wb + (2 * j) * KROW);
    const uint32_t w1 = *reinterpret_cast<const uint32_t*>(wb + (2 * j + 1) * KROW);
    const uint4 xv = *reinterpret_cast<const uint4*>(xb + j * 64);
    const float zero[4] = {0.f, 0.f, 0.f, 0.f};
    float d[4];
    uint32_t a[4];
    decode_int4_fast(w0, a);
    mma_f16(d, a, xv.x, xv.y, zero);
    decode_int4_fast(w1, a);
    mma_f16(d, a, xv.z, xv.w, d);
    const float s0 = __uint_as_float(sw << 16), s1 = __uint_as_float(sw & 0xFFFF0000u);
    acc[0] = fmaf(s0, d[0], acc[0]);
    acc[1] = fmaf(s0, d[1], acc[1]);
    acc[2] = fmaf(s1, d[2], acc[2]);
    acc[3] = fmaf(s1, d[3], acc[3]);
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
constexpr int SCH0 = 12;                        // shared w13 chunks per group
constexpr int SG0 = CHUNKS_S0 / SCH0;           // groups per shared w13 tile
static_assert(CHUNKS_S0 % SCH0 == 0, "shared w13 groups tile K");
struct Group {
  int kind, x, c0, nch;  // x: R0 entry * TILES0 + tile, R1 entry * TILES1 + unit, S tile
};
__device__ __forceinline__ int queue_len(int n, bool sh) {
  return (sh ? TILES_S0 * SG0 + TILES_S1 : 0) + n * (TILES0 + TILES1);
}
__device__ __forceinline__ Group group_at(int gi, int n, bool sh) {
  const int ns0 = sh ? TILES_S0 * SG0 : 0, nr0 = n * TILES0, ns1 = sh ? TILES_S1 : 0;
  if (gi < ns0) return {K_S0, gi / SG0, (gi % SG0) * SCH0, SCH0};
  gi -= ns0;
  if (gi < nr0) return {K_R0, gi, 0, CHUNKS0};
  gi -= nr0;
  if (gi < ns1) return {K_S1, gi, 0, CHUNKS_S1};
  gi -= ns1;
  return {K_R1, gi, 0, 1};
}

__global__ void __launch_bounds__(THREADS, 1)
    layer_kernel(const __grid_constant__ Params p) {
  extern __shared__ __align__(128) unsigned char smem[];
  uint64_t* full = reinterpret_cast<uint64_t*>(smem);
  uint64_t* empty = full + STAGES;
  // 1 KB aligned (128 B swizzle) by an offset into smem, so every load off it
  // stays a shared-space LDS rather than a generic LD
  unsigned char* ring =
      smem + SMEM_HEAD + ((1024u - ((smem_u32(smem) + SMEM_HEAD) & 1023u)) & 1023u);
  __shared__ int4 desc[STAGES];
  // per stage, routed units: each token's destination row (w13: its route's
  // y13 row; w2: its token's y row) and the scale its flush applies
  // (w13: xs13[token]; w2: weight * xs2[route]), written by the producer
  __shared__ int sd_row[STAGES][MAX_TOK];
  __shared__ float sd_f[STAGES][MAX_TOK];
  __shared__ int s_last;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  Workspace* ws = p.ws;
  TD_T0
  uint64_t td_t1 = 0;
  (void)td_t1;
  if (threadIdx.x == 0) {
    for (int s = 0; s < STAGES; ++s) {
      mbar_init(&full[s], 1);
      mbar_init(&empty[s], CONSUMER_WARPS);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  pdl_wait();  // route_prep's lists, rows and counters
  pdl_release();
  TD_T1(td_t1)
  const int n_tier[2] = {ws->counts[0], ws->counts[1]};
  const int epoch = ws->epoch, T = ws->T;
  const bool sh = p.has_shared;
  const int len[2] = {queue_len(n_tier[0], sh), queue_len(n_tier[1], false)};
  const int cold_ctas = len[1] == 0 ? 0 : len[0] == 0 ? GRID : TD_COLD_CTAS;
  const int own = static_cast<int>(blockIdx.x) < cold_ctas ? 1 : 0;

  if (warp == CONSUMER_WARPS) {  // producer warp
    if (lane == 0) {
      for (int q = 0; q < 2; ++q)
        for (int k = 0; k < 2; ++k) {
          asm volatile("prefetch.tensormap [%0];" ::"l"(reinterpret_cast<uint64_t>(&p.tier[q].w[k])) : "memory");
          asm volatile("prefetch.tensormap [%0];" ::"l"(reinterpret_cast<uint64_t>(&p.tier[q].s[k])) : "memory");
        }
      if (sh)
        for (int k = 0; k < 2; ++k)
          asm volatile("prefetch.tensormap [%0];" ::"l"(reinterpret_cast<uint64_t>(&p.sw[k])) : "memory");
    }
    TD_V(const uint64_t td_p0 = td_now(); uint64_t td_spin = 0;)
    int q = own, last_ready = -1, it = 0;
    bool stolen = false;
    while (true) {
      int gi = 0;
      if (lane == 0) gi = atomicAdd(&ws->next[q], 1);
      gi = __shfl_sync(0xffffffffu, gi, 0);
      if (gi >= len[q]) {
        if (stolen) break;
        stolen = true;
        q ^= 1;
        last_ready = -1;
        continue;
      }
      const Group gr = group_at(gi, n_tier[q], q == 0 && sh);
      const Tier& tr = p.tier[q];
      const Expert* experts = ws->lists[q];
      for (int ci = 0; ci < gr.nch; ++ci, ++it) {
        const int s = it % STAGES, c = gr.c0 + ci;
        unsigned char* dst = ring + static_cast<size_t>(s) * STAGE_BYTES;
        if (lane == 0 && it >= STAGES) mbar_wait(&empty[s], ((it / STAGES) - 1) & 1);
        __syncwarp();
        const int hdr = gr.kind | q << 4 | (ci == 0) << 8 | (ci == gr.nch - 1) << 9 | gr.nch << 16;
        if (gr.kind == K_R0) {
          const int ei = gr.x / TILES0, t = gr.x - ei * TILES0;
          const Expert& e = experts[ei];
          if (lane < e.ntok) {
            sd_row[s][lane] = e.route[lane];
            sd_f[s][lane] = ws->xs13[e.tok[lane]];
          }
          __syncwarp();
          if (lane == 0) {
            desc[s] = make_int4(hdr, gr.x, c, e.ntok);
            mbar_expect_tx(&full[s], W_BYTES + S_BYTES + e.ntok * XROW_BYTES0);
            tma_3d(dst, &tr.w[0], t * 128, c * KT0, e.local, &full[s]);
            tma_3d(dst + W_BYTES, &tr.s[0], t * 64, c * G0, e.local, &full[s]);
          }
          __syncwarp();
          if (lane < e.ntok)
            bulk_g2s(dst + W_BYTES + S_BYTES + lane * XROW_STRIDE,
                     ws->x13 + static_cast<size_t>(e.tok[lane]) * HIDDEN + c * CK0,
                     XROW_BYTES0, &full[s]);
        } else if (gr.kind == K_R1) {
          const int ei = gr.x / TILES1, t = gr.x - ei * TILES1;
          const Expert& e = experts[ei];
          if (lane == 0) {
            tma_3d(dst, &tr.w[1], t * 256, 0, e.local, &full[s]);  // weights first
            tma_3d(dst + W_BYTES, &tr.s[1], t * 128, 0, e.local, &full[s]);
            if (ei != last_ready) {
              TD_V(const uint64_t w0 = td_now();)
              while (ld_acquire(&ws->ready[q][ei]) != epoch) __nanosleep(32);
              TD_V(td_spin += td_now() - w0;)
              fence_proxy_async();
            }
          }
          last_ready = ei;
          __syncwarp();
          if (lane < e.ntok) {
            sd_row[s][lane] = e.tok[lane];
            sd_f[s][lane] = e.wt[lane] * __ldcg(&ws->xs2[e.route[lane]]);
          }
          __syncwarp();
          if (lane == 0) {
            desc[s] = make_int4(hdr, gr.x, c, e.ntok);
            mbar_expect_tx(&full[s], W_BYTES + S_BYTES + e.ntok * XROW_BYTES1);
          }
          __syncwarp();
          if (lane < e.ntok)
            bulk_g2s(dst + W_BYTES + S_BYTES + lane * XROW_STRIDE,
                     ws->x2 + static_cast<size_t>(e.route[lane]) * INTER,
                     XROW_BYTES1, &full[s]);
        } else {  // shared expert
          if (lane == 0) {
            desc[s] = make_int4(hdr, gr.x, c, 0);
            mbar_expect_tx(&full[s], W_BYTES + T * XS_BYTES);
            tma_2d(dst, &p.sw[gr.kind == K_S0 ? 0 : 1], c * CKS, gr.x * RS, &full[s]);
            if (gr.kind == K_S1 && last_ready != -2) {
              TD_V(const uint64_t w0 = td_now();)
              while (ld_acquire(&ws->ready_s) != epoch) __nanosleep(32);
              TD_V(td_spin += td_now() - w0;)
              fence_proxy_async();
            }
          }
          if (gr.kind == K_S1) last_ready = -2;
          __syncwarp();
          if (lane < T)
            bulk_g2s(dst + W_BYTES + lane * XS_BYTES,
                     (gr.kind == K_S0 ? ws->x13b + static_cast<size_t>(lane) * HIDDEN
                                      : ws->x2s + static_cast<size_t>(lane) * INTER) + c * CKS,
                     XS_BYTES, &full[s]);
        }
      }
    }
    if (lane == 0) {  // end of work
      const int s = it % STAGES;
      if (it >= STAGES) mbar_wait(&empty[s], ((it / STAGES) - 1) & 1);
      desc[s] = make_int4(K_END, 0, 0, 0);
      mbar_arrive(&full[s]);
    }
    TD_V(if (lane == 0) td_record(60 + own, td_p0, td_p0 + td_spin, n_tier[0], n_tier[1]);)
    return;
  }

  // consumer warps
  const int g = lane / 4, tq = lane % 4;
  const int nt_n = (T + 7) / 8;
  float acc[4] = {};
  float accs[SRB][4][4] = {};
  int acq = -1;  // (tier, entry) whose ready this warp has acquired
  TD_V(uint64_t td_c0 = 0;)
  for (int it = 0;; ++it) {
    const int s = it % STAGES;
    mbar_wait(&full[s], (it / STAGES) & 1);
    const int4 d = desc[s];
    const int kind = d.x & 15;
    if (kind == K_END) break;
    const int q = (d.x >> 4) & 1, last = (d.x >> 9) & 1, nch = d.x >> 16;
    const unsigned char* st = ring + static_cast<size_t>(s) * STAGE_BYTES;
    const Expert* experts = ws->lists[q];
    // this lane's two tokens of a routed unit: destination rows and scales
    const int ntok = d.w;
    int rw[2] = {0, 0};
    float fs[2] = {0.f, 0.f};
    if (kind == K_R0 || kind == K_R1) {
#pragma unroll
      for (int h = 0; h < 2; ++h)
        if (2 * tq + h < ntok) {
          rw[h] = sd_row[s][2 * tq + h];
          fs[h] = sd_f[s][2 * tq + h];
        }
    }
    if (kind == K_R0) {
      consume_routed<0>(st, warp, lane, acc);
      __syncwarp();
      if (lane == 0) mbar_arrive(&empty[s]);
      if (last) {
        const int ei = d.y / TILES0, t = d.y - ei * TILES0;
        const Expert& e = experts[ei];
        const int n0r = t * R0 + (warp % 4) * 16 + g;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          if (2 * tq + (i & 1) < ntok)
            atomicAdd(&ws->y13[static_cast<size_t>(rw[i & 1]) * 2 * INTER + n0r + (i >> 1) * 8],
                      acc[i] * fs[i & 1]);
          acc[i] = 0.f;
        }
        consumer_sync();
        if (threadIdx.x == 0) {
          __threadfence();
          s_last = atomicAdd(&ws->done13[q][ei], nch) + nch == UNITS0;
        }
        consumer_sync();
        if (s_last) {
          __threadfence();
          activate(ws, e);
          fence_proxy_async();
          __threadfence();
          consumer_sync();
          if (threadIdx.x == 0) st_release(&ws->ready[q][ei], epoch);
        }
        TD_V(td_c0 = td_now();)
      }
    } else if (kind == K_R1) {
      consume_routed<1>(st, warp, lane, acc);
      __syncwarp();
      if (lane == 0) mbar_arrive(&empty[s]);
      const int t = d.y % TILES1;
      const int n0r = t * R1 + (warp % 8) * 16 + g;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        if (2 * tq + (i & 1) < ntok)
          atomicAdd(&ws->y[static_cast<size_t>(rw[i & 1]) * HIDDEN + n0r + (i >> 1) * 8],
                    acc[i] * fs[i & 1]);
        acc[i] = 0.f;
      }
    } else {  // shared expert: 32 rows per warp, 4 k16 steps, nt_n token tiles
      const unsigned char* xa = st + W_BYTES;
#pragma unroll
      for (int ks = 0; ks < CKS / 16; ++ks) {
        uint32_t af[SRB][4];
#pragma unroll
        for (int rb = 0; rb < SRB; ++rb) {
          const int row = warp * SROWS + rb * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
          const int ck = ks * 2 + (lane >> 4);
          ldmatrix_x4(af[rb], st + row * 128 + ((ck ^ (row & 7)) << 4));
        }
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
          if (nt < nt_n) {
            const uint2 bv = *reinterpret_cast<const uint2*>(
                xa + (nt * 8 + g) * XS_BYTES + ((ks >> 1) * 4 + tq) * 16 + (ks & 1) * 8);
#pragma unroll
            for (int rb = 0; rb < SRB; ++rb) mma_bf16(accs[rb][nt], af[rb], bv.x, bv.y);
          }
        }
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(&empty[s]);
      if (last) {
        const int t = d.y;
#pragma unroll
        for (int rb = 0; rb < SRB; ++rb)
#pragma unroll
          for (int nt = 0; nt < 4; ++nt)
#pragma unroll
            for (int i = 0; i < 4; ++i) {
              const int tok = nt * 8 + 2 * tq + (i & 1);
              const int row = t * RS + warp * SROWS + rb * 16 + g + (i >> 1) * 8;
              if (nt < nt_n && tok < T) {
                if (kind == K_S0)
                  atomicAdd(&ws->y13s[static_cast<size_t>(tok) * 2 * INTER + row],
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
            s_last = atomicAdd(&ws->done_s, nch) + nch == UNITS_S0;
          }
          consumer_sync();
          if (s_last) {
            __threadfence();
            activate_shared(ws, T);
            fence_proxy_async();
            __threadfence();
            consumer_sync();
            if (threadIdx.x == 0) st_release(&ws->ready_s, epoch);
          }
        }
      }
    }
  }
#ifdef TD_CTA_TRACE
  if (threadIdx.x == 0) {
    td_record(own, td_t1, td_c0, n_tier[0], n_tier[1]);
  }
#endif
}

// ---------------------------------------------------------------- 5: finalize
// One float4 per thread: grid [T][HIDDEN / 1024] x 256.
__global__ void finalize_kernel(Workspace* ws,
                                __nv_bfloat16* __restrict__ out) {
  TD_K0
  pdl_wait();
  TD_K1
  const size_t i = (static_cast<size_t>(blockIdx.y) * HIDDEN +
                    blockIdx.x * 1024 + threadIdx.x * 4);
  float4* y = reinterpret_cast<float4*>(ws->y + i);
  const float4 v = *y;
  *y = make_float4(0.f, 0.f, 0.f, 0.f);
  __nv_bfloat162* o = reinterpret_cast<__nv_bfloat162*>(out + i);
  o[0] = __floats2bfloat162_rn(v.x, v.y);
  o[1] = __floats2bfloat162_rn(v.z, v.w);
  TD_KREC(54)
}

// ---------------------------------------------------------------- host entry
static CUtensorMap make_map(const torch::Tensor& t, CUtensorMapDataType type,
                            int elem_bytes, uint32_t box0, uint32_t box1) {
  // t is [E][rows][cols] contiguous; the map's innermost dimension is cols
  CUtensorMap map;
  const cuuint64_t dims[3] = {static_cast<cuuint64_t>(t.size(2)),
                              static_cast<cuuint64_t>(t.size(1)),
                              static_cast<cuuint64_t>(t.size(0))};
  const cuuint64_t strides[2] = {
      static_cast<cuuint64_t>(t.stride(1)) * elem_bytes,
      static_cast<cuuint64_t>(t.stride(0)) * elem_bytes};
  const cuuint32_t box[3] = {box0, box1, 1};
  const cuuint32_t unit[3] = {1, 1, 1};
  const CUresult r = cuTensorMapEncodeTiled(
      &map, type, 3, t.data_ptr(), dims, strides, box, unit,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS,
              "cuTensorMapEncodeTiled failed: ", static_cast<int>(r));
  return map;
}

// [rows][cols] bf16 with a {cols_box, rows_box} box, 128 B swizzled.
static CUtensorMap make_map_2d_sw128(const torch::Tensor& t, uint32_t box0,
                                     uint32_t box1) {
  CUtensorMap map;
  const cuuint64_t dims[2] = {static_cast<cuuint64_t>(t.size(1)),
                              static_cast<cuuint64_t>(t.size(0))};
  const cuuint64_t strides[1] = {static_cast<cuuint64_t>(t.stride(0)) * 2};
  const cuuint32_t box[2] = {box0, box1};
  const cuuint32_t unit[2] = {1, 1};
  const CUresult r = cuTensorMapEncodeTiled(
      &map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, t.data_ptr(), dims, strides, box,
      unit, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", static_cast<int>(r));
  return map;
}

// The format a tier's scales imply: e8m0 bytes for MXFP4, bf16 for INT4.
static int tier_format(const torch::Tensor& s13) {
  if (s13.scalar_type() == at::kBFloat16) return INT4;
  TORCH_CHECK(s13.element_size() == 1, "scales must be e8m0 bytes or bf16");
  return MXFP4;
}

static void fill_tier(Tier& tr, const torch::Tensor& w13,
                      const torch::Tensor& s13, const torch::Tensor& w2,
                      const torch::Tensor& s2) {
  if (!w13.defined() || w13.numel() == 0) {
    std::memset(&tr, 0, sizeof(tr));
    return;
  }
  TORCH_CHECK(s2.scalar_type() == s13.scalar_type(),
              "w13 and w2 scales must share a format");
  const torch::Tensor* ts[4] = {&w13, &w2, &s13, &s2};
  for (const torch::Tensor* t : ts)
    TORCH_CHECK(t->is_contiguous(), "tier tensors must be contiguous");
  TORCH_CHECK(w13.size(1) == HIDDEN / 16 && w13.size(2) == 2 * INTER * 2,
              "unexpected w13 layout");
  TORCH_CHECK(w2.size(1) == INTER / 16 && w2.size(2) == HIDDEN * 2,
              "unexpected w2 layout");
  TORCH_CHECK(s13.size(1) == HIDDEN / 32 && s13.size(2) == 2 * INTER,
              "unexpected w13 scale layout");
  TORCH_CHECK(s2.size(1) == INTER / 32 && s2.size(2) == HIDDEN,
              "unexpected w2 scale layout");
  tr.w[0] = make_map(w13, CU_TENSOR_MAP_DATA_TYPE_UINT32, 4, 2 * R0, KT0);
  tr.w[1] = make_map(w2, CU_TENSOR_MAP_DATA_TYPE_UINT32, 4, 2 * R1, KT1);
  const bool int4 = tier_format(s13) == INT4;
  const CUtensorMapDataType st =
      int4 ? CU_TENSOR_MAP_DATA_TYPE_UINT16 : CU_TENSOR_MAP_DATA_TYPE_UINT8;
  TORCH_CHECK(int4, "the slice kernel takes INT4 (bf16 scales)");
  tr.s[0] = make_map(s13, st, 2, R0, G0);
  tr.s[1] = make_map(s2, st, 2, R1, G1);
}

int64_t workspace_bytes() { return sizeof(Workspace); }

void forward(torch::Tensor out, torch::Tensor x, torch::Tensor topk_ids,
             torch::Tensor topk_weights, torch::Tensor hot_map,
             torch::Tensor cold_map, torch::Tensor primary,
             torch::Tensor secondary, torch::Tensor primary_hot,
             int64_t ep_rank, bool schedule, torch::Tensor hot_w13,
             torch::Tensor hot_s13, torch::Tensor hot_w2, torch::Tensor hot_s2,
             torch::Tensor cold_w13, torch::Tensor cold_s13,
             torch::Tensor cold_w2, torch::Tensor cold_s2,
             torch::Tensor workspace, bool pdl_launch, torch::Tensor padding,
             torch::Tensor shared_w13, torch::Tensor shared_w2,
             double shared_scale) {
  const int T = static_cast<int>(x.size(0));
  TORCH_CHECK(T >= 1 && T <= MAX_TOKENS, "tiered decode handles 1..32 tokens");
  TORCH_CHECK(x.size(1) == HIDDEN && x.scalar_type() == at::kBFloat16 &&
                  x.is_contiguous(),
              "x must be [T, 6144] bf16");
  TORCH_CHECK(topk_ids.size(1) == TOPK && topk_ids.is_contiguous(),
              "topk_ids must be [T, 8]");
  TORCH_CHECK(
      topk_weights.scalar_type() == at::kFloat && topk_weights.is_contiguous(),
      "topk_weights must be fp32");
  TORCH_CHECK(
      hot_map.scalar_type() == at::kInt && cold_map.scalar_type() == at::kInt,
      "slot maps must be int32");
  TORCH_CHECK(workspace.numel() >= static_cast<int64_t>(sizeof(Workspace)),
              "workspace too small");
  // empty placement tensors: hot_map / cold_map already say what runs here
  Placement pl{};
  if (primary.numel() > 0) {
    pl.num_experts = static_cast<int>(primary.numel());
    TORCH_CHECK(pl.num_experts <= MAX_EXPERTS &&
                    secondary.numel() == pl.num_experts &&
                    primary_hot.numel() == pl.num_experts &&
                    hot_map.numel() == pl.num_experts &&
                    cold_map.numel() == pl.num_experts,
                "placement tables must cover the same <= 512 experts");
    TORCH_CHECK(primary.scalar_type() == at::kInt &&
                    secondary.scalar_type() == at::kInt &&
                    primary_hot.scalar_type() == at::kInt,
                "placement tables must be int32");
    TORCH_CHECK(ep_rank >= 0 && ep_rank < EP, "the assignment is for 4 GPUs");
    pl.primary = primary.data_ptr<int>();
    pl.secondary = secondary.data_ptr<int>();
    pl.primary_hot = primary_hot.data_ptr<int>();
    pl.ep_rank = static_cast<int>(ep_rank);
    pl.schedule = schedule;
  }
  const bool has_hot = hot_w13.defined() && hot_w13.numel() > 0;
  const bool has_cold = cold_w13.defined() && cold_w13.numel() > 0;
  TORCH_CHECK(has_hot || has_cold || (shared_w13.defined() && shared_w13.numel() > 0),
              "at least one tier or the shared expert is needed");
  const int fmt =
      has_hot || has_cold ? tier_format(has_hot ? hot_s13 : cold_s13) : INT4;
  TORCH_CHECK(!(has_hot && has_cold) || tier_format(cold_s13) == fmt,
              "both tiers must share a weight format");
  Params p{};
  fill_tier(p.tier[0], hot_w13, hot_s13, hot_w2, hot_s2);
  fill_tier(p.tier[1], cold_w13, cold_s13, cold_w2, cold_s2);
  p.ws = reinterpret_cast<Workspace*>(workspace.data_ptr());
  p.has_shared = shared_w13.defined() && shared_w13.numel() > 0;
  p.shared_scale = static_cast<float>(shared_scale);
  if (p.has_shared) {
    TORCH_CHECK(shared_w13.scalar_type() == at::kBFloat16 &&
                    shared_w2.scalar_type() == at::kBFloat16 &&
                    shared_w13.is_contiguous() && shared_w2.is_contiguous() &&
                    shared_w13.size(0) == 2 * INTER && shared_w13.size(1) == HIDDEN &&
                    shared_w2.size(0) == HIDDEN && shared_w2.size(1) == INTER,
                "shared expert: bf16 w13 [1024, 6144] and w2 [6144, 512]");
    p.sw[0] = make_map_2d_sw128(shared_w13, CKS, RS);
    p.sw[1] = make_map_2d_sw128(shared_w2, CKS, RS);
  }
  const int hot_size =
      hot_w13.defined() ? static_cast<int>(hot_w13.size(0)) : 0;
  const int cold_size =
      cold_w13.defined() ? static_cast<int>(cold_w13.size(0)) : 0;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const size_t route_smem =
      2 * static_cast<size_t>(hot_size + cold_size) * sizeof(int);
  static bool attrs = false;
  if (!attrs) {
    cudaFuncSetAttribute(layer_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         SMEM_BYTES);
    attrs = true;
  }
  (void)fmt;
  cudaLaunchAttribute pdl[1];
  pdl[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  pdl[0].val.programmaticStreamSerializationAllowed = 1;
  auto config = [&](dim3 grid, dim3 block, size_t smem) {
    cudaLaunchConfig_t c = {};
    c.gridDim = grid;
    c.blockDim = block;
    c.dynamicSmemBytes = smem;
    c.stream = stream;
    c.attrs = pdl;
    c.numAttrs = pdl_launch ? 1 : 0;
    return c;
  };
  const bool* pad = nullptr;
  if (padding.defined() && padding.numel() > 0) {
    TORCH_CHECK(padding.scalar_type() == at::kBool && padding.is_contiguous() &&
                    padding.numel() >= T,
                "padding must be >= T contiguous bools");
    pad = padding.data_ptr<bool>();
  }
  const auto* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  cudaLaunchConfig_t c = config(dim3(T + 1), dim3(PREP_THREADS), route_smem);
  if (topk_ids.scalar_type() == at::kInt)
    C10_CUDA_CHECK(cudaLaunchKernelEx(
        &c, route_prep_kernel<int>, p.ws, xp, topk_ids.data_ptr<int>(), pad,
        topk_weights.data_ptr<float>(), hot_map.data_ptr<int>(),
        cold_map.data_ptr<int>(), pl, T, hot_size, cold_size));
  else
    C10_CUDA_CHECK(cudaLaunchKernelEx(
        &c, route_prep_kernel<int64_t>, p.ws, xp, topk_ids.data_ptr<int64_t>(),
        pad, topk_weights.data_ptr<float>(), hot_map.data_ptr<int>(),
        cold_map.data_ptr<int>(), pl, T, hot_size, cold_size));
  c = config(dim3(GRID), dim3(THREADS), SMEM_BYTES);
  C10_CUDA_CHECK(cudaLaunchKernelEx(&c, layer_kernel, p));
  c = config(dim3(HIDDEN / 1024, T), dim3(256), 0);
  C10_CUDA_CHECK(
      cudaLaunchKernelEx(&c, finalize_kernel, p.ws,
                         reinterpret_cast<__nv_bfloat16*>(out.data_ptr())));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace tiered_decode

#ifdef TD_CTA_TRACE
void td_stamp(int64_t tag) {
  tiered_decode::td_stamp_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
      static_cast<int>(tag));
}
torch::Tensor td_dump() {
  unsigned int n = 0;
  C10_CUDA_CHECK(cudaDeviceSynchronize());
  C10_CUDA_CHECK(
      cudaMemcpyFromSymbol(&n, tiered_decode::td_trace_n, sizeof(n)));
  n = n > (1u << 18) ? (1u << 18) : n;
  auto out = torch::empty({static_cast<int64_t>(n), 4}, torch::kInt64);
  if (n)
    C10_CUDA_CHECK(cudaMemcpyFromSymbol(out.data_ptr(), tiered_decode::td_trace,
                                        n * 4 * sizeof(uint64_t)));
  const unsigned int zero = 0;
  C10_CUDA_CHECK(
      cudaMemcpyToSymbol(tiered_decode::td_trace_n, &zero, sizeof(zero)));
  return out;
}
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#ifdef TD_CTA_TRACE
  m.def("td_stamp", &td_stamp, "probe: timestamp on the current stream");
  m.def("td_dump", &td_dump, "probe: per-CTA records since the last dump");
#endif
  m.def("forward", &tiered_decode::forward,
        "Tiered MXFP4 / INT4 decode MoE (6144 x 2048 experts, top-8)");
  m.def("workspace_bytes", &tiered_decode::workspace_bytes,
        "Workspace size in bytes");
}

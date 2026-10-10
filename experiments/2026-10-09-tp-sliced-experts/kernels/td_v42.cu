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
// One persistent kernel per layer (route_prep -> layer_kernel -> finalize,
// chained with programmatic dependent launch). layer_kernel is warp
// specialized: a producer warp claims groups of 128-row x 512-K units from a
// dynamic per-tier queue (shared-expert w13, routed w13, shared-expert w2,
// routed w2) and streams them with TMA into a 4-stage smem ring; eight
// consumer warps decode INT4 to exact f16 (one shift, four lop3, four hfma2
// per word) and run mma.sync m16n8k16 with fp32 accumulation. Routed w13
// partials go to y13 through fp32 reds; the CTA that completes an entry's w13
// applies silu * up and publishes the entry, whose w2 units then run. The
// shared expert (bf16 TP slice) is computed in the same kernel, so a side
// stream is not needed. The output is routed_scale * routed + shared, the
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
#ifndef TD_V37_OFF  // consumer changes of v37 (Astra consult 9)
  #define TD_MMA_NV
  #define TD_MB2
  #define TD_GLOOP
#endif
#ifdef TD_ABL_STATIC
  #define TD_NO_STEAL
#endif
#ifndef TD_NO_RECPF
  #define TD_RECPF
#endif
#ifndef TD_GUIDED
  #define TD_GUIDED 0  // max routed w2 units per claim, guided (0: GR1)
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
// timing probe: x rows reserved per stage (copies and reads clamped; results
// invalid when an entry has more tokens)
#ifndef TD_XROWS
  #define TD_XROWS MAX_TOK
#endif
constexpr int XROWS = TD_XROWS;
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
// stages start on 1 KB boundaries (128 B swizzle)
constexpr int STAGE_BYTES =
    (W_BYTES + S_BYTES + XROWS * XROW_STRIDE + 1023) / 1024 * 1024;
static_assert(MAX_TOKENS * XS_BYTES <= STAGE_BYTES - W_BYTES,
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
constexpr int PREP_THREADS =
    1024;  // route_prep block: 6 hidden elements per thread
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
  alignas(128) __half x2[MAX_ROUTES * X2_LD];
  alignas(128) float y13[MAX_ROUTES * 2 * INTER];
  alignas(128) float y[MAX_TOKENS * HIDDEN];
  int done13[2]
            [MAX_LIST];    // w13 chunks flushed per entry, zeroed by route_prep
  int ready[2][MAX_LIST];  // epoch once the entry's activation rows are written
  int epoch;               // bumped by route_prep every call
  alignas(128) __nv_bfloat16 x13b[MAX_TOKENS * HIDDEN];  // shared expert input
  alignas(128) __nv_bfloat16 x2s[MAX_TOKENS * INTER];    // its activation
  alignas(128) float y13s[MAX_TOKENS * 2 * INTER];
  int next[2];  // per tier: next group to claim (hot, cold), zeroed by
                // route_prep
  int done_s;   // shared w13 chunks flushed
  int ready_s;  // epoch once x2s is written
  int T;
};
static_assert(offsetof(Workspace, x13) % 16 == 0 &&
                  offsetof(Workspace, x2) % 16 == 0,
              "TMA alignment");

struct Params {
  Tier tier[2];
  CUtensorMap
      sw[2];  // shared expert w13 [2 * INTER][HIDDEN], w2 [HIDDEN][INTER]
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
#ifdef TD_MMA_NV  // pure arithmetic: let the compiler schedule it
  #define TD_MMA_ASM asm
#else
  #define TD_MMA_ASM asm volatile
#endif
__device__ __forceinline__ void mma_f16(float* d, const uint32_t* a,
                                        uint32_t b0, uint32_t b1,
                                        const float* c) {
  TD_MMA_ASM(
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
// the same into a slot the caller reserved, word 3 stamped by the caller: no
// atomic round trip on the traced warp
__device__ __forceinline__ void td_record_at(unsigned slot, int ph, uint64_t t0, uint64_t t1,
                                             uint64_t t3, int n_hot,
                                             int n_cold) {
  uint32_t sm;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
  const unsigned i = slot & ((1u << 18) - 1);
  td_trace[i][0] = uint64_t(ph & 0xFF) | (uint64_t(blockIdx.x & 0xFF) << 8) |
                   (uint64_t(sm & 0xFFFF) << 16) |
                   (uint64_t(n_hot & 0xFFFF) << 32) |
                   (uint64_t(n_cold & 0xFFFF) << 48);
  td_trace[i][1] = t0;
  td_trace[i][2] = t1;
  td_trace[i][3] = t3;
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
                  int num_tokens, int hot_size, int cold_size,
                  int num_experts, float routed_scale) {
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
    __nv_bfloat16* __restrict__ outb =
        ws->x13b + static_cast<size_t>(t) * HIDDEN;
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
    wt = topk_weights[r] * routed_scale;
    if (e >= 0 && E > 0 && e >= E) e = -1;
  }
  // without a placement the slot maps are read with the ids, not after them
  __shared__ int maps[2 * MAX_EXPERTS];
  if (E == 0)
    for (int i = threadIdx.x; i < num_experts; i += blockDim.x) {
      maps[i] = hot_map[i];
      maps[MAX_EXPERTS + i] = cold_map[i];
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
      const int h = maps[e], c = maps[MAX_EXPERTS + e];
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
#ifdef TD_ABL_NOFLUSH  // timing ablation: no stores / reds (acc kept live)
  float z = 0.f;
#pragma unroll
  for (int mb = 0; mb < 4; ++mb)
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      z += acc[mb][i];
      acc[mb][i] = 0.f;
    }
  if (z == 1234.5f) red_add_if(base, z, true);
  return;
#endif
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

// The shared expert's silu * up for all T tokens (bf16 fragments of x2s);
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
      out[frag_slot(k)] =
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
#ifdef TD_ABL_NOMMA  // timing ablation: skip the routed math
  return;
#endif
#ifdef TD_ABL_LDSONLY  // timing ablation: the routed smem reads, no math
  {
    constexpr int KR = PH ? 1024 : 512, SR = PH ? 256 : 128;
    uint32_t z = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const uint4 a = lds_v4(wb + (2 * j) * KR), b = lds_v4(wb + (2 * j + 1) * KR),
                  c = lds_v4(sb + j * SR), e = lds_v4(xb + j * 64);
      z ^= a.x ^ a.y ^ a.z ^ a.w ^ b.x ^ b.y ^ b.z ^ b.w ^ c.x ^ c.w ^ e.x ^ e.w;
    }
    acc[0][0] += __uint_as_float(z & 1u);
    return;
  }
#endif
  constexpr int KROW = PH ? 1024 : 512;  // bytes per k16 row of the box
  constexpr int SROW = PH ? 256 : 128;   // bytes per scale group row
#pragma unroll
  for (int j = 0; j < 4; ++j) {
#ifdef TD_ABL_NOLDS  // timing ablation: the routed math on register data
    const uint4 w0 = make_uint4(wb ^ j, wb + j, wb * 3 + j, wb ^ (j << 7)),
                w1 = make_uint4(sb ^ j, sb + j, sb * 3 + j, sb ^ (j << 7)),
                sw = make_uint4(0x3c003c00u, 0x3c003c00u, 0x3c003c00u, 0x3c003c00u),
                xv = make_uint4(xb ^ j, xb + j, xb * 5, xb ^ 0x3c00u);
#else
    const uint4 w0 = lds_v4(wb + (2 * j) * KROW);
    const uint4 w1 = lds_v4(wb + (2 * j + 1) * KROW);
    const uint4 sw = lds_v4(sb + j * SROW);
    const uint4 xv = lds_v4(xb + j * 64);
#endif
    const uint32_t w0s[4] = {w0.x, w0.y, w0.z, w0.w},
                   w1s[4] = {w1.x, w1.y, w1.z, w1.w};
    const uint32_t sws[4] = {sw.x, sw.y, sw.z, sw.w};
#ifdef TD_MB2  // two row blocks at a time: independent decode / MMA chains
    const float zero[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int mb = 0; mb < 4; mb += 2) {
      float d0[4], d1[4];
      uint32_t a0[4], a1[4];
      decode_int4_fast(w0s[mb], a0);
      decode_int4_fast(w0s[mb + 1], a1);
      mma_f16(d0, a0, xv.x, xv.y, zero);
      mma_f16(d1, a1, xv.x, xv.y, zero);
      decode_int4_fast(w1s[mb], a0);
      decode_int4_fast(w1s[mb + 1], a1);
      mma_f16(d0, a0, xv.z, xv.w, d0);
      mma_f16(d1, a1, xv.z, xv.w, d1);
      const float s00 = __uint_as_float(sws[mb] << 16),
                  s01 = __uint_as_float(sws[mb] & 0xFFFF0000u),
                  s10 = __uint_as_float(sws[mb + 1] << 16),
                  s11 = __uint_as_float(sws[mb + 1] & 0xFFFF0000u);
      acc[mb][0] = fmaf(s00, d0[0], acc[mb][0]);
      acc[mb][1] = fmaf(s00, d0[1], acc[mb][1]);
      acc[mb][2] = fmaf(s01, d0[2], acc[mb][2]);
      acc[mb][3] = fmaf(s01, d0[3], acc[mb][3]);
      acc[mb + 1][0] = fmaf(s10, d1[0], acc[mb + 1][0]);
      acc[mb + 1][1] = fmaf(s10, d1[1], acc[mb + 1][1]);
      acc[mb + 1][2] = fmaf(s11, d1[2], acc[mb + 1][2]);
      acc[mb + 1][3] = fmaf(s11, d1[3], acc[mb + 1][3]);
    }
#else
#pragma unroll
    for (int mb = 0; mb < 4; ++mb) {
      const float zero[4] = {0.f, 0.f, 0.f, 0.f};
      float d[4];
      uint32_t a[4];
#ifdef TD_ABL_NODECODE  // timing ablation: raw words as fragments
      a[0] = a[1] = a[2] = a[3] = w0s[mb];
      mma_f16(d, a, xv.x, xv.y, zero);
      a[0] = a[1] = a[2] = a[3] = w1s[mb];
      mma_f16(d, a, xv.z, xv.w, d);
#else
      decode_int4_fast(w0s[mb], a);
      mma_f16(d, a, xv.x, xv.y, zero);
      decode_int4_fast(w1s[mb], a);
      mma_f16(d, a, xv.z, xv.w, d);
#endif
      const float s0 = __uint_as_float(sws[mb] << 16),
                  s1 = __uint_as_float(sws[mb] & 0xFFFF0000u);
      acc[mb][0] = fmaf(s0, d[0], acc[mb][0]);
      acc[mb][1] = fmaf(s0, d[1], acc[mb][1]);
      acc[mb][2] = fmaf(s1, d[2], acc[mb][2]);
      acc[mb][3] = fmaf(s1, d[3], acc[mb][3]);
    }
#endif
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
// routed w2 units per claim (consecutive tiles of one entry): GR1 above
// GR1_T tokens, else 1. Larger claims amortize the producer's claim, record
// and ready round trips (M=16/32 -5..7%); at M=8 they unbalance the tail.
#ifndef TD_GR1
  #define TD_GR1 2
#endif
#ifndef TD_GR1_T
  #define TD_GR1_T 8
#endif
constexpr int GR1 = TD_GR1;
static_assert(TILES1 % GR1 == 0, "w2 groups tile an entry");
static_assert(TD_GUIDED == 0, "v39 sizes w2 claims per call, not guided");
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
__device__ __forceinline__ int queue_len(int n, bool sh, int g1) {
  return (sh ? TILES_S0 * SG0 + TILES_S1 : 0) +
         n * (TILES0 * R0S + TILES1 / g1);
}
__device__ __forceinline__ Group group_at(int gi, int n, bool sh, int g1) {
  const int ns0 = sh ? TILES_S0 * SG0 : 0, nr0 = n * TILES0 * R0S,
            ns1 = sh ? TILES_S1 : 0;
  if (gi < ns0) return {K_S0, gi / SG0, (gi % SG0) * SCH0, SCH0};
  gi -= ns0;
  if (gi < nr0) return {K_R0, gi / R0S, (gi % R0S) * R0CH, R0CH};
  gi -= nr0;
  if (gi < ns1) return {K_S1, gi, 0, CHUNKS_S1};
  gi -= ns1;
  return {K_R1, gi, 0, g1};
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
#ifdef TD_UNIT_TRACE
  __shared__ uint64_t s_issue[STAGES];  // producer's issue time per stage
#endif
  __shared__ __align__(16) float scr_all[CONSUMER_WARPS][MAX_TOK * SCR_LD];
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
  const int g1 = T > TD_GR1_T ? GR1 : 1;
  const int len[2] = {queue_len(n_tier[0], sh, g1),
                      queue_len(n_tier[1], false, g1)};
  const int cold_ctas = len[1] == 0 ? 0 : len[0] == 0 ? GRID : TD_COLD_CTAS;
  const int own = static_cast<int>(blockIdx.x) < cold_ctas ? 1 : 0;

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
    TD_V(const uint64_t td_p0 = td_now(); uint64_t td_spin = 0, td_empty = 0;)
#ifdef TD_UNIT_TRACE
    unsigned td_ps = 0;  // producer lane 0: this CTA's producer records
    if (lane == 0) td_ps = atomicAdd(&td_trace_n, 512u);
#endif
    const float xs13r =
        lane < T ? ws->xs13[lane] : 0.f;  // token lane's x13 scale
    int q = own, last_ready = -1, it = 0;
    bool stolen = false;
    int gi = 0, gk = 1;  // the current claim and its unit count
#ifdef TD_RECPF
    // the next claim, and the entry record prefetched for it
    int gq = 0, pf_e = -1, pf_q = -1, pf_ntok = 0, pf_local = 0, pf_tok = 0,
        pf_route = 0;
    float pf_wt = 0.f;
    if (lane == 0) {
      gi = atomicAdd(&ws->next[q], 1);
      gq = atomicAdd(&ws->next[q], 1);
    }
    gq = __shfl_sync(0xffffffffu, gq, 0);
#else
    if (lane == 0) gi = atomicAdd(&ws->next[q], 1);
#endif
    gi = __shfl_sync(0xffffffffu, gi, 0);
#ifdef TD_ABL_STATIC
    const int ns = own ? cold_ctas : GRID - cold_ctas;
    gi = own ? static_cast<int>(blockIdx.x) : static_cast<int>(blockIdx.x) - cold_ctas;
  #ifdef TD_RECPF
    gq = gi + ns;
  #endif
#endif
    while (true) {
      if (gi >= (q ? len[1] : len[0])) {
#ifdef TD_NO_STEAL
        break;
#endif
#ifdef TD_NO_STEAL_COLD
        if (q == 0) break;  // hot CTAs never take cold work
#endif
        if (stolen) break;
        stolen = true;
        q ^= 1;
        last_ready = -1;
        gk = 1;
#ifdef TD_RECPF
        if (lane == 0) {
          gi = atomicAdd(&ws->next[q], 1);
          gq = atomicAdd(&ws->next[q], 1);
        }
        gq = __shfl_sync(0xffffffffu, gq, 0);
#else
        if (lane == 0) gi = atomicAdd(&ws->next[q], 1);
#endif
        gi = __shfl_sync(0xffffffffu, gi, 0);
        continue;
      }
#ifdef TD_UNIT_TRACE
      if (lane == 0) td_record_at(td_ps++, 170, td_now(), 0, 0, 0, 0);
#endif
      // claim the next group now: its round trip overlaps this group's issue
      int gn = 0, kn = 1;
#if TD_GUIDED > 0
      {
        // past the start of R1 the counter only moves through R1 units
        const int r1s = (q ? len[1] : len[0]) - (q ? n_tier[1] : n_tier[0]) * TILES1;
        if (gi >= r1s)
          kn = max(1, min(TD_GUIDED, ((q ? len[1] : len[0]) - gi - gk) / (2 * GRID)));
      }
#endif
#ifndef TD_NO_PREFETCH_CLAIM
#ifdef TD_ABL_STATIC
  #ifdef TD_RECPF
      gn = gq + ns;
  #else
      gn = gi + ns;
  #endif
#else
      if (lane == 0) gn = atomicAdd(&ws->next[q], kn);
#endif
#endif
      Group gr = group_at(gi, (q ? n_tier[1] : n_tier[0]), q == 0 && sh, g1);
#if TD_GUIDED > 0
      if (gr.kind == K_R1) gr.nch = min(gk, (q ? len[1] : len[0]) - gi);
#endif
      const Tier& tr = p.tier[q];
      // a routed group's entry record, once per group
      const bool routed = gr.kind == K_R0 || gr.kind == K_R1;
      int ei = 0, t0 = 0, ntok = 0, local = 0, tokl = 0, routel = 0;
      float fl = 0.f;
      if (gr.kind == K_R0) {
        ei = gr.x / TILES0;
        t0 = gr.x - ei * TILES0;
      } else if (gr.kind == K_R1) {
        ei = gr.x / (TILES1 / g1);
        t0 = (gr.x - ei * (TILES1 / g1)) * g1;
      }
      const auto load_record = [&](int e_) {
        // one round trip: every field at once (lanes past ntok read a valid
        // slot and are masked later)
        const Expert& e = ws->lists[q][e_];
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
      };
#ifdef TD_RECPF
      const auto prefetch = [&](int q_, int e_) {
        const Expert& e = ws->lists[q_][e_];
        const int l7 = lane & (MAX_TOK - 1);
        pf_e = e_;
        pf_q = q_;
        pf_ntok = e.ntok;
        pf_local = e.local;
        pf_tok = e.tok[l7];
        pf_route = e.route[l7];
        pf_wt = e.wt[l7];
      };
      if (routed) {
        if (pf_e != ei || pf_q != q) prefetch(q, ei);  // not prefetched
        ntok = pf_ntok;
        local = pf_local;
        tokl = pf_tok;
        routel = pf_route;
        const float xs = __shfl_sync(0xffffffffu, xs13r, tokl & 31);
        fl = gr.kind == K_R0 ? xs : pf_wt;
      }
      // the next group's record: its loads stay in flight across this
      // group's issue
      if (gq < (q ? len[1] : len[0])) {
        const Group gq_ = group_at(gq, (q ? n_tier[1] : n_tier[0]), q == 0 && sh, g1);
        if (gq_.kind == K_R0)
          prefetch(q, gq_.x / TILES0);
        else if (gq_.kind == K_R1)
          prefetch(q, gq_.x / (TILES1 / g1));
      }
#else
      if (routed) load_record(ei);
#endif
      for (int ci = 0; ci < gr.nch; ++ci, ++it) {
        const int s = it % STAGES, c = gr.c0 + ci;
        unsigned char* dst = ring + static_cast<size_t>(s) * STAGE_BYTES;
        TD_V(const uint64_t td_e0 = td_now();)
        if (lane == 0 && it >= STAGES)
          mbar_wait(&empty[s], ((it / STAGES) - 1) & 1);
        TD_V(td_empty += td_now() - td_e0;)
#ifdef TD_UNIT_TRACE
        if (lane == 0) {
          s_issue[s] = td_now();
          td_record_at(td_ps++, 171, td_e0, s_issue[s], 0, gr.kind, ci);
        }
#else
#ifdef TD_UNIT_TRACE
        if (lane == 0) s_issue[s] = td_now();
#endif
#endif
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
            mbar_expect_tx(&full[s], W_BYTES + S_BYTES + min(ntok, XROWS) * XROW_BYTES0);
            tma_3d(dst, &tr.w[0], t0 * 2 * R0, c * KT0, local, &full[s]);
            tma_3d(dst + W_BYTES, &tr.s[0], t0 * R0, c * G0, local, &full[s]);
          }
          __syncwarp();
          if (lane < ntok && lane < XROWS)
            bulk_g2s(dst + W_BYTES + S_BYTES + lane * XROW_STRIDE,
                     ws->x13 + static_cast<size_t>(tokl) * HIDDEN + c * CK0,
                     XROW_BYTES0, &full[s]);
        } else if (gr.kind == K_R1) {
#if TD_GUIDED > 0
          const int u = gr.x + ci, eu = u / TILES1, t = u - eu * TILES1;
          if (eu != ei) {
            ei = eu;
            load_record(ei);
          }
#else
          const int t = t0 + ci;
#endif
          if (lane < ntok) {
            sd_row[s][lane] = tokl;
            sd_f[s][lane] = fl;
          }
          __syncwarp();
          if (lane == 0) {
#if defined(TD_CTA_TRACE) && defined(TD_DEBUG_SUM)
            td_record(90 + q, ei, t | own << 16 | gi << 20, ntok, it);
#endif
            desc[s] = make_int4(hdr, ei, t, ntok);
            mbar_expect_tx(&full[s], W_BYTES + S_BYTES + min(ntok, XROWS) * X2_COPY);
            tma_3d(dst, &tr.w[1], t * 256, 0, local, &full[s]);
            tma_3d(dst + W_BYTES, &tr.s[1], t * 128, 0, local, &full[s]);
          }
          // weights are in flight; only the activation rows wait for the
          // entry's ready (every copying lane acquires for itself)
          if (ei != last_ready) {
            TD_V(const uint64_t w0 = td_now();)
#ifndef TD_ABL_NOREADY  // timing ablation: w2 never waits (results invalid)
            while (ld_acquire(&ws->ready[q][ei]) != epoch) __nanosleep(32);
#endif
            TD_V(td_spin += td_now() - w0;)
#ifdef TD_UNIT_TRACE
            if (lane == 0) td_record_at(td_ps++, 172, w0, td_now(), 0, 0, 0);
#endif
            fence_proxy_async();
            last_ready = ei;
          }
          __syncwarp();
#ifdef TD_SPIN_EVERY_UNIT
          while (ld_acquire(&ws->ready[q][ei]) != epoch) __nanosleep(32);
          fence_proxy_async();
#endif
          if (lane < ntok && lane < XROWS)
            bulk_g2s(dst + W_BYTES + S_BYTES + lane * XROW_STRIDE,
                     ws->x2 + static_cast<size_t>(routel) * X2_LD, X2_COPY,
                     &full[s]);
        } else {  // shared expert
          if (lane == 0) {
            desc[s] = make_int4(hdr, gr.x, c, 0);
            mbar_expect_tx(&full[s], W_BYTES + T * XS_BYTES);
            tma_2d(dst, &p.sw[gr.kind == K_S0 ? 0 : 1], c * CKS, gr.x * RS,
                   &full[s]);
          }
          if (gr.kind == K_S1 &&
              last_ready != -2) {  // every copying lane acquires
            TD_V(const uint64_t w0 = td_now();)
            while (ld_acquire(&ws->ready_s) != epoch) __nanosleep(32);
            TD_V(td_spin += td_now() - w0;)
            fence_proxy_async();
          }
          if (gr.kind == K_S1) last_ready = -2;
          __syncwarp();
          if (lane < T)
            bulk_g2s(dst + W_BYTES + lane * XS_BYTES,
                     (gr.kind == K_S0
                          ? ws->x13b + static_cast<size_t>(lane) * HIDDEN
                          : ws->x2s + static_cast<size_t>(lane) * INTER) +
                         c * CKS,
                     XS_BYTES, &full[s]);
        }
      }
#ifdef TD_NO_PREFETCH_CLAIM
      if (lane == 0) gn = atomicAdd(&ws->next[q], 1);
#endif
#ifdef TD_UNIT_TRACE
      const uint64_t td_cl = td_now();
#endif
#ifdef TD_RECPF
      gi = gq;
      gq = __shfl_sync(0xffffffffu, gn, 0);
#else
      gi = __shfl_sync(0xffffffffu, gn, 0);
#endif
      gk = kn;
#ifdef TD_UNIT_TRACE
      if (lane == 0) td_record_at(td_ps++, 173, td_cl, td_now(), 0, 0, 0);
#endif
    }
    if (lane == 0) {  // end of work
      const int s = it % STAGES;
      if (it >= STAGES) mbar_wait(&empty[s], ((it / STAGES) - 1) & 1);
      desc[s] = make_int4(K_END, 0, 0, 0);
      mbar_arrive(&full[s]);
    }
    TD_V(if (lane == 0)
             td_record(60 + own, td_p0, td_p0 + td_spin, n_tier[0], n_tier[1]);)
    TD_V(if (lane == 0) td_record(80 + own, td_p0, td_p0 + td_empty, it, 0);)
    return;
  }

  // consumer warps
  const int g = lane / 4, tq = lane % 4;
  const int nt_n = (T + 7) / 8;
  float acc[4][4] = {};
  float accs[2][4][4] = {};
  int acq = -1;  // (tier, entry) whose ready this warp has acquired
  TD_V(uint64_t td_c0 = 0;)
  TD_V(uint64_t td_wait = 0; const uint64_t td_q0 = td_now();)
#ifdef TD_UNIT_TRACE
  unsigned td_slot = 0;  // warp 0 lane 0: this CTA's unit records
  if (warp == 0 && lane == 0) td_slot = atomicAdd(&td_trace_n, 512u);
#endif
  // per-warp smem offsets within a stage, computed once
  const uint32_t ring_u = smem_u32(ring), full_u = smem_u32(full),
                 empty_u = smem_u32(empty);
  const uint32_t desc_u = smem_u32(desc), sdf_u = smem_u32(&sd_f[0][2 * tq]);
  const uint32_t scr_u = smem_u32(scr_all[warp]),
                 sdr0_u = smem_u32(&sd_row[0][0]);
  static_assert(CONSUMER_WARPS == 8,
                "8 K slices (w13), 2 halves x 4 K slices (w2)");
  const uint32_t wo0 = (warp * 8) * 512 + lane * 16;
  const uint32_t so0 = W_BYTES + (warp * 4) * 128 + 16 * g;
  const uint32_t xo0 =
      W_BYTES + S_BYTES + (g % XROWS) * XROW_STRIDE + (warp * 16 + tq) * 16;
  const int h1 = warp / 4, k1 = warp % 4;
  const uint32_t wo1 = (k1 * 8) * 1024 + h1 * 512 + lane * 16;
  const uint32_t so1 = W_BYTES + (k1 * 4) * 256 + h1 * 128 + 16 * g;
  const uint32_t xo1 =
      W_BYTES + S_BYTES + (g % XROWS) * XROW_STRIDE + (k1 * 16 + tq) * 16;
  int s = 0;
  uint32_t ph = 0;
  for (int it = 0;; ++it) {
    TD_V(const uint64_t td_w0 = td_now();)
    mbar_wait_a(full_u + 8 * s, ph);
#ifdef TD_UNIT_TRACE
    const uint64_t td_f = td_now();
#endif
    TD_V(td_wait += td_now() - td_w0;)
    const uint4 du = lds_v4(desc_u + 16 * s);
    const int4 d = make_int4(du.x, du.y, du.z, du.w);
    const int kind = d.x & 15;
    if (kind == K_END) break;
#ifdef TD_UNIT_TRACE
    if (warp == 0 && lane == 0)
      td_record_at(td_slot++, 120 + kind, s_issue[s], td_w0, td_f, d.w, (d.x >> 4) & 1);
#endif
    const int q = (d.x >> 4) & 1, last = (d.x >> 9) & 1, nch = d.x >> 16;
    const unsigned char* st = ring + static_cast<size_t>(s) * STAGE_BYTES;
    const uint32_t st_u = ring_u + s * STAGE_BYTES;
    const uint32_t empty_s = empty_u + 8 * s;
    const Expert* experts = ws->lists[q];
    // this lane's two tokens of a routed unit: destination rows and scales
    const int ntok = d.w;
#ifdef TD_GLOOP
    // flush metadata of stage s (plain shared loads the compiler may schedule
    // into the math), read before the stage is released
    const auto flush_meta = [&](int s_, float* fs_, int* rows_, bool r1) {
      fs_[0] = sd_f[s_][2 * tq];
      fs_[1] = sd_f[s_][2 * tq + 1];
      if (r1) {
        const unsigned char* xr = ring + static_cast<size_t>(s_) * STAGE_BYTES +
                                  W_BYTES + S_BYTES +
                                  ((2 * tq) % XROWS) * XROW_STRIDE + XROW_BYTES1;
        fs_[0] *= *reinterpret_cast<const float*>(xr);
        fs_[1] *= *reinterpret_cast<const float*>(
            xr + (XROWS > 1 ? XROW_STRIDE : 0));
      }
#pragma unroll
      for (int j = 0; j < MAX_TOK / 2; ++j)
        rows_[j] = sd_row[s_][(lane >> 4) + 2 * j];
    };
    float fs[2];
    int rows4[MAX_TOK / 2];
#else
    // read unconditionally (stale past ntok, masked at the flush)
    float fs[2] = {__uint_as_float(lds_u32(sdf_u + s * MAX_TOK * 4)),
                   __uint_as_float(lds_u32(sdf_u + s * MAX_TOK * 4 + 4))};
    if (kind == K_R1) {  // * the x2 row's scale, which arrived with the row
      const uint32_t xr = ring_u + s * STAGE_BYTES + W_BYTES + S_BYTES +
                          ((2 * tq) % XROWS) * XROW_STRIDE + XROW_BYTES1;
      fs[0] *= __uint_as_float(lds_u32(xr));
      fs[1] *= __uint_as_float(lds_u32(xr + (XROWS > 1 ? XROW_STRIDE : 0)));
    }
    // the flush's destination rows (tokens lane / 16 + 2 j), read before the
    // stage is released: the producer rewrites sd_row[s] for the next unit
    int rows4[MAX_TOK / 2];
#pragma unroll
    for (int j = 0; j < MAX_TOK / 2; ++j)
      rows4[j] = static_cast<int>(
          lds_u32(sdr0_u + (s * MAX_TOK + (lane >> 4) + 2 * j) * 4));
#endif

    if (kind == K_R0) {
#ifdef TD_GLOOP
      // the group's nch chunks in a row: its later stages carry nothing new
      // for the consumer (same entry, tile, tokens), so no descriptor reads
#ifdef TD_UNIT_TRACE
      uint64_t td_fc = td_f;
#endif
      for (int c = 0;; ++c) {
        const uint32_t su = ring_u + s * STAGE_BYTES;
#ifdef TD_UNIT_TRACE
        const uint64_t td_cs = td_now();
#endif
        consume_routed<1>(su + wo1, su + so1, su + xo1, acc);
        if (c == nch - 1) flush_meta(s, fs, rows4, false);
        __syncwarp();
        if (lane == 0) mbar_arrive_a(empty_u + 8 * s);
#ifdef TD_UNIT_TRACE
        if (warp == 0 && lane == 0)
          td_record_at(td_slot++, 151, td_cs, td_now(), td_fc, 0, 0);
#endif
        if (c == nch - 1) break;
        if (++s == STAGES) {
          s = 0;
          ph ^= 1;
        }
        TD_V(const uint64_t td_w1 = td_now();)
        mbar_wait_a(full_u + 8 * s, ph);
        TD_V(td_wait += td_now() - td_w1;)
#ifdef TD_UNIT_TRACE
        td_fc = td_now();
        if (warp == 0 && lane == 0)
          td_record_at(td_slot++, 120 + K_R0, s_issue[s], td_w1, td_fc, ntok,
                       q);
#endif
      }
      {
#else
#ifdef TD_UNIT_TRACE
      const uint64_t td_cs = td_now();
#endif
      consume_routed<1>(st_u + wo1, st_u + so1, st_u + xo1, acc);
      __syncwarp();
      if (lane == 0) mbar_arrive_a(empty_s);
#ifdef TD_UNIT_TRACE
      if (warp == 0 && lane == 0)
        td_record_at(td_slot++, 151, td_cs, td_now(), td_f, 0, 0);
#endif
      if (last) {
#endif
        const int ei = d.y, t = d.z;
#ifdef TD_UNIT_TRACE
        const uint64_t td_ha = td_now();
#endif
        flush_rows(acc, fs, scr_u, ws->y13 + t * R0 + h1 * 64, rows4, 2 * INTER,
                   ntok, g, tq, lane);
#ifdef TD_UNIT_TRACE
        const uint64_t td_hb = td_now();
#endif
#ifdef TD_FLUSH_FENCE
        fence_acq_rel_gpu();  // every thread's own y13 atomics, before the
                              // count
#endif
        // warps 1..7 hand their reds to warp 0 and go on; warp 0 counts the
        // chunks and, if this CTA completed the entry, activates its routes
        // and publishes ready. Nobody else waits on the count's round trip.
        if (warp != 0) {
          handoff_arrive();
        } else {
          handoff_sync();
#ifdef TD_UNIT_TRACE
          const uint64_t td_hc = td_now();
#endif
          int done = 0;
          if (lane == 0) {
#ifdef TD_ACQREL
            // the CTA barrier carries every warp's y13 reds to lane 0; its
            // acq_rel count publishes them and, for the winner, acquires all
            // other CTAs' (cumulativity), with no full fence
            int old;
            asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;"
                         : "=r"(old)
                         : "l"(&ws->done13[q][ei]), "r"(nch)
                         : "memory");
            done = old + nch == UNITS0;
#else
            __threadfence();
            done = atomicAdd(&ws->done13[q][ei], nch) + nch == UNITS0;
#endif
          }
#ifdef TD_UNIT_TRACE
          const int td_done = __shfl_sync(0xffffffffu, done, 0);
          const uint64_t td_hd = td_now();
#endif
          if (__shfl_sync(0xffffffffu, done, 0)) {
#ifndef TD_ACQREL
            __threadfence();
#endif
            __syncwarp();  // lane 0's acquire (the count) ordered before every
                           // lane's y13 reads
            const Expert& e = experts[ei];
            const int n = e.ntok;
            for (int r = 0; r < n; ++r) activate_route(ws, e.route[r], lane);
            fence_proxy_async();
#ifndef TD_ACQREL
            __threadfence();
#endif
            __syncwarp();
            if (lane == 0) st_release(&ws->ready[q][ei], epoch);
          }
#ifdef TD_UNIT_TRACE
          if (lane == 0) {
            td_record_at(td_slot++, 140, td_ha, td_hb, td_hc, 0, 0);
            td_record_at(td_slot++, 141 + td_done, td_hc, td_hd, td_now(), 0, 0);
          }
#endif
        }
        TD_V(td_c0 = td_now();)
      }
    } else if (kind == K_R1) {
#ifdef TD_UNIT_TRACE
      const uint64_t td_cs = td_now();
#endif
#if defined(TD_DEBUG_SUM) && defined(TD_CTA_TRACE)
      if (warp == 0) {  // checksum of half A's weights (32 k16 rows x 512 B) as
                        // seen in smem
        uint32_t x = 0;
        for (int r = 0; r < 32; ++r)
          for (int w = lane; w < 128; w += 32)
            x += lds_u32(st_u + r * 1024 + w * 4) * (r * 131 + w + 1);
        for (int o = 16; o; o >>= 1) x += __shfl_xor_sync(0xffffffffu, x, o);
        if (lane == 0)
          td_record(95 + q, (uint64_t)d.y << 8 | d.z, x, blockIdx.x, it);
      }
#endif
      consume_routed<1>(st_u + wo1, st_u + so1, st_u + xo1, acc);
#ifdef TD_GLOOP
      flush_meta(s, fs, rows4, true);
#endif
      __syncwarp();
      if (lane == 0) mbar_arrive_a(empty_s);
      const int t = d.z;
#ifdef TD_UNIT_TRACE
      const uint64_t td_ce = td_now();
#endif
      flush_rows(acc, fs, scr_u, ws->y + t * R1 + h1 * 64, rows4, HIDDEN, ntok,
                 g, tq, lane);
#ifdef TD_UNIT_TRACE
      if (warp == 0 && lane == 0)
        td_record_at(td_slot++, 153, td_cs, td_ce, td_now(), 0, 0);
      if (warp == 0 && lane == 0)
        td_record_at(td_slot++, 154, td_f, td_cs, td_now(), 0, 0);
#endif
    } else {  // shared expert: 32 rows per warp, 4 k16 steps, nt_n token tiles
      const unsigned char* xa = st + W_BYTES;
#ifndef TD_ABL_NOSHMMA  // timing ablation: skip the shared expert's math
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
            const uint2 bv = *reinterpret_cast<const uint2*>(
                xa + (nt * 8 + g) * XS_BYTES + ((ks >> 1) * 4 + tq) * 16 +
                (ks & 1) * 8);
            mma_bf16(accs[0][nt], af[0], bv.x, bv.y);
            mma_bf16(accs[1][nt], af[1], bv.x, bv.y);
          }
        }
      }
#endif
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
#ifndef TD_NO_FLUSH_FENCE
          __threadfence();  // every thread's own y13s atomics, before the count
#endif
          consumer_sync();
          if (threadIdx.x == 0) {
            __threadfence();
            s_last = atomicAdd(&ws->done_s, nch) + nch == UNITS_S0;
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
        }
      }
    }
    if (++s == STAGES) {
      s = 0;
      ph ^= 1;
    }
  }
#ifdef TD_CTA_TRACE
  if (lane == 0) td_record(130 + warp, td_q0, td_q0 + td_wait, 0, 0);
#endif
#ifdef TD_CTA_TRACE
  if (threadIdx.x == 0) {
    td_record(own, td_t1, td_c0, n_tier[0], n_tier[1]);
    td_record(70 + own, td_q0, td_q0 + td_wait, 0, 0);
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
                          float routed_scale, int num_experts,
                          void* stream_ptr) {
  TD_REQUIRE(T >= 1 && T <= MAX_TOKENS, "1..32 tokens");
  TD_REQUIRE(hot_size + cold_size <= 2 * MAX_LIST, "at most 512 tier slots");
  TD_REQUIRE(num_experts >= 1 && num_experts <= MAX_EXPERTS,
             "1..512 experts");
  Params p{};
  TD_REQUIRE(fill_tier(p.tier[0], hw13, hs13, hw2, hs2, hot_size) == 0,
             "hot tier maps");
  TD_REQUIRE(fill_tier(p.tier[1], cw13, cs13, cw2, cs2, cold_size) == 0,
             "cold tier maps");
  p.ws = reinterpret_cast<Workspace*>(workspace);
  p.has_shared = sw13 != nullptr;
  p.shared_scale = shared_scale;
  if (p.has_shared) {
    TD_REQUIRE(
        make_map_2d_sw128(&p.sw[0], sw13, HIDDEN, 2 * INTER, CKS, RS) == 0,
        "sw13 map");
    TD_REQUIRE(make_map_2d_sw128(&p.sw[1], sw2, INTER, HIDDEN, CKS, RS) == 0,
               "sw2 map");
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  static bool attrs = false;
  if (!attrs) {
    cudaFuncSetAttribute(
        layer_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
    attrs = true;
  }
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
  Placement pl{};
  const size_t route_smem =
      2 * static_cast<size_t>(hot_size + cold_size) * sizeof(int);
  cudaLaunchConfig_t c = config(dim3(T + 1), dim3(PREP_THREADS), route_smem);
  if (cudaLaunchKernelEx(&c, route_prep_kernel<int>, p.ws,
                         reinterpret_cast<const __nv_bfloat16*>(x), ids,
                         padding, wt, hot_map, cold_map, pl, T, hot_size,
                         cold_size, num_experts, routed_scale) != cudaSuccess)
    return -2;
  c = config(dim3(GRID), dim3(THREADS), SMEM_BYTES);
  if (cudaLaunchKernelEx(&c, layer_kernel, p) != cudaSuccess) return -3;
  c = config(dim3(HIDDEN / 1024, T), dim3(256), 0);
  if (cudaLaunchKernelEx(&c, finalize_kernel, p.ws,
                         reinterpret_cast<__nv_bfloat16*>(out)) != cudaSuccess)
    return -4;
  return 0;
}

#ifdef TD_CTA_TRACE
extern "C" int td_dump(unsigned long long* host, int cap) {
  unsigned int n = 0;
  cudaDeviceSynchronize();
  cudaMemcpyFromSymbol(&n, tiered_decode::td_trace_n, sizeof(n));
  n = n > (1u << 18) ? (1u << 18) : n;
  n = n > static_cast<unsigned>(cap) ? cap : n;
  if (n)
    cudaMemcpyFromSymbol(host, tiered_decode::td_trace,
                         n * 4 * sizeof(uint64_t));
  const unsigned int zero = 0;
  cudaMemcpyToSymbol(tiered_decode::td_trace_n, &zero, sizeof(zero));
  return static_cast<int>(n);
}
#endif

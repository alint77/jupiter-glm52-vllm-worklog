// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Prefill-size W4A16 expert GEMM on sm_90a for weights in Marlin's layout
// (symmetric INT4, bf16 group-32 scales: GLM-5.3 W4A16).
//
// Y[n][f] = sum_k X[n][k] * W[f][k] for one 256-row block of W per CTA. W is
// the wgmma A operand, sourced from registers: a Marlin word is one m16k16 A
// fragment (see tiered_decode.cu), so each of the warpgroup's four warps reads
// one 64-row Marlin tile with conflict-free 16 B loads and decodes it, and the
// warpgroup issues one wgmma per 16-row block of the tiles. X is the B operand:
// tokens on N, loaded by TMA into 128 B swizzled K-major tiles.
//
// Numerics match tiered_decode: weights decode to exact f16 (code - 8) * 2^-14,
// activations are f16 under a power-of-two per-row scale (exact for bf16),
// each 32-wide group's products accumulate exactly in fp32 and its bf16 scale
// applies in fp32. A round of the main loop is one group: two k16 steps into a
// fresh accumulator, then scaled into the running sum.
//
// Milestone 1: every expert multiplies the same X, one CTA per (block, expert).

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

#include <cstdint>
#include <type_traits>

namespace tiered_prefill {

constexpr int ROWS = 256;    // W rows per CTA: four 64-row Marlin tiles
constexpr int CK = 64;       // K per pipeline stage: small stages, deep ring
constexpr int KT = CK / 16;  // k16 steps per stage
constexpr int GR = CK / 32;  // scale groups (= rounds) per stage
constexpr int W_BYTES = KT * ROWS * 8;  // 8 KB: [kt][tile][lane][4 words]
constexpr int S_BYTES = GR * ROWS * 2;  // 1 KB: [group][tile][64 bf16]
constexpr int XBOX = 64;                // K elements per 128 B swizzle atom
// One warpgroup; lane 0 of warp 0 also issues the TMA loads. Without a
// separate producer warp, 2 CTAs per SM may use up to 255 registers.
constexpr int THREADS = 128;
constexpr int SMEM_SM = 227 * 1024;
constexpr int PLACEMENT_EXP = 14;  // decoded weights are value * 2^-14
constexpr int MAX_NT = 32;         // tokens per launch; larger N is chunked
constexpr int PREP_THREADS = 256;
static_assert(GR % 2 == 0, "register sets alternate within a stage");

// Several CTAs per SM: ptxas retires each CTA's wgmmas before it decodes the
// next round, so the SM overlaps one CTA's decode with another's wgmmas.
#ifndef TP_CTAS16
  #define TP_CTAS16 3
#endif
#ifndef TP_CTAS32
  #define TP_CTAS32 2
#endif
#ifndef TP_DEFER16
  #define TP_DEFER16 0
#endif

template <int NT>
struct Cfg {
  static constexpr int CTAS = NT <= 16 ? TP_CTAS16 : TP_CTAS32;
  static constexpr int X_BYTES = CK / XBOX * NT * 128;
  static constexpr int STAGE = W_BYTES + S_BYTES + X_BYTES;
  static constexpr int BUDGET = SMEM_SM / CTAS - 2048 - 1024;  // + barriers
  static constexpr int STAGES = BUDGET / STAGE < 8 ? BUDGET / STAGE : 8;
  static_assert(STAGES >= 2, "too little shared memory per CTA");
  static constexpr int SMEM = 1024 + STAGES * STAGE;  // barriers + 1 KB align
  static_assert(STAGE % 1024 == 0, "swizzled tiles need 1 KB alignment");
};

struct Params {
  CUtensorMap w;     // [E][K/16][F*2] int32 as uint64 {F, K/16, E}
  CUtensorMap s;     // [E][K/32][F] bf16 {F, K/32, E}
  CUtensorMap x;     // [n][K] f16 rows of this chunk, 128 B swizzle
  const float* xs;   // [n] per-row scale (times 2^PLACEMENT_EXP)
  __nv_bfloat16* y;  // [E][N][F], this chunk's rows start at y_row0
  int n, k, f, y_rows, y_row0;
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
__device__ __forceinline__ uint4 lds128(uint32_t addr) {
  uint4 v;
  asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "r"(addr));
  return v;
}

__device__ __forceinline__ void wgmma_fence() {
  asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
}
__device__ __forceinline__ void wgmma_commit() {
  asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
}
template <int N>
__device__ __forceinline__ void wgmma_wait() {
  asm volatile("wgmma.wait_group.sync.aligned %0;" ::"n"(N) : "memory");
}

// K-major operand in 128 B swizzled rows: 8-row atoms 1 KB apart.
__device__ __forceinline__ uint64_t desc_sw128(const void* p) {
  const uint64_t addr = smem_u32(p);
  return ((addr & 0x3FFFF) >> 4) | (uint64_t(1024 >> 4) << 32) |
         (uint64_t(1) << 62);
}

// D[64 x N] (+)= A[64 x 16] (registers) * B[16 x N] (smem, K-major), f16 in,
// f32 out; acc == 0 overwrites D.
template <int N>
struct Wgmma;
template <>
struct Wgmma<16> {
  __device__ __forceinline__ static void run(float* d, const uint32_t* a,
                                             uint64_t b, int acc) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %13, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, %12, p, 1, 1, 0;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]),
          "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(b), "r"(acc));
  }
};
template <>
struct Wgmma<32> {
  __device__ __forceinline__ static void run(float* d, const uint32_t* a,
                                             uint64_t b, int acc) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %21, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, "
        "{%16,%17,%18,%19}, %20, p, 1, 1, 0;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]),
          "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]),
          "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]),
          "+f"(d[15])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(b), "r"(acc));
  }
};

// After a wgmma wait: redefine its outputs so ptxas stops tracking them as
// in-flight GMMA results (it otherwise injects a wait before their next use).
template <int N>
__device__ __forceinline__ void fence_operand(float (&r)[N]) {
#pragma unroll
  for (int i = 0; i < N; ++i) asm volatile("" : "+f"(r[i])::"memory");
}

// (a & mask) | magic in one LOP3: with both constants as immediates the
// compiler emits two.
__device__ __forceinline__ uint32_t and_or(uint32_t a, uint32_t mask,
                                           uint32_t magic) {
  uint32_t out;
  asm("lop3.b32 %0, %1, %2, %3, 0xEA;"
      : "=r"(out)
      : "r"(a), "r"(mask), "r"(magic));
  return out;
}

struct Decode {
  uint32_t lo_mask, hi_mask, magic;  // in registers: one LOP3 per pair
};

// One Marlin word -> the four f16x2 A registers of a 16-row block, exact
// (code - 8) * 2^-14. Nibbles, low to high: (g,k0) (g,k8) (g+8,k0) (g+8,k8)
// (g,k1) (g,k9) (g+8,k1) (g+8,k9). A low nibble under 0x6400 is 1024 + code,
// a high one 1024 + 16 code, so only the second byte pair needs a shift.
__device__ __forceinline__ void decode(uint32_t w, const Decode& c,
                                       uint32_t* a) {
  const __half2 unit_lo = __halves2half2(__ushort_as_half(0x0400),
                                         __ushort_as_half(0x0400));  // 2^-14
  const __half2 bias_lo = __halves2half2(
      __ushort_as_half(0xAC08), __ushort_as_half(0xAC08));  // -1032 * 2^-14
  const __half2 unit_hi = __halves2half2(__ushort_as_half(0x0040),
                                         __ushort_as_half(0x0040));  // 2^-18
  const __half2 bias_hi = __halves2half2(
      __ushort_as_half(0x9C80), __ushort_as_half(0x9C80));  // -1152 * 2^-18
  const uint32_t w8 = w >> 8;
  const uint32_t t[4] = {and_or(w, c.lo_mask, c.magic),    // rows g,   k0 k1
                         and_or(w8, c.lo_mask, c.magic),   // rows g+8, k0 k1
                         and_or(w, c.hi_mask, c.magic),    // rows g,   k8 k9
                         and_or(w8, c.hi_mask, c.magic)};  // rows g+8, k8 k9
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const __half2 v =
        __hfma2(*reinterpret_cast<const __half2*>(&t[i]),
                i < 2 ? unit_lo : unit_hi, i < 2 ? bias_lo : bias_hi);
    a[i] = *reinterpret_cast<const uint32_t*>(&v);
  }
}

// ---------------------------------------------------------------- prep
// Row t of x (bf16) -> f16 with its largest magnitude at most 2^13; the power
// of two (times the weights' 2^14) goes to xs[t]. As in tiered_decode.
__global__ void prep_kernel(const __nv_bfloat16* __restrict__ x,
                            __half* __restrict__ xh, float* __restrict__ xs,
                            int k) {
  __shared__ float red[PREP_THREADS / 32];
  const int t = blockIdx.x;
  const __nv_bfloat162* row =
      reinterpret_cast<const __nv_bfloat162*>(x + static_cast<size_t>(t) * k);
  float m = 0.f;
  for (int i = threadIdx.x; i < k / 2; i += PREP_THREADS) {
    const float2 v = __bfloat1622float2(row[i]);
    m = fmaxf(m, fmaxf(fabsf(v.x), fabsf(v.y)));
  }
  for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
  if (threadIdx.x % 32 == 0) red[threadIdx.x / 32] = m;
  __syncthreads();
  m = 0.f;
#pragma unroll
  for (int i = 0; i < PREP_THREADS / 32; ++i) m = fmaxf(m, red[i]);
  const int e = m > 0.f ? static_cast<int>(ceilf(log2f(m))) - 13 : 0;
  const float inv = exp2f(static_cast<float>(-e));
  if (threadIdx.x == 0) xs[t] = exp2f(static_cast<float>(e + PLACEMENT_EXP));
  __half2* out = reinterpret_cast<__half2*>(xh + static_cast<size_t>(t) * k);
  for (int i = threadIdx.x; i < k / 2; i += PREP_THREADS) {
    const float2 v = __bfloat1622float2(row[i]);
    out[i] = __floats2half2_rn(v.x * inv, v.y * inv);
  }
}

// ---------------------------------------------------------------- GEMM
template <int NT, int MODE = 0>
__global__ void __launch_bounds__(THREADS, Cfg<NT>::CTAS)
    gemm_kernel(const __grid_constant__ Params p) {
  using C = Cfg<NT>;
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  // stages start on a 1 KB boundary; barriers live in the first KB
  unsigned char* smem = reinterpret_cast<unsigned char*>(
      (reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  uint64_t* full = reinterpret_cast<uint64_t*>(smem);
  unsigned char* ring = smem + 1024;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int block = blockIdx.x, expert = blockIdx.y;
  const int stages_total = p.k / CK;

  const bool issuer = threadIdx.x == 0;
  auto issue = [&](int it) {  // TMA for stage `it` into its ring slot
    const int s = it % C::STAGES;
    unsigned char* dst = ring + static_cast<size_t>(s) * C::STAGE;
    mbar_expect_tx(&full[s], C::STAGE);
    tma_3d(dst, &p.w, block * ROWS, it * KT, expert, &full[s]);
    tma_3d(dst + W_BYTES, &p.s, block * ROWS, it * GR, expert, &full[s]);
#pragma unroll
    for (int b = 0; b < CK / XBOX; ++b)
      tma_2d(dst + W_BYTES + S_BYTES + b * NT * 128, &p.x, it * CK + b * XBOX,
             0, &full[s]);
  };
  if (issuer) {
    for (int s = 0; s < C::STAGES; ++s) mbar_init(&full[s], 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    const int first = MODE == 2 ? 1 : stages_total;
    for (int it = 0; it < (first < C::STAGES ? first : C::STAGES); ++it)
      issue(it);
  }
  __syncthreads();
  // A stage's slot is free once its last wgmma retired: wgmmas are
  // warpgroup-collective, so every warp's reads of the slot came before.
  auto refill = [&](int it) {
    if (MODE != 2 && issuer && it + C::STAGES < stages_total)
      issue(it + C::STAGES);
  };

  if constexpr (MODE == 1) {  // loads only
    for (int it = 0; it < stages_total; ++it) {
      mbar_wait(&full[it % C::STAGES], (it / C::STAGES) & 1);
      refill(it);
    }
    return;
  }

  // consumer warpgroup: warp w owns Marlin tile w of the block
  const int g = lane / 4, tq = lane % 4;
  // DEFER: round r's partial sums are scaled into acc while round r+1's
  // wgmmas run (two part[] sets); otherwise right after round r retires.
  constexpr bool DEFER = NT <= 16 && TP_DEFER16;
  constexpr int PARTS = DEFER ? 2 : 1;
  float acc[4][NT / 2], part[PARTS][4][NT / 2];
#pragma unroll
  for (int j = 0; j < 4; ++j)
#pragma unroll
    for (int i = 0; i < NT / 2; ++i) acc[j][i] = 0.f;

  Decode dc{0x000F000Fu, 0x00F000F0u, 0x64006400u};
  asm volatile("" : "+r"(dc.lo_mask), "+r"(dc.hi_mask), "+r"(dc.magic));
  const uint32_t ring_u32 = smem_u32(ring);
  // Round r of a stage is scale group r: its two k16 steps go into part[]
  // while the next round decodes into the other register set; each decode
  // starts after every earlier wgmma retired (wait<0>), so ptxas need not
  // serialize. Rounds run across stage boundaries.
  uint32_t a[2][2][4][4];
  uint32_t sw[2][4];  // per round: block j's rows g (lo), g+8 (hi) as bf16
  auto decode_round = [&](int it, int r, int set) {
    const uint32_t st = ring_u32 + (it % C::STAGES) * C::STAGE;
    // row g + 8i of tile w at bf16 8g + i: block j's rows g, g+8 are the
    // word at 8g + 2j
    const uint4 sc = lds128(st + W_BYTES + r * 512 + warp * 128 + g * 16);
    sw[set][0] = sc.x;
    sw[set][1] = sc.y;
    sw[set][2] = sc.z;
    sw[set][3] = sc.w;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const uint4 wv = lds128(st + (2 * r + i) * 2048 + warp * 512 + lane * 16);
      const uint32_t words[4] = {wv.x, wv.y, wv.z, wv.w};
#pragma unroll
      for (int j = 0; j < 4; ++j) decode(words[j], dc, a[set][i][j]);
    }
  };
  // MODE 2 (compute only): every stage reuses stage 0's resident data
  auto stage_of = [&](int it) { return MODE == 2 ? 0 : it; };

  auto scale_into_acc = [&](const float (&pt)[4][NT / 2], const uint32_t* swr) {
  // D fragment regs 4i + {0,1} are row g, {2,3} row g+8
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float s_g = __uint_as_float(swr[j] << 16);
      const float s_g8 = __uint_as_float(swr[j] & 0xFFFF0000u);
#pragma unroll
      for (int e = 0; e < NT / 2; ++e)
        acc[j][e] = fmaf((e & 2) ? s_g8 : s_g, pt[j][e], acc[j][e]);
    }
  };

  mbar_wait(&full[0], 0);
  decode_round(0, 0, 0);
  if constexpr (DEFER) {
    // the first deferred scaling adds zero: no branch for ptxas to doubt
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      sw[1][j] = 0u;
#pragma unroll
      for (int e = 0; e < NT / 2; ++e) part[DEFER ? 1 : 0][j][e] = 0.f;
    }
  }
  for (int it = 0; it < stages_total; ++it) {
    const int s = stage_of(it) % C::STAGES;
    const unsigned char* xs =
        ring + static_cast<size_t>(s) * C::STAGE + W_BYTES + S_BYTES;
#pragma unroll
    for (int r = 0; r < GR; ++r) {
      const int set = r & 1;
      float (&pt)[4][NT / 2] = part[DEFER ? set : 0];
      wgmma_fence();
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const int kt = 2 * r + i;
        const uint64_t b = desc_sw128(xs + (kt / 4) * NT * 128 + (kt % 4) * 32);
#pragma unroll
        for (int j = 0; j < 4; ++j) Wgmma<NT>::run(pt[j], a[set][i][j], b, i);
      }
      wgmma_commit();
      if constexpr (DEFER) {
        // only this round's group may still run: tells ptxas the other
        // part set is settled (FA3's pattern), so it injects no full wait
        wgmma_wait<1>();
        scale_into_acc(part[set ^ 1], sw[set ^ 1]);
      }
      if (r + 1 < GR) {
        decode_round(stage_of(it), r + 1, set ^ 1);
      } else if (it + 1 < stages_total) {
        if (MODE != 2)
          mbar_wait(&full[(it + 1) % C::STAGES], ((it + 1) / C::STAGES) & 1);
        decode_round(stage_of(it + 1), 0, set ^ 1);
      }
      wgmma_wait<0>();
#pragma unroll
      for (int j = 0; j < 4; ++j) fence_operand(pt[j]);
      if constexpr (!DEFER) scale_into_acc(pt, sw[set]);
    }
    refill(it);
  }
  if constexpr (DEFER) scale_into_acc(part[(GR - 1) & 1], sw[(GR - 1) & 1]);

  // D fragment: rows 16*warp + g (+8) of wgmma j, i.e. tile `warp`, block j;
  // columns 8i + 2tq (+1) are tokens.
  const int f0 = block * ROWS + warp * 64;
#pragma unroll
  for (int i = 0; i < NT / 8; ++i)
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      const int n = 8 * i + 2 * tq + c;
      if (n >= p.n) continue;
      const float xs_n = p.xs[n];
      __nv_bfloat16* y =
          p.y + (static_cast<size_t>(expert) * p.y_rows + p.y_row0 + n) * p.f;
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int h = 0; h < 2; ++h)
          y[f0 + 16 * j + g + 8 * h] =
              __float2bfloat16_rn(acc[j][i * 4 + 2 * h + c] * xs_n);
    }
}

// ---------------------------------------------------------------- host entry
static CUtensorMap make_map_3d(const torch::Tensor& t, CUtensorMapDataType type,
                               uint64_t inner, uint32_t box0, uint32_t box1) {
  // t is [E][rows][cols]; the innermost dimension holds `inner` elements
  CUtensorMap map;
  const cuuint64_t dims[3] = {inner, static_cast<cuuint64_t>(t.size(1)),
                              static_cast<cuuint64_t>(t.size(0))};
  const cuuint64_t strides[2] = {
      static_cast<cuuint64_t>(t.stride(1)) * t.element_size(),
      static_cast<cuuint64_t>(t.stride(0)) * t.element_size()};
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

static CUtensorMap make_x_map(const torch::Tensor& x, uint32_t rows) {
  CUtensorMap map;
  const cuuint64_t dims[2] = {static_cast<cuuint64_t>(x.size(1)),
                              static_cast<cuuint64_t>(x.size(0))};
  const cuuint64_t strides[1] = {static_cast<cuuint64_t>(x.stride(0)) * 2};
  const cuuint32_t box[2] = {XBOX, rows};
  const cuuint32_t unit[2] = {1, 1};
  const CUresult r = cuTensorMapEncodeTiled(
      &map, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2, x.data_ptr(), dims, strides,
      box, unit, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS,
              "cuTensorMapEncodeTiled failed: ", static_cast<int>(r));
  return map;
}

template <int NT, int MODE>
static void launch(const Params& p, int blocks, int experts,
                   cudaStream_t stream) {
  static bool attr = false;
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        gemm_kernel<NT, MODE>, cudaFuncAttributeMaxDynamicSharedMemorySize,
        Cfg<NT>::SMEM + 1024));
    attr = true;
  }
  gemm_kernel<NT, MODE>
      <<<dim3(blocks, experts), THREADS, Cfg<NT>::SMEM + 1024, stream>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int NT>
static void launch_mode(const Params& p, int blocks, int experts, int mode,
                        cudaStream_t stream) {
  if (mode == 1)
    launch<NT, 1>(p, blocks, experts, stream);
  else if (mode == 2)
    launch<NT, 2>(p, blocks, experts, stream);
  else
    launch<NT, 0>(p, blocks, experts, stream);
}

// y[e] = x @ dequant(w[e])^T for every expert e. w: [E][K/16][F*2] int32
// (Marlin), s: [E][K/32][F] bf16 (Marlin-permuted), x: [N][K] bf16,
// y: [E][N][F] bf16. mode 1 / 2: loads-only / compute-only probes.
void dense(torch::Tensor y, torch::Tensor x, torch::Tensor w, torch::Tensor s,
           int64_t mode) {
  TORCH_CHECK(
      x.scalar_type() == at::kBFloat16 && x.is_contiguous() && x.dim() == 2,
      "x must be contiguous [N, K] bf16");
  TORCH_CHECK(w.scalar_type() == at::kInt && w.is_contiguous() && w.dim() == 3,
              "w must be contiguous [E, K/16, F*2] int32");
  TORCH_CHECK(s.scalar_type() == at::kBFloat16 && s.is_contiguous(),
              "s must be contiguous [E, K/32, F] bf16");
  const int n = static_cast<int>(x.size(0)), k = static_cast<int>(x.size(1));
  const int e = static_cast<int>(w.size(0)),
            f = static_cast<int>(w.size(2) / 2);
  TORCH_CHECK(w.size(1) == k / 16 && s.size(1) == k / 32 && s.size(2) == f,
              "w / s shapes do not match x");
  TORCH_CHECK(k % CK == 0 && f % ROWS == 0, "K % 64 and F % 256 must be 0");
  TORCH_CHECK(y.size(0) == e && y.size(1) == n && y.size(2) == f &&
                  y.scalar_type() == at::kBFloat16 && y.is_contiguous(),
              "y must be contiguous [E, N, F] bf16");
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto xh = torch::empty({n, k}, x.options().dtype(at::kHalf));
  auto xs = torch::empty({n}, x.options().dtype(at::kFloat));
  prep_kernel<<<n, PREP_THREADS, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
      reinterpret_cast<__half*>(xh.data_ptr()), xs.data_ptr<float>(), k);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  for (int n0 = 0; n0 < n; n0 += MAX_NT) {
    const int nc = n - n0 < MAX_NT ? n - n0 : MAX_NT;
    const int nt = nc <= 16 ? 16 : 32;
    auto xc = xh.narrow(0, n0, nc);
    Params p{};
    p.w = make_map_3d(w, CU_TENSOR_MAP_DATA_TYPE_UINT64, f, ROWS, KT);
    p.s = make_map_3d(s, CU_TENSOR_MAP_DATA_TYPE_UINT16, f, ROWS, GR);
    p.x = make_x_map(xc, nt);
    p.xs = xs.data_ptr<float>() + n0;
    p.y = reinterpret_cast<__nv_bfloat16*>(y.data_ptr());
    p.n = nc;
    p.k = k;
    p.f = f;
    p.y_rows = n;
    p.y_row0 = n0;
    if (nt == 16)
      launch_mode<16>(p, f / ROWS, e, static_cast<int>(mode), stream);
    else
      launch_mode<32>(p, f / ROWS, e, static_cast<int>(mode), stream);
  }
}

}  // namespace tiered_prefill

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dense", &tiered_prefill::dense,
        "W4A16 (Marlin layout) GEMM via f16 wgmma, every expert on the same x");
}

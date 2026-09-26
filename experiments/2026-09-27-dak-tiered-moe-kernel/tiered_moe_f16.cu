// Tiered mxfp4 MoE GEMM, f16 tensor-core path via e4m3 unpack (v5).
//
// Hopper has no native FP8 mma.sync (ptxas emulates it; see v4), but it has
// F2FP.F16.E4M3.UNPACK_B: two e4m3 bytes -> f16x2 in one instruction. So:
//
//  * e2m1 is packed as e4m3 bit placements (value * 2^-6, exact, zero and the
//    0.5 subnormal included): per k16 block a thread's 32-bit word carries its
//    row-g values in slot a (bits {7,4,3,2} of each byte, 1 AND) and its
//    row-g+8 values in slot b (bits {6,1,0,5}, a few shifts), in the order
//    {k 2tq, 2tq+1, 2tq+8, 2tq+9}.
//  * cvt.rn.f16x2.e4m3x2 turns each byte pair into an exact f16x2 A register,
//    and a native f16 mma.m16n8k16 (f32 accumulate) consumes it.
//  * Block scales are applied in fp32 once per 32-K group (2 mma):
//    acc += 2^(e-121) * mma_group.
//  * Activations are f16 with a per-token power-of-two scale (10-bit mantissa,
//    more precise than bf16); B layout per token row, per k16 block, per tq:
//    8 bytes = {b0 = k 2tq..2tq+1, b1 = k 2tq+8..2tq+9}: one LDS.64 per mma.
//
//   tiered_moe_f16 check | bench <w13|w2> hot cold cold_ctas stages ntok(0=dist)

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <random>
#include <algorithm>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  printf("CUDA error %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

constexpr int TILE_N = 64;
#ifndef CHUNK_K_DEF
#define CHUNK_K_DEF 1024
#endif
constexpr int CHUNK_K = CHUNK_K_DEF;
constexpr int KB = CHUNK_K / 16;                      // k16 blocks per chunk
constexpr int GROUPS = CHUNK_K / 32;                  // e8m0 groups per chunk
constexpr int W_BYTES = TILE_N * CHUNK_K / 2;         // 32768
constexpr int S_BYTES = TILE_N * GROUPS;              // 2048
constexpr int CHUNK_BYTES = W_BYTES + S_BYTES;        // 34816
constexpr int MAX_TOK = 8;
constexpr int XROW_BYTES = CHUNK_K * 2;               // f16 per chunk per token
constexpr int XROW_STRIDE = XROW_BYTES + 32;          // rows land 8 banks apart
constexpr int X_BYTES = MAX_TOK * XROW_STRIDE;
constexpr int STAGE_BYTES = CHUNK_BYTES + X_BYTES;    // 51200
#ifndef CONSUMER_WARPS_DEF
#define CONSUMER_WARPS_DEF 8
#endif
constexpr int CONSUMER_WARPS = CONSUMER_WARPS_DEF;
constexpr int KSPLIT = CONSUMER_WARPS / 4;
constexpr int THREADS = (CONSUMER_WARPS + 1) * 32;
constexpr int RED_FLOATS = (KSPLIT - 1) * 4 * 32 * 4;
constexpr int EX_FLOATS = 64 * 8;
constexpr int SMEM_HEAD = 256 + (RED_FLOATS + EX_FLOATS) * 4;
#ifndef DIAG
#define DIAG 0
#endif

#ifndef GEMV_PATH
#define GEMV_PATH 1
#endif
#ifndef ABL
#define ABL 0
#endif
#ifndef PROF
#define PROF 0
#endif
// PROF=1: clock64 cycles per CTA: [consumer wait full, consumer total, producer wait empty, producer total]
__device__ unsigned long long g_prof[8];  // [4] consumer epilogue, [5] CTAs, [6] consumer ns
__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t;
}

enum Epilogue { RAW = 0, ACT = 1, WSUM = 2 };

struct Expert {
  const uint8_t* w;
  int ntok;
  int tok[MAX_TOK];
  int route[MAX_TOK];
  float wt[MAX_TOK];
};

struct Params {
  const Expert* hot; int n_hot;
  const Expert* cold; int n_cold;
  int cold_ctas;
  int N, K;
  const uint8_t* x8;       // [rows][K/16 kb][4 tq][8 B] f16 B fragments
  const float* xscale;     // [rows] power-of-two activation scale
  float* y_raw;
  __nv_bfloat16* y_act;
  float* y_sum;
  int stages;
};

// ---------------------------------------------------------------- PTX helpers
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(bar)), "r"(count));
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(bar)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
  asm volatile("{\n .reg .pred p;\n WAIT_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
               " @!p bra WAIT_%=;\n}\n" :: "r"(smem_u32(bar)), "r"(parity) : "memory");
}
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void consumer_sync() {
  asm volatile("bar.sync 1, %0;" :: "n"(CONSUMER_WARPS * 32) : "memory");
}
__device__ __forceinline__ void mma_f16(float* d, const uint32_t* a, uint32_t b0, uint32_t b1, const float* c) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1),
        "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}
// Two e4m3 in the low / high 16 bits -> f16x2.
__device__ __forceinline__ uint32_t e4m3x2_lo(uint32_t v) {
  uint32_t out;
  asm("{ .reg .b16 lo, hi; mov.b32 {lo, hi}, %1; cvt.rn.f16x2.e4m3x2 %0, lo; }" : "=r"(out) : "r"(v));
  return out;
}
__device__ __forceinline__ uint32_t e4m3x2_hi(uint32_t v) {
  uint32_t out;
  asm("{ .reg .b16 lo, hi; mov.b32 {lo, hi}, %1; cvt.rn.f16x2.e4m3x2 %0, hi; }" : "=r"(out) : "r"(v));
  return out;
}
__device__ __forceinline__ __half2 u32_h2(uint32_t v) { return *reinterpret_cast<__half2*>(&v); }
__device__ __forceinline__ uint32_t reg_a(uint32_t w) { return w & 0x9C9C9C9Cu; }
__device__ __forceinline__ uint32_t reg_b(uint32_t w) {
  return ((w << 1) & 0x80808080u) | ((w << 3) & 0x18181818u) | ((w >> 3) & 0x04040404u);
}
// fp32 2^(e-127) * 2^6: undo the e4m3 placement's 2^-6.
__device__ __forceinline__ float group_scale(uint32_t e) { return __uint_as_float((e + 6u) << 23); }

// ---------------------------------------------------------------- the kernel
template <int EPI>
__global__ void __launch_bounds__(THREADS, 1) tiered_moe_f16_kernel(Params p) {
  extern __shared__ __align__(128) unsigned char smem[];
  const int stages = p.stages;
  uint64_t* full = reinterpret_cast<uint64_t*>(smem);
  uint64_t* empty = full + stages;
  float* red = reinterpret_cast<float*>(smem + 256);
  float* ex = red + RED_FLOATS;
  unsigned char* ring = smem + SMEM_HEAD;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;

  const bool cold = blockIdx.x < p.cold_ctas;
  const Expert* experts = cold ? p.cold : p.hot;
  const int n_exp = cold ? p.n_cold : p.n_hot;
  const int role_ctas = cold ? p.cold_ctas : gridDim.x - p.cold_ctas;
  const int role_idx = cold ? blockIdx.x : blockIdx.x - p.cold_ctas;
  const int tiles_per_exp = p.N / TILE_N;
  const int chunks = p.K / CHUNK_K;
  const int n_tiles = n_exp * tiles_per_exp;
  const size_t tile_bytes = static_cast<size_t>(chunks) * CHUNK_BYTES;
  const size_t xrow_bytes = static_cast<size_t>(p.K) * 2;

  if (threadIdx.x == 0) {
    for (int s = 0; s < stages; ++s) { mbar_init(&full[s], 1); mbar_init(&empty[s], CONSUMER_WARPS); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (role_ctas <= 0 || n_tiles == 0) return;

  if (warp == CONSUMER_WARPS) {
    if (lane == 0) {
      long long pw = 0, pt0 = clock64();
      int it = 0;
      for (int t = role_idx; t < n_tiles; t += role_ctas) {
        const Expert& e = experts[t / tiles_per_exp];
        const uint8_t* tile = e.w + static_cast<size_t>(t % tiles_per_exp) * tile_bytes;
        for (int c = 0; c < chunks; ++c, ++it) {
          int s = it % stages;
          if (it >= stages) {
            long long w0 = PROF ? clock64() : 0;
            mbar_wait(&empty[s], ((it / stages) - 1) & 1);
            if (PROF) pw += clock64() - w0;
          }
          if (DIAG == 2) { mbar_arrive(&full[s]); continue; }
          unsigned char* dst = ring + static_cast<size_t>(s) * STAGE_BYTES;
          mbar_expect_tx(&full[s], CHUNK_BYTES + e.ntok * XROW_BYTES);
          bulk_g2s(dst, tile + static_cast<size_t>(c) * CHUNK_BYTES, CHUNK_BYTES, &full[s]);
          for (int j = 0; j < e.ntok; ++j)
            bulk_g2s(dst + CHUNK_BYTES + j * XROW_STRIDE,
                     p.x8 + static_cast<size_t>(e.tok[j]) * xrow_bytes + static_cast<size_t>(c) * XROW_BYTES,
                     XROW_BYTES, &full[s]);
        }
      }
      if (PROF) { atomicAdd(&g_prof[2], (unsigned long long)pw); atomicAdd(&g_prof[3], (unsigned long long)(clock64() - pt0)); }
    }
    return;
  }

  const int mb = warp % 4, slice = warp / 4;
  const int g = lane / 4, tq = lane % 4;
  constexpr int GS = GROUPS / KSPLIT;         // groups per warp per stage
  int it = 0;
  long long cw = 0, ce = 0, ct0 = clock64();
  unsigned long long gt0 = PROF ? gtimer() : 0;
  for (int t = role_idx; t < n_tiles; t += role_ctas) {
    const Expert& e = experts[t / tiles_per_exp];
    const int ntile = t % tiles_per_exp;
    const int ntok = e.ntok;
    float acc[4] = {};
    for (int c = 0; c < chunks; ++c, ++it) {
      int s = it % stages;
      {
        long long w0 = PROF ? clock64() : 0;
        mbar_wait(&full[s], (it / stages) & 1);
        if (PROF) cw += clock64() - w0;
      }
      if (DIAG == 1) { __syncwarp(); if (lane == 0) mbar_arrive(&empty[s]); continue; }
      const unsigned char* st = ring + static_cast<size_t>(s) * STAGE_BYTES;
      const int g0 = slice * GS;
      const uint32_t* wq = reinterpret_cast<const uint32_t*>(st) + (mb * KB + 2 * g0) * 32 + lane;
      const uint16_t* sc = reinterpret_cast<const uint16_t*>(st + W_BYTES) + (mb * GROUPS + g0) * 8 + g;
      const uint2* xs = reinterpret_cast<const uint2*>(st + CHUNK_BYTES + g * XROW_STRIDE) + 2 * g0 * 4 + tq;
      const bool has_tok = g < ntok;
      if (GEMV_PATH && ntok == 1) {
        // One token: f16x2 FMAs instead of an mma with 7 empty columns (the
        // tensor pipe is what pushes the chip into its power cap). Each lane
        // keeps rows g and g+8 over its k pairs; f16 sums only span one group.
        const uint2* x0 = reinterpret_cast<const uint2*>(st + CHUNK_BYTES) + 2 * g0 * 4 + tq;
#pragma unroll
        for (int j = 0; j < GS; ++j) {
          uint32_t sp = sc[j * 8];
          __half2 h0 = __float2half2_rn(0.f), h1 = h0;
#pragma unroll
          for (int u = 0; u < 2; ++u) {
            uint32_t w = wq[(2 * j + u) * 32];
            uint2 xb = x0[(2 * j + u) * 4];
            uint32_t ra = reg_a(w), rb = reg_b(w);
            h0 = __hfma2(u32_h2(e4m3x2_lo(ra)), u32_h2(xb.x), h0);
            h1 = __hfma2(u32_h2(e4m3x2_lo(rb)), u32_h2(xb.x), h1);
            h0 = __hfma2(u32_h2(e4m3x2_hi(ra)), u32_h2(xb.y), h0);
            h1 = __hfma2(u32_h2(e4m3x2_hi(rb)), u32_h2(xb.y), h1);
          }
          float2 f0 = __half22float2(h0), f1 = __half22float2(h1);
          acc[0] = fmaf(group_scale(sp & 0xFFu), f0.x + f0.y, acc[0]);
          acc[2] = fmaf(group_scale(sp >> 8), f1.x + f1.y, acc[2]);
        }
        __syncwarp();
        if (lane == 0) mbar_arrive(&empty[s]);
        continue;
      }
#pragma unroll
      for (int j = 0; j < GS; ++j) {
        uint32_t sp = sc[j * 8];
        const float zero[4] = {0.f, 0.f, 0.f, 0.f};
        float d[4];
#pragma unroll
        for (int u = 0; u < 2; ++u) {
          uint32_t w = wq[(2 * j + u) * 32];
          uint2 xb = has_tok ? xs[(2 * j + u) * 4] : make_uint2(0u, 0u);
#if ABL == 3
          uint32_t a[4] = {w, w ^ xb.x, w + 1, w ^ 0x1234u};
#elif ABL == 2
          uint32_t ra = reg_a(w), rb = reg_b(w);
          uint32_t a[4] = {ra, rb, ra ^ rb, rb + ra};
#else
          uint32_t ra = reg_a(w), rb = reg_b(w);
          uint32_t a[4] = {e4m3x2_lo(ra), e4m3x2_lo(rb), e4m3x2_hi(ra), e4m3x2_hi(rb)};
#endif
#if ABL == 1
          float fz = __uint_as_float((a[0] ^ a[1] ^ a[2] ^ a[3] ^ xb.x ^ xb.y) & 0x3fffffffu);
          if (u) { d[0] += fz; d[1] -= fz; d[2] += 2.f * fz; d[3] *= fz; }
          else { d[0] = fz; d[1] = -fz; d[2] = fz; d[3] = 1.f; }
#else
          mma_f16(d, a, xb.x, xb.y, u ? d : zero);
#endif
        }
        float s0 = group_scale(sp & 0xFFu), s1 = group_scale(sp >> 8);
        acc[0] = fmaf(s0, d[0], acc[0]);
        acc[1] = fmaf(s0, d[1], acc[1]);
        acc[2] = fmaf(s1, d[2], acc[2]);
        acc[3] = fmaf(s1, d[3], acc[3]);
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(&empty[s]);
    }
    long long ep0 = PROF ? clock64() : 0;
    if (GEMV_PATH && ntok == 1) {
#pragma unroll
      for (int i = 0; i < 4; i += 2) {
        acc[i] += __shfl_xor_sync(0xffffffffu, acc[i], 1);
        acc[i] += __shfl_xor_sync(0xffffffffu, acc[i], 2);
      }
    }
    float r[4] = {acc[0], acc[1], acc[2], acc[3]};
    if (slice > 0) {
      float* slot = red + (((slice - 1) * 4 + mb) * 32 + lane) * 4;
#pragma unroll
      for (int i = 0; i < 4; ++i) slot[i] = r[i];
    }
    consumer_sync();
    if (slice == 0) {
#pragma unroll
      for (int h = 0; h < KSPLIT - 1; ++h) {
        const float* slot = red + ((h * 4 + mb) * 32 + lane) * 4;
#pragma unroll
        for (int i = 0; i < 4; ++i) r[i] += slot[i];
      }
      const int row0 = mb * 16 + g;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        int tok = 2 * tq + (i & 1);
        int row = row0 + (i >> 1) * 8;
        if (tok < ntok) {
          float v = r[i] * p.xscale[e.tok[tok]];
          int n = ntile * TILE_N + row;
          if constexpr (EPI == RAW) p.y_raw[static_cast<size_t>(e.route[tok]) * p.N + n] = v;
          else if constexpr (EPI == WSUM) atomicAdd(&p.y_sum[static_cast<size_t>(e.tok[tok]) * p.N + n], e.wt[tok] * v);
          else ex[row * 8 + tok] = v;
        }
      }
    }
    if constexpr (EPI == ACT) {
      consumer_sync();
      for (int idx = threadIdx.x; idx < 32 * ntok; idx += CONSUMER_WARPS * 32) {
        int row = idx % 32, tok = idx / 32;
        float gt = ex[row * 8 + tok], up = ex[(row + 32) * 8 + tok];
        p.y_act[static_cast<size_t>(e.route[tok]) * (p.N / 2) + ntile * 32 + row] =
            __float2bfloat16(gt / (1.f + __expf(-gt)) * up);
      }
    }
    consumer_sync();
    if (PROF) ce += clock64() - ep0;
  }
  if (PROF && threadIdx.x == 0) {
    atomicAdd(&g_prof[0], (unsigned long long)cw); atomicAdd(&g_prof[1], (unsigned long long)(clock64() - ct0));
    atomicAdd(&g_prof[4], (unsigned long long)ce); atomicAdd(&g_prof[5], 1ull);
    atomicAdd(&g_prof[6], gtimer() - gt0);
  }
}

// bf16 rows -> f16 B-fragment layout + per-row power-of-two scale.
// One block per row. Scale t: x / 2^t puts max|x| at <= 256 (e4m3 max 448).
__global__ void prep_kernel(const __nv_bfloat16* x, int K, uint8_t* x8, float* xscale) {
  const int row = blockIdx.x;
  const __nv_bfloat16* xr = x + static_cast<size_t>(row) * K;
  __shared__ float red[32];
  float m = 0.f;
  for (int k = threadIdx.x; k < K; k += blockDim.x) m = fmaxf(m, fabsf(__bfloat162float(xr[k])));
  for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
  if (threadIdx.x % 32 == 0) red[threadIdx.x / 32] = m;
  __syncthreads();
  if (threadIdx.x < 32) {
    m = threadIdx.x < blockDim.x / 32 ? red[threadIdx.x] : 0.f;
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if (threadIdx.x == 0) red[0] = m;
  }
  __syncthreads();
  m = red[0];
  int t = m > 0.f ? static_cast<int>(ceilf(log2f(m))) - 13 : 0;
  float inv = exp2f(static_cast<float>(-t));
  if (threadIdx.x == 0) xscale[row] = exp2f(static_cast<float>(t));
  __half* out = reinterpret_cast<__half*>(x8 + static_cast<size_t>(row) * K * 2);
  // element k: kb = k/16, r = k%16; tq = (r%8)/2, reg = r/8, half = r%2
  for (int k = threadIdx.x; k < K; k += blockDim.x) {
    float v = __bfloat162float(xr[k]) * inv;
    int kb = k / 16, r = k % 16, tq = (r % 8) / 2, reg = r / 8, h = r % 2;
    out[(static_cast<size_t>(kb) * 4 + tq) * 4 + reg * 2 + h] = __float2half_rn(v);
  }
}

__global__ void reference_kernel(const uint8_t* codes, const uint8_t* scales, int N, int K,
                                 const __nv_bfloat16* x, int tok, float* y) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  const float lut[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  float acc = 0.f;
  for (int k = 0; k < K; ++k) {
    uint8_t cd = codes[static_cast<size_t>(n) * K + k];
    float w = lut[cd & 7] * ((cd & 8) ? -1.f : 1.f) *
              exp2f(float(scales[static_cast<size_t>(n) * (K / 32) + k / 32]) - 127.f);
    acc += w * __bfloat162float(x[static_cast<size_t>(tok) * K + k]);
  }
  y[n] = acc;
}

// ---------------------------------------------------------------- host side
// Byte encodings of one e2m1 code (s,e1,e0,m) for the two register slots.
static inline uint8_t enc_a(int c) { return uint8_t(((c & 8) << 4) | ((c & 7) << 2)); }     // {7,4,3,2}
static inline uint8_t enc_b(int c) {                                                        // {6,1,0,5}
  int s = (c >> 3) & 1, e = (c >> 1) & 3, m = c & 1;
  return uint8_t((s << 6) | e | (m << 5));
}

static std::vector<uint8_t> pack(const std::vector<uint8_t>& codes, const std::vector<uint8_t>& scales,
                                 int N, int K, bool act_rows) {
  int tiles = N / TILE_N, chunks = K / CHUNK_K;
  std::vector<uint8_t> out(static_cast<size_t>(tiles) * chunks * CHUNK_BYTES);
  for (int t = 0; t < tiles; ++t) {
    auto logical_row = [&](int r) {
      if (!act_rows) return t * TILE_N + r;
      return r < 32 ? t * 32 + r : N / 2 + t * 32 + (r - 32);
    };
    for (int c = 0; c < chunks; ++c) {
      uint8_t* blob = out.data() + (static_cast<size_t>(t) * chunks + c) * CHUNK_BYTES;
      for (int mbk = 0; mbk < 4; ++mbk)
        for (int kb = 0; kb < KB; ++kb)
          for (int ln = 0; ln < 32; ++ln) {
            int r0 = mbk * 16 + ln / 4, r1 = r0 + 8;
            int k0 = c * CHUNK_K + kb * 16 + (ln % 4) * 2;
            auto code = [&](int r, int k) { return codes[static_cast<size_t>(logical_row(r)) * K + k] & 0xF; };
            // byte order {k0, k0+1, k0+8, k0+9}: low 16 bits -> a0/a1, high 16 bits -> a2/a3
            const int ks[4] = {k0, k0 + 1, k0 + 8, k0 + 9};
            uint32_t w = 0;
            for (int b = 0; b < 4; ++b) w |= uint32_t(enc_a(code(r0, ks[b])) | enc_b(code(r1, ks[b]))) << (8 * b);
            reinterpret_cast<uint32_t*>(blob)[(mbk * KB + kb) * 32 + ln] = w;
          }
      uint8_t* sb = blob + W_BYTES;   // [mb][group][8 g][row g, row g+8]
      for (int mbk = 0; mbk < 4; ++mbk)
        for (int gr = 0; gr < GROUPS; ++gr)
          for (int r = 0; r < 8; ++r)
            for (int h = 0; h < 2; ++h)
              sb[((mbk * GROUPS + gr) * 8 + r) * 2 + h] =
                  scales[static_cast<size_t>(logical_row(mbk * 16 + r + 8 * h)) * (K / 32) + c * GROUPS + gr];
    }
  }
  return out;
}

struct Host { std::vector<uint8_t> codes, scales, packed; };

static Host make_expert(int N, int K, bool act_rows, std::mt19937& rng) {
  Host h;
  h.codes.resize(static_cast<size_t>(N) * K);
  h.scales.resize(static_cast<size_t>(N) * (K / 32));
  std::uniform_int_distribution<int> nib(0, 15), sc(118, 126);
  for (auto& v : h.codes) v = nib(rng);
  for (auto& v : h.scales) v = sc(rng);
  h.packed = pack(h.codes, h.scales, N, K, act_rows);
  return h;
}

static uint8_t* host_pinned(size_t bytes) {
  size_t round = (bytes + (1 << 21) - 1) / (1 << 21) * (1 << 21);
  uint8_t* p = static_cast<uint8_t*>(aligned_alloc(1 << 21, round));
  memset(p, 0, round);
  CK(cudaHostRegister(p, round, cudaHostRegisterMapped));
  uint8_t* d;
  CK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&d), p, 0));
  return d;
}

template <int EPI>
static void launch(const Params& p, int ctas, cudaStream_t st) {
  size_t smem = SMEM_HEAD + static_cast<size_t>(p.stages) * STAGE_BYTES;
  static bool set = false;
  if (!set) { CK(cudaFuncSetAttribute(tiered_moe_f16_kernel<EPI>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem)); set = true; }
  tiered_moe_f16_kernel<EPI><<<ctas, THREADS, smem, st>>>(p);
}

struct Acts { __nv_bfloat16* x; uint8_t* x8; float* xs; };
static Acts make_acts(int rows, int K, std::mt19937& rng, std::vector<__nv_bfloat16>* keep) {
  std::vector<__nv_bfloat16> xh(static_cast<size_t>(rows) * K);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::uniform_real_distribution<float> mag(-3.f, 3.f);
  for (int r = 0; r < rows; ++r) {
    float rs = exp2f(mag(rng));   // rows at different magnitudes exercise the scale
    for (int k = 0; k < K; ++k) xh[static_cast<size_t>(r) * K + k] = __float2bfloat16(nd(rng) * rs);
  }
  Acts a;
  CK(cudaMalloc(&a.x, xh.size() * 2)); CK(cudaMemcpy(a.x, xh.data(), xh.size() * 2, cudaMemcpyHostToDevice));
  CK(cudaMalloc(&a.x8, xh.size() * 2)); CK(cudaMalloc(&a.xs, rows * sizeof(float)));
  prep_kernel<<<rows, 256>>>(a.x, K, a.x8, a.xs);
  CK(cudaGetLastError());
  if (keep) *keep = xh;
  return a;
}

static int check(int N, int K, int n_hot, int n_cold, int cold_ctas) {
  std::mt19937 rng(1234);
  const int ctas = 132, stages = 4, tokens = 8;
  bool ok = true;
  for (int mode = 0; mode < 2; ++mode) {
    bool act = mode == 1;
    int n = n_hot + n_cold;
    std::vector<Host> hs;
    for (int i = 0; i < n; ++i) hs.push_back(make_expert(N, K, act, rng));
    std::vector<Expert> ex(n);
    std::uniform_int_distribution<int> ntok(1, 8), tk(0, tokens - 1);
    int routes = 0;
    for (int i = 0; i < n; ++i) {
      ex[i].ntok = ntok(rng) <= 5 ? 1 : ntok(rng);
      for (int j = 0; j < ex[i].ntok; ++j) { ex[i].tok[j] = tk(rng); ex[i].route[j] = routes++; ex[i].wt[j] = 1.f; }
      size_t bytes = hs[i].packed.size();
      uint8_t* d;
      if (i < n_hot) CK(cudaMalloc(&d, bytes)); else d = host_pinned(bytes);
      CK(cudaMemcpy(d, hs[i].packed.data(), bytes, cudaMemcpyHostToDevice));
      ex[i].w = d;
    }
    Acts a = make_acts(tokens, K, rng, nullptr);
    Expert* ed; CK(cudaMalloc(&ed, sizeof(Expert) * n));
    CK(cudaMemcpy(ed, ex.data(), sizeof(Expert) * n, cudaMemcpyHostToDevice));
    float* yraw; CK(cudaMalloc(&yraw, sizeof(float) * routes * N)); CK(cudaMemset(yraw, 0, sizeof(float) * routes * N));
    __nv_bfloat16* yact; CK(cudaMalloc(&yact, 2 * routes * (N / 2)));
    Params p{ed, n_hot, ed + n_hot, n_cold, cold_ctas, N, K, a.x8, a.xs, yraw, yact, nullptr, stages};
    if (act) launch<ACT>(p, ctas, 0); else launch<RAW>(p, ctas, 0);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    uint8_t *cd, *sd; float* yr;
    CK(cudaMalloc(&cd, static_cast<size_t>(N) * K)); CK(cudaMalloc(&sd, static_cast<size_t>(N) * K / 32));
    CK(cudaMalloc(&yr, sizeof(float) * N));
    double max_rel = 0, sum_rel = 0; long cnt = 0; int bad = 0;
    for (int i = 0; i < n; ++i) {
      CK(cudaMemcpy(cd, hs[i].codes.data(), hs[i].codes.size(), cudaMemcpyHostToDevice));
      CK(cudaMemcpy(sd, hs[i].scales.data(), hs[i].scales.size(), cudaMemcpyHostToDevice));
      for (int j = 0; j < ex[i].ntok; ++j) {
        reference_kernel<<<(N + 127) / 128, 128>>>(cd, sd, N, K, a.x, ex[i].tok[j], yr);
        std::vector<float> ref(N);
        CK(cudaMemcpy(ref.data(), yr, sizeof(float) * N, cudaMemcpyDeviceToHost));
        double scale = 0; for (float v : ref) scale = std::max(scale, (double)fabs(v));
        if (!act) {
          std::vector<float> got(N);
          CK(cudaMemcpy(got.data(), yraw + static_cast<size_t>(ex[i].route[j]) * N, sizeof(float) * N, cudaMemcpyDeviceToHost));
          for (int k = 0; k < N; ++k) {
            double rel = fabs(got[k] - ref[k]) / (scale + 1e-30);
            max_rel = std::max(max_rel, rel); sum_rel += rel; ++cnt; bad += rel > 1e-2;
          }
        } else {
          std::vector<__nv_bfloat16> got(N / 2);
          CK(cudaMemcpy(got.data(), yact + static_cast<size_t>(ex[i].route[j]) * (N / 2), N, cudaMemcpyDeviceToHost));
          for (int k = 0; k < N / 2; ++k) {
            float gt = ref[k], up = ref[N / 2 + k];
            float want = gt / (1.f + expf(-gt)) * up;
            double rel = fabs(__bfloat162float(got[k]) - want) / (scale * scale * 0.5 + 1e-30);
            max_rel = std::max(max_rel, rel); sum_rel += rel; ++cnt; bad += rel > 2e-2;
          }
        }
      }
    }
    printf("check %s N=%d K=%d hot=%d cold=%d: max rel err %.2e, mean %.2e, %d bad -> %s\n", act ? "ACT" : "RAW",
           N, K, n_hot, n_cold, max_rel, sum_rel / cnt, bad, bad ? "FAIL" : "ok");
    ok &= bad == 0;
  }
  return ok ? 0 : 1;
}

static void bench(int N, int K, int n_hot, int n_cold, int cold_ctas, int stages, int ntok_mode, bool act_rows) {
  const int ctas = 132, reps = 20, pool_hot = 48, pool_cold = 24;
  std::mt19937 rng(7);
  Host proto = make_expert(N, K, act_rows, rng);
  size_t bytes = proto.packed.size();
  std::vector<uint8_t*> hot_pool(pool_hot), cold_pool(pool_cold);
  for (auto& d : hot_pool) { CK(cudaMalloc(&d, bytes)); CK(cudaMemcpy(d, proto.packed.data(), bytes, cudaMemcpyHostToDevice)); }
  for (auto& d : cold_pool) { d = host_pinned(bytes); CK(cudaMemcpy(d, proto.packed.data(), bytes, cudaMemcpyHostToDevice)); }
  const int tokens = 8;
  Acts a = make_acts(64, K, rng, nullptr);
  float* yraw; CK(cudaMalloc(&yraw, sizeof(float) * 64 * N));
  __nv_bfloat16* yact; CK(cudaMalloc(&yact, 2 * 64 * (N / 2)));
  float* ysum; CK(cudaMalloc(&ysum, sizeof(float) * tokens * N));
  std::vector<Expert*> lists(reps);
  std::uniform_int_distribution<int> tk(0, tokens - 1);
  std::discrete_distribution<int> tok_dist({0, 74.1, 17.0, 5.4, 2.1, 0.8, 0.4, 0.1, 0.1});
  for (int r = 0; r < reps; ++r) {
    std::vector<Expert> ex(n_hot + n_cold);
    int routes = 0;
    for (int i = 0; i < n_hot + n_cold; ++i) {
      ex[i].w = i < n_hot ? hot_pool[(r * n_hot + i) % pool_hot] : cold_pool[(r * n_cold + i - n_hot) % pool_cold];
      ex[i].ntok = ntok_mode > 0 ? ntok_mode : std::max(1, tok_dist(rng));
      for (int j = 0; j < ex[i].ntok; ++j) { ex[i].tok[j] = tk(rng); ex[i].route[j] = routes++ % 64; ex[i].wt[j] = 0.1f; }
    }
    CK(cudaMalloc(&lists[r], sizeof(Expert) * ex.size()));
    CK(cudaMemcpy(lists[r], ex.data(), sizeof(Expert) * ex.size(), cudaMemcpyHostToDevice));
  }
  cudaStream_t st; CK(cudaStreamCreate(&st));
  auto run = [&](int r) {
    Params p{lists[r], n_hot, lists[r] + n_hot, n_cold, n_cold ? cold_ctas : 0, N, K, a.x8, a.xs, yraw, yact, ysum, stages};
    if (act_rows) launch<ACT>(p, ctas, st); else launch<WSUM>(p, ctas, st);
  };
  run(0); CK(cudaStreamSynchronize(st));
  cudaGraph_t graph; cudaGraphExec_t exec;
  CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
  for (int r = 0; r < reps; ++r) run(r);
  CK(cudaStreamEndCapture(st, &graph));
  CK(cudaGraphInstantiate(&exec, graph, 0));
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  CK(cudaGraphLaunch(exec, st)); CK(cudaStreamSynchronize(st));
  float best = 1e30f;
  for (int trial = 0; trial < 5; ++trial) {
    CK(cudaEventRecord(e0, st)); CK(cudaGraphLaunch(exec, st)); CK(cudaEventRecord(e1, st));
    CK(cudaEventSynchronize(e1));
    float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); best = std::min(best, ms);
  }
  double us = best * 1000.0 / reps;
  if (PROF) {
    unsigned long long z[8] = {}, h[8];
    CK(cudaMemcpyToSymbol(g_prof, z, sizeof(z)));
    CK(cudaGraphLaunch(exec, st)); CK(cudaStreamSynchronize(st));
    CK(cudaMemcpyFromSymbol(h, g_prof, sizeof(h)));
    double mhz = h[1] * 1e3 / h[6];
    double k = 1.0 / (h[5] * mhz);  // us per CTA per launch at the measured SM clock
    printf("  effective SM clock %.0f MHz\n", mhz);
    printf("  prof per CTA: consumer total %.1f us, wait-full %.1f, epilogue %.1f, compute %.1f | producer total %.1f, wait-empty %.1f\n",
           h[1] * k, h[0] * k, h[4] * k, (h[1] - h[0] - h[4]) * k, h[3] * k, h[2] * k);
  }
  double hb = double(n_hot) * bytes, cb = double(n_cold) * bytes;
  double sol = std::max(hb / 3.626e12, cb / 0.409e12) * 1e6;
  printf("f16 %s hot=%2d cold=%d tok=%s cold_ctas=%2d stages=%d: %7.1f us  (SOL %6.1f us, %3.0f%%)  %.0f GB/s total\n",
         act_rows ? "w13" : "w2 ", n_hot, n_cold, ntok_mode ? std::to_string(ntok_mode).c_str() : "dist", cold_ctas,
         stages, us, sol, sol / us * 100, (hb + cb) / (us * 1e-6) / 1e9);
}

int main(int argc, char** argv) {
  if (argc < 2) { printf("usage: tiered_moe_f16 check | bench <w13|w2> hot cold cold_ctas stages ntok\n"); return 1; }
  if (!strcmp(argv[1], "check")) {
    int rc = 0;
    rc |= check(4096, 6144, 3, 2, 16);
    rc |= check(6144, 2048, 2, 3, 24);
    rc |= check(4096, 6144, 4, 0, 0);
    rc |= check(4096, 6144, 0, 2, 132);
    return rc;
  }
  bool w13 = !strcmp(argv[2], "w13");
  int n_hot = atoi(argv[3]), n_cold = atoi(argv[4]), cold_ctas = atoi(argv[5]), stages = atoi(argv[6]);
  int ntok = argc > 7 ? atoi(argv[7]) : 0;
  if (w13) bench(4096, 6144, n_hot, n_cold, cold_ctas, stages, ntok, true);
  else bench(6144, 2048, n_hot, n_cold, cold_ctas, stages, ntok, false);
  return 0;
}

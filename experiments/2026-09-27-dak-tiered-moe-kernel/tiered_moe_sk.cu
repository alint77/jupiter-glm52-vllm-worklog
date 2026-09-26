// Tiered mxfp4 MoE GEMM for MiMo decode (v6: stream-K, sync-free epilogue).
//
// v6 over v5: CTAs of a tier split that tier's (tile, chunk) units into equal
// contiguous ranges (stream-K), and every consumer warp adds its partial sums
// straight into fp32 outputs with red.global.add, so there is no inter-warp
// reduction, named barrier, or wave-quantization tail. w13 therefore returns
// raw gate/up sums (silu * up moves into w2's activation prep, which has to
// run anyway); w2 adds weight * y into the per-token sum. One-token experts
// (74% in MiMo decode) take an f16x2-FMA path: an mma would waste 7 of its 8
// columns, and the tensor pipe is what drives the chip into its 680 W cap.
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
//   tiered_moe_sk check | bench <w13|w2> hot cold cold_ctas stages ntok(0=dist)

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
constexpr int SMEM_HEAD = 256;
#ifndef DIAG
#define DIAG 0
#endif

#ifndef GEMV_PATH
#define GEMV_PATH 1
#endif
#ifndef PROF
#define PROF 0
#endif
// PROF=1, summed over CTAs (consumer = warp 0): [0] consumer wait-full cycles, [1] consumer cycles,
// [2] producer wait-empty cycles, [3] producer cycles, [4] flush cycles, [5] CTAs, [6] consumer ns
__device__ unsigned long long g_prof[8];
__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t;
}

enum Epilogue { RAW = 0, WSUM = 1 };  // y[route][n] += v  |  y[tok][n] += wt * v
// MiMo-V2 expert shapes (hidden 6144, moe intermediate 2048): w13 runs RAW, w2 runs WSUM.
template <int EPI> constexpr int SHAPE_N = EPI == RAW ? 4096 : 6144;
template <int EPI> constexpr int SHAPE_K = EPI == RAW ? 6144 : 2048;

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
  float* y;                // zeroed by the caller
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
__device__ __forceinline__ void flush(const Params& p, const Expert& e, int ntok, int n0, int g, int tq, float* acc) {
  constexpr int N = SHAPE_N<EPI>;
  if (GEMV_PATH && ntok == 1) {
#pragma unroll
    for (int i = 0; i < 4; i += 2) {
      acc[i] += __shfl_xor_sync(0xffffffffu, acc[i], 1);
      acc[i] += __shfl_xor_sync(0xffffffffu, acc[i], 2);
    }
    if (tq) { acc[0] = acc[2] = 0.f; return; }
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int tok = 2 * tq + (i & 1);
    if (tok < ntok) {
      const int n = n0 + g + (i >> 1) * 8;
      const float v = acc[i] * p.xscale[e.tok[tok]];
      if constexpr (EPI == RAW) atomicAdd(&p.y[static_cast<size_t>(e.route[tok]) * N + n], v);
      else atomicAdd(&p.y[static_cast<size_t>(e.tok[tok]) * N + n], e.wt[tok] * v);
    }
    acc[i] = 0.f;
  }
}

template <int EPI>
__global__ void __launch_bounds__(THREADS, 1) tiered_moe_sk_kernel(Params p) {
  extern __shared__ __align__(128) unsigned char smem[];
  const int stages = p.stages;
  uint64_t* full = reinterpret_cast<uint64_t*>(smem);
  uint64_t* empty = full + stages;
  unsigned char* ring = smem + SMEM_HEAD;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;

  const bool cold = blockIdx.x < p.cold_ctas;
  const Expert* experts = cold ? p.cold : p.hot;
  const int n_exp = cold ? p.n_cold : p.n_hot;
  const int role_ctas = cold ? p.cold_ctas : gridDim.x - p.cold_ctas;
  const int role_idx = cold ? blockIdx.x : blockIdx.x - p.cold_ctas;
  constexpr int tiles_per_exp = SHAPE_N<EPI> / TILE_N;
  constexpr int chunks = SHAPE_K<EPI> / CHUNK_K;
  const long long units = static_cast<long long>(n_exp) * tiles_per_exp * chunks;
  const int u0 = static_cast<int>(units * role_idx / max(role_ctas, 1));
  const int u1 = static_cast<int>(units * (role_idx + 1) / max(role_ctas, 1));
  constexpr size_t tile_bytes = static_cast<size_t>(chunks) * CHUNK_BYTES;
  constexpr size_t xrow_bytes = static_cast<size_t>(SHAPE_K<EPI>) * 2;

  if (threadIdx.x == 0) {
    for (int s = 0; s < stages; ++s) { mbar_init(&full[s], 1); mbar_init(&empty[s], CONSUMER_WARPS); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (role_ctas <= 0 || u0 >= u1) return;

  if (warp == CONSUMER_WARPS) {
    if (lane == 0) {
      long long pw = 0, pt0 = clock64();
      for (int u = u0, it = 0; u < u1; ++u, ++it) {
        const int t = u / chunks, c = u - t * chunks;
        const Expert& e = experts[t / tiles_per_exp];
        const uint8_t* tile = e.w + static_cast<size_t>(t % tiles_per_exp) * tile_bytes;
        const int s = it % stages;
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
      if (PROF) { atomicAdd(&g_prof[2], (unsigned long long)pw); atomicAdd(&g_prof[3], (unsigned long long)(clock64() - pt0)); }
    }
    return;
  }

  const int mb = warp % 4, slice = warp / 4;
  const int g = lane / 4, tq = lane % 4;
  constexpr int GS = GROUPS / KSPLIT;         // groups per warp per stage
  const int g0 = slice * GS;
  float acc[4] = {};
  long long cw = 0, cf = 0, ct0 = clock64();
  unsigned long long gt0 = PROF ? gtimer() : 0;
  int ntok_next = experts[u0 / chunks / tiles_per_exp].ntok;
  for (int u = u0, it = 0; u < u1; ++u, ++it) {
    const int t = u / chunks, c = u - t * chunks;
    const Expert& e = experts[t / tiles_per_exp];
    const int ntok = ntok_next;   // loaded one unit ahead: keeps the L1/L2 latency off the stage
    if (u + 1 < u1) ntok_next = experts[(u + 1) / chunks / tiles_per_exp].ntok;
    const int s = it % stages;
    {
      long long w0 = PROF ? clock64() : 0;
      mbar_wait(&full[s], (it / stages) & 1);
      if (PROF) cw += clock64() - w0;
    }
    if (DIAG == 1) { __syncwarp(); if (lane == 0) mbar_arrive(&empty[s]); continue; }
    const unsigned char* st = ring + static_cast<size_t>(s) * STAGE_BYTES;
    const uint32_t* wq = reinterpret_cast<const uint32_t*>(st) + (mb * KB + 2 * g0) * 32 + lane;
    const uint16_t* sc = reinterpret_cast<const uint16_t*>(st + W_BYTES) + (mb * GROUPS + g0) * 8 + g;
    if (GEMV_PATH && ntok == 1) {
      // f16x2 FMAs on rows g and g+8 over this lane's k pairs; f16 sums span one group
      const uint2* x0 = reinterpret_cast<const uint2*>(st + CHUNK_BYTES) + 2 * g0 * 4 + tq;
#pragma unroll
      for (int j = 0; j < GS; ++j) {
        uint32_t sp = sc[j * 8];
        __half2 h0 = __float2half2_rn(0.f), h1 = h0;
#pragma unroll
        for (int v = 0; v < 2; ++v) {
          uint32_t w = wq[(2 * j + v) * 32];
          uint2 xb = x0[(2 * j + v) * 4];
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
    } else {
      const uint2* xs = reinterpret_cast<const uint2*>(st + CHUNK_BYTES + g * XROW_STRIDE) + 2 * g0 * 4 + tq;
      const bool has_tok = g < ntok;
#pragma unroll
      for (int j = 0; j < GS; ++j) {
        uint32_t sp = sc[j * 8];
        const float zero[4] = {0.f, 0.f, 0.f, 0.f};
        float d[4];
#pragma unroll
        for (int v = 0; v < 2; ++v) {
          uint32_t w = wq[(2 * j + v) * 32];
          uint2 xb = has_tok ? xs[(2 * j + v) * 4] : make_uint2(0u, 0u);
          uint32_t ra = reg_a(w), rb = reg_b(w);
          uint32_t a[4] = {e4m3x2_lo(ra), e4m3x2_lo(rb), e4m3x2_hi(ra), e4m3x2_hi(rb)};
          mma_f16(d, a, xb.x, xb.y, v ? d : zero);
        }
        float s0 = group_scale(sp & 0xFFu), s1 = group_scale(sp >> 8);
        acc[0] = fmaf(s0, d[0], acc[0]);
        acc[1] = fmaf(s0, d[1], acc[1]);
        acc[2] = fmaf(s1, d[2], acc[2]);
        acc[3] = fmaf(s1, d[3], acc[3]);
      }
    }
    __syncwarp();
    if (lane == 0) mbar_arrive(&empty[s]);
    if (c == chunks - 1 || u == u1 - 1) {
      long long f0 = PROF ? clock64() : 0;
      flush<EPI>(p, e, ntok, (t % tiles_per_exp) * TILE_N + mb * 16, g, tq, acc);
      if (PROF) cf += clock64() - f0;
    }
  }
  if (PROF && threadIdx.x == 0) {
    atomicAdd(&g_prof[0], (unsigned long long)cw); atomicAdd(&g_prof[1], (unsigned long long)(clock64() - ct0));
    atomicAdd(&g_prof[4], (unsigned long long)cf); atomicAdd(&g_prof[5], 1ull);
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
                                 int N, int K) {
  int tiles = N / TILE_N, chunks = K / CHUNK_K;
  std::vector<uint8_t> out(static_cast<size_t>(tiles) * chunks * CHUNK_BYTES);
  for (int t = 0; t < tiles; ++t) {
    auto logical_row = [&](int r) { return t * TILE_N + r; };
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

static Host make_expert(int N, int K, std::mt19937& rng) {
  Host h;
  h.codes.resize(static_cast<size_t>(N) * K);
  h.scales.resize(static_cast<size_t>(N) * (K / 32));
  std::uniform_int_distribution<int> nib(0, 15), sc(118, 126);
  for (auto& v : h.codes) v = nib(rng);
  for (auto& v : h.scales) v = sc(rng);
  h.packed = pack(h.codes, h.scales, N, K);
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
  if (p.N != SHAPE_N<EPI> || p.K != SHAPE_K<EPI>) { printf("shape %dx%d is not MiMo's for this epilogue\n", p.N, p.K); exit(1); }
  size_t smem = SMEM_HEAD + static_cast<size_t>(p.stages) * STAGE_BYTES;
  static bool set = false;
  if (!set) { CK(cudaFuncSetAttribute(tiered_moe_sk_kernel<EPI>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem)); set = true; }
  tiered_moe_sk_kernel<EPI><<<ctas, THREADS, smem, st>>>(p);
}

struct Acts { __nv_bfloat16* x; uint8_t* x8; float* xs; };
static Acts make_acts(int rows, int K, std::mt19937& rng) {
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
  return a;
}

// w13 (RAW): every route row against the dequantized reference. w2 (WSUM):
// every token row against sum over its routes of wt * reference. Token counts per expert
// follow the MiMo decode mix, with extra multi-token experts so both paths run.
static int check(int N, int K, int n_hot, int n_cold, int cold_ctas) {
  std::mt19937 rng(1234);
  const int ctas = 132, stages = 4, tokens = 8;
  bool ok = true;
  const int n = n_hot + n_cold;
  std::vector<Host> hs;
  for (int i = 0; i < n; ++i) hs.push_back(make_expert(N, K, rng));
  std::vector<Expert> ex(n);
  std::uniform_int_distribution<int> ntok(1, 8), tk(0, tokens - 1);
  std::uniform_real_distribution<float> wt(0.05f, 1.f);
  int routes = 0;
  for (int i = 0; i < n; ++i) {
    ex[i].ntok = ntok(rng) <= 4 ? 1 : ntok(rng);
    std::vector<int> perm(tokens);
    for (int j = 0; j < tokens; ++j) perm[j] = j;
    std::shuffle(perm.begin(), perm.end(), rng);
    for (int j = 0; j < ex[i].ntok; ++j) { ex[i].tok[j] = perm[j]; ex[i].route[j] = routes++; ex[i].wt[j] = wt(rng); }
    size_t bytes = hs[i].packed.size();
    uint8_t* d;
    if (i < n_hot) CK(cudaMalloc(&d, bytes)); else d = host_pinned(bytes);
    CK(cudaMemcpy(d, hs[i].packed.data(), bytes, cudaMemcpyHostToDevice));
    ex[i].w = d;
  }
  Acts a = make_acts(tokens, K, rng);
  Expert* ed; CK(cudaMalloc(&ed, sizeof(Expert) * n));
  CK(cudaMemcpy(ed, ex.data(), sizeof(Expert) * n, cudaMemcpyHostToDevice));
  // per (expert, token) reference outputs
  uint8_t *cd, *sd; float* yr;
  CK(cudaMalloc(&cd, static_cast<size_t>(N) * K)); CK(cudaMalloc(&sd, static_cast<size_t>(N) * K / 32));
  CK(cudaMalloc(&yr, sizeof(float) * N));
  std::vector<std::vector<float>> ref(routes, std::vector<float>(N));
  for (int i = 0; i < n; ++i) {
    CK(cudaMemcpy(cd, hs[i].codes.data(), hs[i].codes.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(sd, hs[i].scales.data(), hs[i].scales.size(), cudaMemcpyHostToDevice));
    for (int j = 0; j < ex[i].ntok; ++j) {
      reference_kernel<<<(N + 127) / 128, 128>>>(cd, sd, N, K, a.x, ex[i].tok[j], yr);
      CK(cudaMemcpy(ref[ex[i].route[j]].data(), yr, sizeof(float) * N, cudaMemcpyDeviceToHost));
    }
  }
  {
    const bool wsum = N == SHAPE_N<WSUM>;
    const int rows = wsum ? tokens : routes;
    std::vector<std::vector<float>> want(rows, std::vector<float>(N, 0.f));
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < ex[i].ntok; ++j)
        for (int k = 0; k < N; ++k) {
          const float r = ref[ex[i].route[j]][k];
          if (wsum) want[ex[i].tok[j]][k] += ex[i].wt[j] * r; else want[ex[i].route[j]][k] = r;
        }
    float* y; CK(cudaMalloc(&y, sizeof(float) * rows * N)); CK(cudaMemset(y, 0, sizeof(float) * rows * N));
    Params p{ed, n_hot, ed + n_hot, n_cold, cold_ctas, N, K, a.x8, a.xs, y, stages};
    if (wsum) launch<WSUM>(p, ctas, 0); else launch<RAW>(p, ctas, 0);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    std::vector<float> got(static_cast<size_t>(rows) * N);
    CK(cudaMemcpy(got.data(), y, sizeof(float) * got.size(), cudaMemcpyDeviceToHost));
    double max_rel = 0, sum_rel = 0; long cnt = 0; int bad = 0;
    for (int r = 0; r < rows; ++r) {
      double scale = 0; for (float v : want[r]) scale = std::max(scale, (double)fabs(v));
      if (scale == 0) continue;
      for (int k = 0; k < N; ++k) {
        double rel = fabs(got[static_cast<size_t>(r) * N + k] - want[r][k]) / scale;
        max_rel = std::max(max_rel, rel); sum_rel += rel; ++cnt; bad += rel > 1e-2;
      }
    }
    printf("check %s N=%d K=%d hot=%d cold=%d: max rel err %.2e, mean %.2e, %d bad -> %s\n", wsum ? "WSUM" : "RAW ",
           N, K, n_hot, n_cold, max_rel, sum_rel / cnt, bad, bad ? "FAIL" : "ok");
    ok &= bad == 0;
    CK(cudaFree(y));
  }
  return ok ? 0 : 1;
}

static void bench(int N, int K, int n_hot, int n_cold, int cold_ctas, int stages, int ntok_mode, bool w13) {
  const int ctas = 132, reps = 20, pool_hot = 48, pool_cold = 24;
  std::mt19937 rng(7);
  Host proto = make_expert(N, K, rng);
  size_t bytes = proto.packed.size();
  std::vector<uint8_t*> hot_pool(pool_hot), cold_pool(pool_cold);
  for (auto& d : hot_pool) { CK(cudaMalloc(&d, bytes)); CK(cudaMemcpy(d, proto.packed.data(), bytes, cudaMemcpyHostToDevice)); }
  for (auto& d : cold_pool) { d = host_pinned(bytes); CK(cudaMemcpy(d, proto.packed.data(), bytes, cudaMemcpyHostToDevice)); }
  const int tokens = 8;
  Acts a = make_acts(64, K, rng);
  float* y; CK(cudaMalloc(&y, sizeof(float) * 64 * N)); CK(cudaMemset(y, 0, sizeof(float) * 64 * N));
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
    Params p{lists[r], n_hot, lists[r] + n_hot, n_cold, n_cold ? cold_ctas : 0, N, K, a.x8, a.xs, y, stages};
    if (w13) launch<RAW>(p, ctas, st); else launch<WSUM>(p, ctas, st);
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
  double hb = double(n_hot) * bytes, cb = double(n_cold) * bytes;
  double sol = std::max(hb / 3.626e12, cb / 0.409e12) * 1e6;
  printf("sk %s hot=%2d cold=%d tok=%s cold_ctas=%2d stages=%d: %7.1f us  (SOL %6.1f us, %3.0f%%)  %.0f GB/s total\n",
         w13 ? "w13" : "w2 ", n_hot, n_cold, ntok_mode ? std::to_string(ntok_mode).c_str() : "dist", cold_ctas,
         stages, us, sol, sol / us * 100, (hb + cb) / (us * 1e-6) / 1e9);
  if (PROF) {
    unsigned long long z[8] = {}, h[8];
    CK(cudaMemcpyToSymbol(g_prof, z, sizeof(z)));
    CK(cudaGraphLaunch(exec, st)); CK(cudaStreamSynchronize(st));
    CK(cudaMemcpyFromSymbol(h, g_prof, sizeof(h)));
    double mhz = h[1] * 1e3 / h[6];
    double k = 1.0 / (h[5] * mhz);
    printf("  prof: SM clock %.0f MHz | per CTA consumer %.1f us (wait-full %.1f, flush %.1f, compute %.1f)"
           " | producer %.1f (wait-empty %.1f)\n", mhz, h[1] * k, h[0] * k, h[4] * k, (h[1] - h[0] - h[4]) * k,
           h[3] * k, h[2] * k);
  }
}

int main(int argc, char** argv) {
  if (argc < 2) { printf("usage: tiered_moe_sk check | bench <w13|w2> hot cold cold_ctas stages ntok\n"); return 1; }
  if (!strcmp(argv[1], "check")) {
    int rc = 0;
    rc |= check(4096, 6144, 3, 2, 16);
    rc |= check(6144, 2048, 2, 3, 24);
    rc |= check(4096, 6144, 5, 0, 0);
    rc |= check(4096, 6144, 0, 2, 132);
    rc |= check(6144, 2048, 9, 2, 20);
    return rc;
  }
  bool w13 = !strcmp(argv[2], "w13");
  int n_hot = atoi(argv[3]), n_cold = atoi(argv[4]), cold_ctas = atoi(argv[5]), stages = atoi(argv[6]);
  int ntok = argc > 7 ? atoi(argv[7]) : 0;
  if (w13) bench(4096, 6144, n_hot, n_cold, cold_ctas, stages, ntok, true);
  else bench(6144, 2048, n_hot, n_cold, cold_ctas, stages, ntok, false);
  return 0;
}

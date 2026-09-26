// Tiered mxfp4 MoE GEMM for decode: hot experts in HBM and cold experts in
// pinned Grace memory, in ONE persistent, warp-specialized launch (DAK-style).
//
// Problem: per layer and GPU, ~9 hot and ~2 cold experts, each seeing 1-2 of
// 8 verify tokens. Each expert GEMM is [N x K] mxfp4 weights times a handful of
// bf16 token rows -- a dequant-GEMV, purely bandwidth-bound.
//
// Layout (packed offline, see pack()): an expert's matrix is a sequence of
// N-tiles of 64 rows; a tile is a sequence of K-chunks of 1024; a chunk is one
// contiguous blob = weights in mma-fragment order (32 KB) + e8m0 scales (2 KB).
// One 1D bulk-TMA copy moves a whole chunk into shared memory.
//
// CTA roles: CTAs [0, cold_ctas) stream cold tiles over NVLink-C2C, the rest
// stream hot tiles from HBM. Warp 8 is the TMA producer; warps 0-7 consume:
// warp w owns rows (w % 4) * 16 .. +16 and K half (w / 4) of every chunk.
// Weights are the mma's M operand (16 rows) and tokens its N operand (8), so
// mma.m16n8k16 covers up to 8 tokens per expert exactly.
//
// Epilogues: RAW stores fp32 rows per route (validation), ACT fuses
// silu(gate) * up (w13; tile rows 0-31 are gate rows, 32-63 the matching up
// rows) into bf16, WSUM adds route_weight * y into fp32 token rows (w2).
//
//   tiered_moe check|bench ...   (see main)

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  printf("CUDA error %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

constexpr int TILE_N = 64;
constexpr int CHUNK_K = 1024;
constexpr int KB = CHUNK_K / 16;                      // k16 blocks per chunk
constexpr int GROUPS = CHUNK_K / 32;                  // e8m0 groups per row per chunk
constexpr int W_BYTES = TILE_N * CHUNK_K / 2;         // 32768
constexpr int S_BYTES = TILE_N * GROUPS;              // 2048
constexpr int CHUNK_BYTES = W_BYTES + S_BYTES;        // 34816
constexpr int MAX_TOK = 8;
constexpr int X_STRIDE = CHUNK_K + 8;                 // bf16 elements, 16 B pad
constexpr int X_BYTES = MAX_TOK * X_STRIDE * 2;       // 16512
constexpr int STAGE_BYTES = CHUNK_BYTES + X_BYTES;    // 51328
#ifndef CONSUMER_WARPS_DEF
#define CONSUMER_WARPS_DEF 16
#endif
constexpr int CONSUMER_WARPS = CONSUMER_WARPS_DEF;
constexpr int KSPLIT = CONSUMER_WARPS / 4;            // K slices per chunk
constexpr int RED_FLOATS = (KSPLIT - 1) * 4 * 32 * 4; // partial accumulators
constexpr int EX_FLOATS = 64 * 8;                     // ACT exchange [64 rows][8 tok]
constexpr int SMEM_HEAD = 256 + (RED_FLOATS + EX_FLOATS) * 4;
constexpr int THREADS = (CONSUMER_WARPS + 1) * 32;

enum Epilogue { RAW = 0, ACT = 1, WSUM = 2 };

// Diagnostic builds: DIAG=1 consumers skip compute (pure TMA pipeline rate),
// DIAG=2 the producer skips the copies (pure consumer rate on stale smem).
#ifndef DIAG
#define DIAG 0
#endif

struct Expert {
  const uint8_t* w;       // packed matrix (tiles contiguous)
  int ntok;
  int tok[MAX_TOK];       // token row in the input activations
  int route[MAX_TOK];     // output row (RAW/ACT) -- route slot
  float wt[MAX_TOK];      // route weight (WSUM)
};

struct Params {
  const Expert* hot; int n_hot;
  const Expert* cold; int n_cold;
  int cold_ctas;
  int N, K;               // logical GEMM shape
  const __nv_bfloat16* x; // [tokens or routes][K] input rows
  float* y_raw;           // RAW: [routes][N]
  __nv_bfloat16* y_act;   // ACT: [routes][N/2]
  float* y_sum;           // WSUM: [tokens][N]
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
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
               :: "r"(smem_u32(bar)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
  asm volatile(
      "{\n .reg .pred p;\n WAIT_%=:\n"
      " mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      " @!p bra WAIT_%=;\n}\n" :: "r"(smem_u32(bar)), "r"(parity) : "memory");
}
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
      :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void consumer_sync() {
  asm volatile("bar.sync 1, %0;" :: "n"(CONSUMER_WARPS * 32) : "memory");
}

__device__ __forceinline__ void mma_bf16(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// 8 e2m1 nibbles -> 4 bf16x2 A-fragment registers, then scale rows.
// Nibble placement (pack()): reg0={n3,n7} reg1={n2,n6} reg2={n1,n5} reg3={n0,n4}.
// The shift puts e1e0m into bf16 exponent[1:0]/mantissa[6]; multiplying by
// 2^126 * 2^(e8m0-127) = 2^(e8m0-1) restores the value and applies the scale.
__device__ __forceinline__ void dequant(uint32_t q, uint32_t s_lo, uint32_t s_hi, uint32_t* a) {
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    uint32_t v = (q & 0x80008000u) | ((q & 0x70007000u) >> 6);
    __nv_bfloat162 f = *reinterpret_cast<__nv_bfloat162*>(&v);
    __nv_bfloat162 s = *reinterpret_cast<const __nv_bfloat162*>((i & 1) ? &s_hi : &s_lo);
    f = __hmul2(f, s);
    a[i] = *reinterpret_cast<uint32_t*>(&f);
    q <<= 4;
  }
}

__device__ __forceinline__ uint32_t scale_bf16x2(uint8_t e) {
  uint32_t b = (static_cast<uint32_t>(e) + 126u) << 7;  // bf16 bits of 2^(e-1)
  return b | (b << 16);
}

// ---------------------------------------------------------------- the kernel
template <int EPI>
__global__ void __launch_bounds__(THREADS, 1) tiered_moe_kernel(Params p) {
  extern __shared__ __align__(128) unsigned char smem[];
  const int stages = p.stages;
  uint64_t* full = reinterpret_cast<uint64_t*>(smem);
  uint64_t* empty = full + stages;
  float* red = reinterpret_cast<float*>(smem + 256);   // [KSPLIT-1][4 mb][32][4]
  float* ex = red + RED_FLOATS;                         // ACT exchange
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

  if (threadIdx.x == 0) {
    for (int s = 0; s < stages; ++s) { mbar_init(&full[s], 1); mbar_init(&empty[s], CONSUMER_WARPS); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (role_ctas <= 0 || n_tiles == 0) return;

  if (warp == CONSUMER_WARPS) {
    // ------------------------------------------------------------ producer
    if (lane == 0) {
      int it = 0;
      for (int t = role_idx; t < n_tiles; t += role_ctas) {
        const Expert& e = experts[t / tiles_per_exp];
        const uint8_t* tile = e.w + static_cast<size_t>(t % tiles_per_exp) * tile_bytes;
        for (int c = 0; c < chunks; ++c, ++it) {
          int s = it % stages;
          if (it >= stages) mbar_wait(&empty[s], ((it / stages) - 1) & 1);
          unsigned char* dst = ring + static_cast<size_t>(s) * STAGE_BYTES;
          if (DIAG == 2) { mbar_arrive(&full[s]); continue; }
          mbar_expect_tx(&full[s], CHUNK_BYTES + e.ntok * CHUNK_K * 2);
          bulk_g2s(dst, tile + static_cast<size_t>(c) * CHUNK_BYTES, CHUNK_BYTES, &full[s]);
          for (int j = 0; j < e.ntok; ++j) {
            const __nv_bfloat16* src = p.x + static_cast<size_t>(e.tok[j]) * p.K + c * CHUNK_K;
            bulk_g2s(dst + CHUNK_BYTES + j * X_STRIDE * 2, src, CHUNK_K * 2, &full[s]);
          }
        }
      }
    }
    return;
  }

  // -------------------------------------------------------------- consumers
  const int mb = warp % 4, half = warp / 4;   // `half` = K slice index
  const int g = lane / 4, tq = lane % 4;   // fragment row group / thread-in-quad
  int it = 0;
  for (int t = role_idx; t < n_tiles; t += role_ctas) {
    const Expert& e = experts[t / tiles_per_exp];
    const int ntile = t % tiles_per_exp;
    const int ntok = e.ntok;
    const bool has_tok = g < ntok;          // this thread's B column = token g
    float acc[2][4] = {};
    for (int c = 0; c < chunks; ++c, ++it) {
      int s = it % stages;
      mbar_wait(&full[s], (it / stages) & 1);
      const unsigned char* st = ring + static_cast<size_t>(s) * STAGE_BYTES;
      const uint32_t* wq = reinterpret_cast<const uint32_t*>(st) + (mb * KB) * 32 + lane;
      const uint16_t* sc = reinterpret_cast<const uint16_t*>(st + W_BYTES) + (mb * GROUPS) * 8 + g;
      const __nv_bfloat16* xs = reinterpret_cast<const __nv_bfloat16*>(st + CHUNK_BYTES) + g * X_STRIDE + tq * 2;
      const int kb0 = half * (KB / KSPLIT);
      if (DIAG == 1) { __syncwarp(); if (lane == 0) mbar_arrive(&empty[s]); continue; }
#pragma unroll 4
      for (int kk = 0; kk < KB / KSPLIT; kk += 2) {
        const int grp = (kb0 + kk) / 2;               // one e8m0 group = 2 k16 blocks
        uint16_t sp = sc[grp * 8];
        uint32_t s_lo = scale_bf16x2(sp & 0xFF), s_hi = scale_bf16x2(sp >> 8);
#pragma unroll
        for (int u = 0; u < 2; ++u) {
          const int kb = kb0 + kk + u;
          uint32_t a[4];
          dequant(wq[kb * 32], s_lo, s_hi, a);
          uint32_t b0 = 0, b1 = 0;
          if (has_tok) {
            b0 = *reinterpret_cast<const uint32_t*>(xs + kb * 16);
            b1 = *reinterpret_cast<const uint32_t*>(xs + kb * 16 + 8);
          }
          mma_bf16(acc[u], a, b0, b1);
        }
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(&empty[s]);
    }
    // Reduce the two K halves: half 1 publishes, half 0 finishes the tile.
    float r[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) r[i] = acc[0][i] + acc[1][i];
    if (half > 0) {
      float* slot = red + (((half - 1) * 4 + mb) * 32 + lane) * 4;
#pragma unroll
      for (int i = 0; i < 4; ++i) slot[i] = r[i];
    }
    consumer_sync();
    if (half == 0) {
#pragma unroll
      for (int h = 0; h < KSPLIT - 1; ++h) {
        const float* slot = red + ((h * 4 + mb) * 32 + lane) * 4;
#pragma unroll
        for (int i = 0; i < 4; ++i) r[i] += slot[i];
      }
      // Thread holds rows (mb*16 + g) and (+8), tokens 2*tq and 2*tq+1.
      const int row0 = mb * 16 + g;
      if constexpr (EPI == RAW || EPI == WSUM) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          int tok = 2 * tq + (i & 1);
          int row = row0 + (i >> 1) * 8;
          if (tok < ntok) {
            int n = ntile * TILE_N + row;
            if constexpr (EPI == RAW) p.y_raw[static_cast<size_t>(e.route[tok]) * p.N + n] = r[i];
            else atomicAdd(&p.y_sum[static_cast<size_t>(e.tok[tok]) * p.N + n], e.wt[tok] * r[i]);
          }
        }
      } else {
        // ACT: warps 0-1 hold gate rows 0-31, warps 2-3 the matching up rows.
#pragma unroll
        for (int i = 0; i < 4; ++i) ex[(row0 + (i >> 1) * 8) * 8 + 2 * tq + (i & 1)] = r[i];
      }
    }
    if constexpr (EPI == ACT) {
      consumer_sync();
      for (int idx = threadIdx.x; idx < 32 * ntok; idx += CONSUMER_WARPS * 32) {
        int row = idx % 32, tok = idx / 32;
        float gt = ex[row * 8 + tok], up = ex[(row + 32) * 8 + tok];
        float v = gt / (1.f + __expf(-gt)) * up;
        p.y_act[static_cast<size_t>(e.route[tok]) * (p.N / 2) + ntile * 32 + row] = __float2bfloat16(v);
      }
    }
    consumer_sync();   // `red`/`ex` are reused by the next tile
  }
}

// Naive fp32 reference over the logical layout: y[route][n] = sum_k deq(w) x.
__global__ void reference_kernel(const uint8_t* codes, const uint8_t* scales, int N, int K,
                                 const __nv_bfloat16* x, int tok, float* y) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  const float lut[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  float acc = 0.f;
  for (int k = 0; k < K; ++k) {
    uint8_t cd = codes[static_cast<size_t>(n) * K + k];
    float w = lut[cd & 7] * ((cd & 8) ? -1.f : 1.f) * exp2f(float(scales[static_cast<size_t>(n) * (K / 32) + k / 32]) - 127.f);
    acc += w * __bfloat162float(x[static_cast<size_t>(tok) * K + k]);
  }
  y[n] = acc;
}

// ---------------------------------------------------------------- host side
// Pack a logical [N][K] matrix (codes one nibble per byte, e8m0 scales per 32)
// into tiles/chunks. `act_rows` orders tile rows as 32 gate + 32 matching up
// rows (w13 with the fused activation); otherwise rows are contiguous.
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
      uint32_t* wq = reinterpret_cast<uint32_t*>(blob);
      for (int mbk = 0; mbk < 4; ++mbk)
        for (int kb = 0; kb < KB; ++kb)
          for (int ln = 0; ln < 32; ++ln) {
            int r0 = mbk * 16 + ln / 4, r1 = r0 + 8;
            int k0 = c * CHUNK_K + kb * 16 + (ln % 4) * 2;
            auto nib = [&](int r, int k) { return codes[static_cast<size_t>(logical_row(r)) * K + k] & 0xF; };
            // reg0={n3,n7}=(r0,k0),(r0,k0+1); reg1={n2,n6}=(r1,k0),(r1,k0+1)
            // reg2={n1,n5}=(r0,k0+8),(r0,k0+9); reg3={n0,n4}=(r1,k0+8),(r1,k0+9)
            uint32_t q = 0;
            q |= uint32_t(nib(r0, k0)) << 12;     q |= uint32_t(nib(r0, k0 + 1)) << 28;
            q |= uint32_t(nib(r1, k0)) << 8;      q |= uint32_t(nib(r1, k0 + 1)) << 24;
            q |= uint32_t(nib(r0, k0 + 8)) << 4;  q |= uint32_t(nib(r0, k0 + 9)) << 20;
            q |= uint32_t(nib(r1, k0 + 8)) << 0;  q |= uint32_t(nib(r1, k0 + 9)) << 16;
            wq[(mbk * KB + kb) * 32 + ln] = q;
          }
      uint8_t* sb = blob + W_BYTES;   // [mb][group][8 pairs][row r, row r+8]
      for (int mbk = 0; mbk < 4; ++mbk)
        for (int gr = 0; gr < GROUPS; ++gr)
          for (int r = 0; r < 8; ++r)
            for (int h = 0; h < 2; ++h) {
              int row = mbk * 16 + r + 8 * h;
              sb[((mbk * GROUPS + gr) * 8 + r) * 2 + h] =
                  scales[static_cast<size_t>(logical_row(row)) * (K / 32) + c * GROUPS + gr];
            }
    }
  }
  return out;
}

struct Host {
  std::vector<uint8_t> codes, scales, packed;
};

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
  uint8_t* p = static_cast<uint8_t*>(aligned_alloc(1 << 21, (bytes + (1 << 21) - 1) / (1 << 21) * (1 << 21)));
  memset(p, 0, bytes);
  CK(cudaHostRegister(p, bytes, cudaHostRegisterMapped));
  uint8_t* d;
  CK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&d), p, 0));
  return d;
}

template <int EPI>
static void launch(const Params& p, int ctas, cudaStream_t st) {
  size_t smem = SMEM_HEAD + static_cast<size_t>(p.stages) * STAGE_BYTES;
  static bool set = false;
  if (!set) {
    CK(cudaFuncSetAttribute(tiered_moe_kernel<EPI>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    set = true;
  }
  tiered_moe_kernel<EPI><<<ctas, THREADS, smem, st>>>(p);
}

// Correctness: random experts split hot/cold, random tokens per expert, RAW
// and ACT epilogues against the reference.
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
    std::vector<__nv_bfloat16> xh(static_cast<size_t>(tokens) * K);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& v : xh) v = __float2bfloat16(nd(rng));
    __nv_bfloat16* xd; CK(cudaMalloc(&xd, xh.size() * 2));
    CK(cudaMemcpy(xd, xh.data(), xh.size() * 2, cudaMemcpyHostToDevice));
    Expert* ed; CK(cudaMalloc(&ed, sizeof(Expert) * n));
    CK(cudaMemcpy(ed, ex.data(), sizeof(Expert) * n, cudaMemcpyHostToDevice));
    float* yraw; CK(cudaMalloc(&yraw, sizeof(float) * routes * N));
    __nv_bfloat16* yact; CK(cudaMalloc(&yact, 2 * routes * (N / 2)));
    CK(cudaMemset(yraw, 0, sizeof(float) * routes * N));
    Params p{ed, n_hot, ed + n_hot, n_cold, cold_ctas, N, K, xd, yraw, yact, nullptr, stages};
    if (act) launch<ACT>(p, ctas, 0); else launch<RAW>(p, ctas, 0);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    // reference
    uint8_t *cd, *sd; float* yr;
    CK(cudaMalloc(&cd, static_cast<size_t>(N) * K)); CK(cudaMalloc(&sd, static_cast<size_t>(N) * K / 32));
    CK(cudaMalloc(&yr, sizeof(float) * N));
    double max_rel = 0; int bad = 0;
    for (int i = 0; i < n; ++i) {
      CK(cudaMemcpy(cd, hs[i].codes.data(), hs[i].codes.size(), cudaMemcpyHostToDevice));
      CK(cudaMemcpy(sd, hs[i].scales.data(), hs[i].scales.size(), cudaMemcpyHostToDevice));
      for (int j = 0; j < ex[i].ntok; ++j) {
        reference_kernel<<<(N + 127) / 128, 128>>>(cd, sd, N, K, xd, ex[i].tok[j], yr);
        std::vector<float> ref(N);
        CK(cudaMemcpy(ref.data(), yr, sizeof(float) * N, cudaMemcpyDeviceToHost));
        double scale = 0; for (float v : ref) scale = std::max(scale, (double)fabs(v));
        if (!act) {
          std::vector<float> got(N);
          CK(cudaMemcpy(got.data(), yraw + static_cast<size_t>(ex[i].route[j]) * N, sizeof(float) * N, cudaMemcpyDeviceToHost));
          for (int k = 0; k < N; ++k) {
            double rel = fabs(got[k] - ref[k]) / (scale + 1e-9);
            max_rel = std::max(max_rel, rel); bad += rel > 1e-2;
          }
        } else {
          std::vector<__nv_bfloat16> got(N / 2);
          CK(cudaMemcpy(got.data(), yact + static_cast<size_t>(ex[i].route[j]) * (N / 2), N, cudaMemcpyDeviceToHost));
          for (int k = 0; k < N / 2; ++k) {
            float gt = ref[k], up = ref[N / 2 + k];
            float want = gt / (1.f + expf(-gt)) * up;
            double rel = fabs(__bfloat162float(got[k]) - want) / (scale * scale * 0.5 + 1e-9);
            max_rel = std::max(max_rel, rel); bad += rel > 2e-2;
          }
        }
      }
    }
    printf("check %s N=%d K=%d hot=%d cold=%d: max rel err %.2e, %d bad -> %s\n", act ? "ACT" : "RAW",
           N, K, n_hot, n_cold, max_rel, bad, bad ? "FAIL" : "ok");
    ok &= bad == 0;
  }
  return ok ? 0 : 1;
}

// Timing: CUDA graph of `reps` launches, each over a different set of experts
// drawn from pools larger than L2, so nothing is reused across launches.
static void bench(int N, int K, int n_hot, int n_cold, int cold_ctas, int stages, int ntok_mode,
                  bool act_rows) {
  const int ctas = 132, reps = 20, pool_hot = 48, pool_cold = 24;
  std::mt19937 rng(7);
  Host proto = make_expert(N, K, act_rows, rng);
  size_t bytes = proto.packed.size();
  std::vector<uint8_t*> hot_pool(pool_hot), cold_pool(pool_cold);
  for (auto& d : hot_pool) { CK(cudaMalloc(&d, bytes)); CK(cudaMemcpy(d, proto.packed.data(), bytes, cudaMemcpyHostToDevice)); }
  for (auto& d : cold_pool) { d = host_pinned(bytes); CK(cudaMemcpy(d, proto.packed.data(), bytes, cudaMemcpyHostToDevice)); }
  const int tokens = 8;
  __nv_bfloat16* xd; CK(cudaMalloc(&xd, static_cast<size_t>(64) * K * 2)); CK(cudaMemset(xd, 0, static_cast<size_t>(64) * K * 2));
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
    Params p{lists[r], n_hot, lists[r] + n_hot, n_cold, n_cold ? cold_ctas : 0, N, K, xd, yraw, yact, ysum, stages};
    if (act_rows) launch<ACT>(p, ctas, st); else launch<WSUM>(p, ctas, st);
  };
  run(0); CK(cudaStreamSynchronize(st));
  cudaGraph_t graph; cudaGraphExec_t exec;
  CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
  for (int r = 0; r < reps; ++r) run(r);
  CK(cudaStreamEndCapture(st, &graph));
  CK(cudaGraphInstantiate(&exec, graph, 0));
  cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  CK(cudaGraphLaunch(exec, st)); CK(cudaStreamSynchronize(st));
  float best = 1e30f;
  for (int trial = 0; trial < 5; ++trial) {
    CK(cudaEventRecord(a, st)); CK(cudaGraphLaunch(exec, st)); CK(cudaEventRecord(b, st));
    CK(cudaEventSynchronize(b));
    float ms; CK(cudaEventElapsedTime(&ms, a, b)); best = std::min(best, ms);
  }
  double us = best * 1000.0 / reps;
  double hb = double(n_hot) * bytes, cb = double(n_cold) * bytes;
  double sol = std::max(hb / 3.626e12, cb / 0.409e12) * 1e6;
  printf("bench %s hot=%2d cold=%d tok=%s cold_ctas=%2d stages=%d: %7.1f us  (SOL %6.1f us, %3.0f%%)  hot %.0f GB/s-eq cold-share\n",
         act_rows ? "w13" : "w2 ", n_hot, n_cold, ntok_mode ? std::to_string(ntok_mode).c_str() : "dist", cold_ctas,
         stages, us, sol, sol / us * 100, (hb + cb) / (us * 1e-6) / 1e9);
}

int main(int argc, char** argv) {
  if (argc < 2) { printf("usage: tiered_moe check | bench <w13|w2> hot cold cold_ctas stages ntok(0=dist)\n"); return 1; }
  if (!strcmp(argv[1], "check")) {
    int rc = 0;
    rc |= check(4096, 6144, 3, 2, 16);   // w13 shape
    rc |= check(6144, 2048, 2, 3, 24);   // w2 shape
    rc |= check(4096, 6144, 4, 0, 0);    // hot only
    rc |= check(4096, 6144, 0, 2, 132);  // cold only
    return rc;
  }
  bool w13 = !strcmp(argv[2], "w13");
  int n_hot = atoi(argv[3]), n_cold = atoi(argv[4]), cold_ctas = atoi(argv[5]), stages = atoi(argv[6]);
  int ntok = argc > 7 ? atoi(argv[7]) : 0;
  if (w13) bench(4096, 6144, n_hot, n_cold, cold_ctas, stages, ntok, true);
  else bench(6144, 2048, n_hot, n_cold, cold_ctas, stages, ntok, false);
  return 0;
}

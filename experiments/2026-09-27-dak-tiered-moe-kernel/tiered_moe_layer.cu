// Fused MiMo MoE layer, one launch (v8): w13 -> silu*up -> w2 over hot (HBM) and
// cold (pinned Grace, UVA over C2C) experts, from the v7 stream-K GEMM.
//
// Why fuse: in cold-bound layers (most of MiMo decode) C2C is the roofline, and
// the w13 -> w2 kernel boundary leaves it idle (w13 tail, act kernel, w2 launch
// and ramp). Here each CTA keeps one tier for both phases:
//  * w13 phase: stream-K over (tile, chunk) units. A w13 tile holds 32 gate
//    rows and the 32 matching up rows, so 32 activations depend on one tile.
//    Warps add raw sums into y13 with red.global.add and count the chunks they
//    finished per tile; the warp that completes a tile computes silu(gate) * up
//    for its 32 indices and every route, rounds them to f16 under a
//    power-of-two scale per (route, 32-index group) -- exactly one w2 scale
//    group, applied in fp32 beside the weight's e8m0 scale -- zeroes the y13
//    entries, and bumps ready[expert][k half]. The handoff is 32 elements, not
//    a pass over the whole expert.
//  * w2 phase: the producer keeps issuing weight chunks (they don't depend on
//    activations) as ring slots free, and adds a unit's activation rows and
//    group scales once the 32 w13 tiles behind its K half are done; the link
//    keeps streaming across the phase change. Warps add wt * y into the
//    per-token output.
//  * A helper warp does the tile handoff: the consumer that completes a tile
//    only queues it in shared memory and keeps streaming.
// Launched cooperatively: w2 units wait on other CTAs' w13 work, so all CTAs
// must be resident (one per SM). The last CTA out resets the sync counters.
//
// Numerics as v7 (exact products, fp32 accumulation); the act rows are rounded
// to f16 (Marlin rounds them to bf16).
//
//   tiered_moe_layer check | bench hot cold cold_ctas [ntok, 0 = decode mix]

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
#include <chrono>
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
constexpr int XROW_STRIDE = XROW_BYTES + 64;          // rows g, g+1 (one LDS.128 phase) on disjoint banks
constexpr int X_BYTES = MAX_TOK * XROW_STRIDE;
constexpr int SX_BYTES = MAX_TOK * GROUPS * 4;         // w2: per (route, group) activation scales
constexpr int STAGE_BYTES = CHUNK_BYTES + X_BYTES + SX_BYTES;
#ifndef CONSUMER_WARPS_DEF
#define CONSUMER_WARPS_DEF 8
#endif
constexpr int CONSUMER_WARPS = CONSUMER_WARPS_DEF;
constexpr int KSPLIT = CONSUMER_WARPS / 4;
constexpr int PRODUCER = CONSUMER_WARPS, HELPER = CONSUMER_WARPS + 1;   // warp roles
constexpr int THREADS = (CONSUMER_WARPS + 2) * 32;
constexpr int QCAP = 128;   // finished w13 tiles per CTA handed to the helper
constexpr int SMEM_HEAD = 256;
#ifndef DIAG
#define DIAG 0
#endif

#ifndef PROF
#define PROF 0
#endif
// PROF=1, summed over CTAs (consumer = warp 0): [0] consumer wait-full cycles, [1] consumer cycles,
// [2] producer wait-empty cycles, [3] producer cycles, [4] flush cycles, [5] CTAs, [6] consumer ns
__device__ unsigned long long g_prof[8];
__device__ unsigned long long g_lprof[8];
__device__ unsigned long long g_lprof2[8];  // [0] max w13 end, [1] min w13 end, [2] producer ready-wait ns, [3] max end   // PROF: layer-kernel timeline, see bench
__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t;
}


struct Expert {
  const uint8_t* w13;       // packed [64 tiles][6 chunks] blobs
  const uint8_t* w2;        // packed [96 tiles][2 chunks] blobs
  int ntok;
  int tok[MAX_TOK];         // token rows (w13 input, w2 output)
  int route[MAX_TOK];       // route rows (w13 output, w2 input)
  float wt[MAX_TOK];        // router weight
};

struct Params {
  const Expert* hot; int n_hot;
  const Expert* cold; int n_cold;
  int cold_ctas;
  const uint8_t* x13; const float* xs13;   // token rows, f16 B layout (prep_kernel)
  float* y13;                              // [routes][4096] gate|up sums: zero on entry, left zero
  uint8_t* x2; float* xs2;                 // [routes][2048] act rows, [routes][64] group scales: written in-kernel
  float* y;                                // [tokens][6144] output: zero on entry
  int* sync;                               // [E][64] w13 tile progress, [E][2] ready halves, [1] done: zero on entry, left zero
  int stages;
};

template <int PH> struct Phase;            // MiMo-V2: hidden 6144, moe intermediate 2048
template <> struct Phase<0> { static constexpr int N = 4096, K = 6144; };
template <> struct Phase<1> { static constexpr int N = 6144, K = 2048; };
template <int PH> constexpr int TILES = Phase<PH>::N / TILE_N;
template <int PH> constexpr int CHUNKS = Phase<PH>::K / CHUNK_K;
constexpr int TILE_WARP_CHUNKS = CHUNKS<0> * CONSUMER_WARPS;   // w13 tile done: every warp counted every chunk
constexpr int HALF_TILES = TILES<0> / CHUNKS<1>;                // w13 tiles behind one w2 chunk (32)
static_assert(TILE_N / 2 == 32 && GROUPS == HALF_TILES, "a w13 tile feeds exactly one w2 group");

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
#ifndef E5M2
#define E5M2 1
#endif
#if E5M2
// e2m1 at e5m2 bit positions (value * 2^-14, exact; 0.5 is an f16 subnormal):
// e5m2 is the high byte of an f16, so a byte permute with zeros builds the
// f16x2 A register (PRMT, 2.1 cycles/warp-instr vs 2.6 for the e4m3 F2FP).
// Slot a: s:7 e1:3 e0:2 m:1. Slot b: s:6 e1:5 e0:4 m:0.
constexpr int PLACEMENT_EXP = 14;
__device__ __forceinline__ uint32_t f16x2_lo(uint32_t v) {
  uint32_t out; asm("prmt.b32 %0, %1, 0, 0x1404;" : "=r"(out) : "r"(v)); return out;
}
__device__ __forceinline__ uint32_t f16x2_hi(uint32_t v) {
  uint32_t out; asm("prmt.b32 %0, %1, 0, 0x3424;" : "=r"(out) : "r"(v)); return out;
}
__device__ __forceinline__ uint32_t reg_a(uint32_t w) { return w & 0x8E8E8E8Eu; }
__device__ __forceinline__ uint32_t reg_b(uint32_t w) {
  return ((w << 1) & 0x82828282u) | ((w >> 2) & 0x0C0C0C0Cu);
}
#else
// e2m1 at e4m3 bit positions (value * 2^-6): two e4m3 in the low / high 16 bits -> f16x2.
constexpr int PLACEMENT_EXP = 6;
__device__ __forceinline__ uint32_t f16x2_lo(uint32_t v) {
  uint32_t out;
  asm("{ .reg .b16 lo, hi; mov.b32 {lo, hi}, %1; cvt.rn.f16x2.e4m3x2 %0, lo; }" : "=r"(out) : "r"(v));
  return out;
}
__device__ __forceinline__ uint32_t f16x2_hi(uint32_t v) {
  uint32_t out;
  asm("{ .reg .b16 lo, hi; mov.b32 {lo, hi}, %1; cvt.rn.f16x2.e4m3x2 %0, hi; }" : "=r"(out) : "r"(v));
  return out;
}
__device__ __forceinline__ uint32_t reg_a(uint32_t w) { return w & 0x9C9C9C9Cu; }
__device__ __forceinline__ uint32_t reg_b(uint32_t w) {
  return ((w << 2) & 0x8C8C8C8Cu) | ((w >> 2) & 0x10101010u);
}
#endif
// fp32 2^(e-127); the placement's 2^-PLACEMENT_EXP is undone by the activation scale.
__device__ __forceinline__ float group_scale(uint32_t e) { return __uint_as_float(e << 23); }

__device__ __forceinline__ bool mbar_test(uint64_t* bar, uint32_t parity) {
  uint32_t ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
               : "=r"(ok) : "r"(smem_u32(bar)), "r"(parity) : "memory");
  return ok;
}
__device__ __forceinline__ int ld_acquire(const int* p) {
  int v; asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v;
}
__device__ __forceinline__ void st_release(int* p, int v) {
  asm volatile("st.release.gpu.global.b32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}

// ---------------------------------------------------------------- the kernel
// Expert-major split: a CTA takes the same slice [o, o + L) of every expert's
// (tile, chunk) units, expert after expert, so experts finish w13 one after
// another across the whole tier and early experts' w2 inputs are ready long
// before any CTA reaches w2. Flat index k -> unit (k / L) * per + o + k % L.
struct Work {
  const Expert* ex; int eid0;   // eid0: first expert id of the tier
  int L, o, count;
  template <int PH> __device__ int unit(int k) const { return (k / L) * TILES<PH> * CHUNKS<PH> + o + k % L; }
};

template <int PH>
__device__ __forceinline__ Work work_of(const Params& p) {
  const int per = TILES<PH> * CHUNKS<PH>;
  const bool cold = static_cast<int>(blockIdx.x) < p.cold_ctas;
  const int n = cold ? p.n_cold : p.n_hot;
  const int ctas = cold ? p.cold_ctas : gridDim.x - p.cold_ctas;
  const int idx = cold ? blockIdx.x : blockIdx.x - p.cold_ctas;
  Work w{cold ? p.cold : p.hot, cold ? p.n_hot : 0, 0, 0, 0};
  if (ctas > 0) {
    w.o = per * idx / ctas;
    w.L = per * (idx + 1) / ctas - w.o;
    w.count = n * w.L;
  }
  return w;
}

// Issue one unit's weight chunk into stage slot s (tx covers the activation rows too).
template <int PH>
__device__ __forceinline__ void issue_weights(const Work& w, int u, unsigned char* dst, uint64_t* full) {
  const int t = u / CHUNKS<PH>, c = u - t * CHUNKS<PH>;
  const Expert& e = w.ex[t / TILES<PH>];
  const uint8_t* base = PH == 0 ? e.w13 : e.w2;
  const uint8_t* tile = base + static_cast<size_t>(t % TILES<PH>) * CHUNKS<PH> * CHUNK_BYTES;
  mbar_expect_tx(full, CHUNK_BYTES + e.ntok * (XROW_BYTES + (PH == 1 ? GROUPS * 4 : 0)));
  bulk_g2s(dst, tile + static_cast<size_t>(c) * CHUNK_BYTES, CHUNK_BYTES, full);
}
template <int PH>
__device__ __forceinline__ void issue_rows(const Params& p, const Work& w, int u, unsigned char* dst, uint64_t* full) {
  const int t = u / CHUNKS<PH>, c = u - t * CHUNKS<PH>;
  const Expert& e = w.ex[t / TILES<PH>];
  const uint8_t* x = PH == 0 ? p.x13 : p.x2;
  for (int j = 0; j < e.ntok; ++j)
    bulk_g2s(dst + CHUNK_BYTES + j * XROW_STRIDE,
             x + static_cast<size_t>(PH == 0 ? e.tok[j] : e.route[j]) * Phase<PH>::K * 2 +
                 static_cast<size_t>(c) * XROW_BYTES,
             XROW_BYTES, full);
  if (PH == 1)
    for (int j = 0; j < e.ntok; ++j)
      bulk_g2s(dst + CHUNK_BYTES + X_BYTES + j * GROUPS * 4, p.xs2 + static_cast<size_t>(e.route[j]) * 64 + c * GROUPS,
               GROUPS * 4, full);
}

// One warp, one finished w13 tile: silu(gate) * up for its 32 indices and every
// route of the expert, as f16 under a power-of-two scale per (route, group).
__device__ void act_tile(const Params& p, const Expert& e, int tile, int lane) {
  constexpr int I = Phase<1>::K;
  const int i = tile * 32 + lane;   // w2 K index
  const int kb = i / 16, q = i % 16, tq = (q % 8) / 2, reg = q / 8, h = q % 2;
  const size_t slot = ((static_cast<size_t>(kb / 2) * 4 + tq) * 2 + kb % 2) * 4 + reg * 2 + h;
  const int ntok = e.ntok;
  int r[MAX_TOK];
  float g[MAX_TOK], u[MAX_TOK];
#pragma unroll
  for (int j = 0; j < MAX_TOK; ++j) {   // every route's loads in flight together
    r[j] = j < ntok ? e.route[j] : 0;
    const float* yr = p.y13 + static_cast<size_t>(r[j]) * Phase<0>::N + tile * TILE_N;
    g[j] = j < ntok ? __ldcg(yr + lane) : 0.f;
    u[j] = j < ntok ? __ldcg(yr + 32 + lane) : 0.f;
  }
#pragma unroll
  for (int j = 0; j < MAX_TOK; ++j) {
    if (j >= ntok) break;
    const float a = g[j] / (1.f + __expf(-g[j])) * u[j];
    float m = fabsf(a);
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    const int t = m > 0.f ? static_cast<int>(ceilf(log2f(m))) - 13 : 0;
    reinterpret_cast<__half*>(p.x2 + static_cast<size_t>(r[j]) * I * 2)[slot] =
        __float2half_rn(a * exp2f(static_cast<float>(-t)));
    if (lane == 0) p.xs2[static_cast<size_t>(r[j]) * 64 + tile] = exp2f(static_cast<float>(t + PLACEMENT_EXP));
    float* yr = p.y13 + static_cast<size_t>(r[j]) * Phase<0>::N + tile * TILE_N;
    yr[lane] = 0.f;
    yr[32 + lane] = 0.f;
  }
}

template <int PH>
__device__ __forceinline__ void flush(const Params& p, const Expert& e, int n0, int g, int tq, float* acc) {
  constexpr int N = Phase<PH>::N;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int tok = 2 * tq + (i & 1);
    if (tok < e.ntok) {
      const int n = n0 + g + (i >> 1) * 8;
      if constexpr (PH == 0)
        atomicAdd(&p.y13[static_cast<size_t>(e.route[tok]) * N + n], acc[i] * p.xs13[e.tok[tok]]);
      else
        atomicAdd(&p.y[static_cast<size_t>(e.tok[tok]) * N + n], e.wt[tok] * acc[i]);
    }
    acc[i] = 0.f;
  }
}

// Consumer warp: one phase's units, continuing the ring position `it`.
template <int PH>
__device__ __forceinline__ void consume(const Params& p, const Work& w, int& it, uint64_t* full, uint64_t* empty,
                                        const unsigned char* ring, int stages, int warp, int lane,
                                        volatile int* queue, int* q_tail) {
  const int mb = warp % 4, slice = warp / 4;
  const int g = lane / 4, tq = lane % 4;
  constexpr int GS = GROUPS / KSPLIT;
  const int g0 = slice * GS;
  int* progress = p.sync;
  float acc[4] = {};
  int seg = 0;
  for (int k = 0; k < w.count; ++k, ++it) {
    const int u = w.unit<PH>(k);
    const int t = u / CHUNKS<PH>, c = u - t * CHUNKS<PH>;
    const int s = it % stages;
    mbar_wait(&full[s], (it / stages) & 1);
    const unsigned char* st = ring + static_cast<size_t>(s) * STAGE_BYTES;
    const uint4* wq = reinterpret_cast<const uint4*>(st) + (mb * (KB / 4) + g0 / 2) * 32 + lane;
    const uint16_t* sc = reinterpret_cast<const uint16_t*>(st + W_BYTES) + (mb * GROUPS + g0) * 8 + g;
    // rows g >= ntok hold stale data; they only feed output columns never flushed
    const uint4* xs = reinterpret_cast<const uint4*>(st + CHUNK_BYTES + g * XROW_STRIDE) + g0 * 4 + tq;
    const float* sx = reinterpret_cast<const float*>(st + CHUNK_BYTES + X_BYTES) + 2 * tq * GROUPS + g0;
#pragma unroll
    for (int j = 0; j < GS; ++j) {
      uint32_t sp = sc[j * 8];
      const uint4 wv = wq[(j / 2) * 32];
      const uint4 xv = xs[j * 4];
      const float zero[4] = {0.f, 0.f, 0.f, 0.f};
      float d[4];
#pragma unroll
      for (int v = 0; v < 2; ++v) {
        const uint32_t wd = (j & 1) ? (v ? wv.w : wv.z) : (v ? wv.y : wv.x);
        const uint2 xb = v ? make_uint2(xv.z, xv.w) : make_uint2(xv.x, xv.y);
        uint32_t ra = reg_a(wd), rb = reg_b(wd);
        uint32_t a[4] = {f16x2_lo(ra), f16x2_lo(rb), f16x2_hi(ra), f16x2_hi(rb)};
        mma_f16(d, a, xb.x, xb.y, v ? d : zero);
      }
      float s0 = group_scale(sp & 0xFFu), s1 = group_scale(sp >> 8);
      if constexpr (PH == 1) {   // columns 2tq, 2tq+1 carry their own activation group scale
        const float x0 = sx[j], x1 = sx[GROUPS + j];
        acc[0] = fmaf(s0 * x0, d[0], acc[0]);
        acc[1] = fmaf(s0 * x1, d[1], acc[1]);
        acc[2] = fmaf(s1 * x0, d[2], acc[2]);
        acc[3] = fmaf(s1 * x1, d[3], acc[3]);
      } else {
        acc[0] = fmaf(s0, d[0], acc[0]);
        acc[1] = fmaf(s0, d[1], acc[1]);
        acc[2] = fmaf(s1, d[2], acc[2]);
        acc[3] = fmaf(s1, d[3], acc[3]);
      }
    }
    __syncwarp();
    if (lane == 0) mbar_arrive(&empty[s]);
    ++seg;
    if (c == CHUNKS<PH> - 1 || k % w.L == w.L - 1) {
      const int ei = t / TILES<PH>;
      const Expert& e = w.ex[ei];
      flush<PH>(p, e, (t % TILES<PH>) * TILE_N + mb * 16, g, tq, acc);
      if constexpr (PH == 0) {
        // publish this warp's y13 sums, then count them; the warp that completes
        // the tile turns its 32 indices into w2 activations
        const int tile = t % TILES<0>, eid = w.eid0 + ei;
        __threadfence();
        __syncwarp();
        int done = 0;
        if (lane == 0) done = atomicAdd(&progress[eid * TILES<0> + tile], seg) + seg == TILE_WARP_CHUNKS;
        if (lane == 0 && done) {   // hand the tile to the helper warp
          const int slot = atomicAdd(q_tail, 1);
          queue[slot] = (eid << 8) | tile | (1 << 30);
          __threadfence_block();
        }
      }
      seg = 0;
    }
  }
}

__global__ void __launch_bounds__(THREADS, 1) tiered_moe_layer_kernel(Params p) {
  extern __shared__ __align__(128) unsigned char smem[];
  const int stages = p.stages;
  uint64_t* full = reinterpret_cast<uint64_t*>(smem);
  uint64_t* empty = full + stages;
  unsigned char* ring = smem + SMEM_HEAD;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const Work w13 = work_of<0>(p), w2 = work_of<1>(p);
  const int E = p.n_hot + p.n_cold;
  int* ready = p.sync + E * TILES<0>;
  const int n_sync = E * TILES<0> + E * 2;

  __shared__ int queue[QCAP];
  __shared__ int q_tail, w13_done;
  for (int i = threadIdx.x; i < QCAP; i += blockDim.x) queue[i] = 0;
  if (threadIdx.x == 0) {
    q_tail = 0; w13_done = 0;
    for (int s = 0; s < stages; ++s) { mbar_init(&full[s], 1); mbar_init(&empty[s], CONSUMER_WARPS); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();

  if (warp == HELPER) {
    volatile int* vq = queue;
    volatile int* vdone = &w13_done;
    for (int head = 0;;) {
      const int v = vq[head];
      if (v) {
        __threadfence();   // the completing warp saw every contribution through the tile counter
        const int eid = (v >> 8) & 0x3FFFFF, tile = v & 0xFF;
        const Expert& e = eid < p.n_hot ? p.hot[eid] : p.cold[eid - p.n_hot];
        const unsigned long long a0 = PROF ? gtimer() : 0;
        act_tile(p, e, tile, lane);
        __threadfence();
        __syncwarp();
        if (lane == 0) atomicAdd(&ready[eid * 2 + tile / HALF_TILES], 1);
        if (PROF && lane == 0) {
          const unsigned long long dt = gtimer() - a0;
          atomicAdd(&g_lprof[4], 1ull); atomicAdd(&g_lprof[5], dt); atomicMax(&g_lprof[3], dt);
          atomicMax(&g_lprof[7], gtimer());
        }
        ++head;
        continue;
      }
      if (*vdone == CONSUMER_WARPS && vq[head] == 0) break;
      __nanosleep(64);
    }
  } else if (warp == PRODUCER) {
    if (lane == 0) {
      int it = 0;
      for (int k = 0; k < w13.count; ++k, ++it) {
        const int u = w13.unit<0>(k);
        const int s = it % stages;
        if (it >= stages) mbar_wait(&empty[s], ((it / stages) - 1) & 1);
        unsigned char* dst = ring + static_cast<size_t>(s) * STAGE_BYTES;
        issue_weights<0>(w13, u, dst, &full[s]);
        issue_rows<0>(p, w13, u, dst, &full[s]);
      }
      // w2: weights run ahead as slots free; rows follow once their expert is ready
      int kw = 0, kx = 0, itw = it;
      unsigned long long wait0 = 0, waited = 0;
      while (kx < w2.count) {
        bool moved = false;
        if (kw < w2.count) {
          const int s = itw % stages;
          if (itw < stages || mbar_test(&empty[s], ((itw / stages) - 1) & 1)) {
            issue_weights<1>(w2, w2.unit<1>(kw), ring + static_cast<size_t>(s) * STAGE_BYTES, &full[s]);
            ++kw; ++itw; moved = true;
          }
        }
        if (kx < kw) {
          const int ux = w2.unit<1>(kx);
          const int ei = ux / CHUNKS<1> / TILES<1>, c = ux % CHUNKS<1>;
          if (ld_acquire(&ready[(w2.eid0 + ei) * 2 + c]) == HALF_TILES) {
            asm volatile("fence.proxy.async.global;" ::: "memory");   // TMA reads rows written by st.global
            const int s = (it + kx) % stages;
            issue_rows<1>(p, w2, ux, ring + static_cast<size_t>(s) * STAGE_BYTES, &full[s]);
            ++kx; moved = true;
          }
        }
        if (!moved) { if (PROF && !wait0) wait0 = gtimer(); __nanosleep(32); }
        else if (PROF && wait0) { waited += gtimer() - wait0; wait0 = 0; }
      }
      if (PROF) atomicAdd(&g_lprof2[2], waited);
    }
  } else {
    int it = 0;
    const unsigned long long t0 = PROF ? gtimer() : 0;
    consume<0>(p, w13, it, full, empty, ring, stages, warp, lane, queue, &q_tail);
    __syncwarp();
    if (lane == 0) { __threadfence_block(); atomicAdd(&w13_done, 1); }
    const unsigned long long t1 = PROF ? gtimer() : 0;
    consume<1>(p, w2, it, full, empty, ring, stages, warp, lane, queue, &q_tail);
    if (PROF && threadIdx.x == 0) {
      atomicMax(&g_lprof2[0], t1); atomicMin(&g_lprof2[1], t1); atomicMax(&g_lprof2[3], gtimer());
      atomicAdd(&g_lprof[0], t1 - t0); atomicAdd(&g_lprof[2], gtimer() - t0); atomicAdd(&g_lprof[6], 1ull);
      atomicMin(&g_lprof[1], t0);   // kernel start (absolute)
    }
  }

  // the last CTA out leaves the sync words zero for the next launch
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    if (atomicAdd(&p.sync[n_sync], 1) == static_cast<int>(gridDim.x) - 1) {
      for (int i = 0; i < n_sync; ++i) p.sync[i] = 0;
      p.sync[n_sync] = 0;
      __threadfence();
    }
  }
}

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
  if (threadIdx.x == 0) xscale[row] = exp2f(static_cast<float>(t + PLACEMENT_EXP));  // weights sit at 2^-PLACEMENT_EXP
  __half* out = reinterpret_cast<__half*>(x8 + static_cast<size_t>(row) * K * 2);
  // element k: kb = k/16, r = k%16; tq = (r%8)/2, reg = r/8, half = r%2
  for (int k = threadIdx.x; k < K; k += blockDim.x) {
    float v = __bfloat162float(xr[k]) * inv;
    int kb = k / 16, r = k % 16, tq = (r % 8) / 2, reg = r / 8, h = r % 2;
    out[((static_cast<size_t>(kb / 2) * 4 + tq) * 2 + kb % 2) * 4 + reg * 2 + h] = __float2half_rn(v);
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
#if E5M2
static inline uint8_t enc_a(int c) { return uint8_t(((c & 8) << 4) | ((c & 7) << 1)); }     // s:7 e1:3 e0:2 m:1
static inline uint8_t enc_b(int c) {                                                        // s:6 e1:5 e0:4 m:0
  int s = (c >> 3) & 1, e1 = (c >> 2) & 1, e0 = (c >> 1) & 1, m = c & 1;
  return uint8_t((s << 6) | (e1 << 5) | (e0 << 4) | m);
}
#else
static inline uint8_t enc_a(int c) { return uint8_t(((c & 8) << 4) | ((c & 7) << 2)); }     // {7,4,3,2}
static inline uint8_t enc_b(int c) {                                                        // s:5 e1:6 e0:1 m:0
  int s = (c >> 3) & 1, e1 = (c >> 2) & 1, e0 = (c >> 1) & 1, m = c & 1;
  return uint8_t((s << 5) | (e1 << 6) | (e0 << 1) | m);
}
#endif

// interleave: w13 tile t holds gate rows 32t..32t+31 then up rows N/2 + 32t..
static std::vector<uint8_t> pack(const std::vector<uint8_t>& codes, const std::vector<uint8_t>& scales,
                                 int N, int K, bool interleave) {
  int tiles = N / TILE_N, chunks = K / CHUNK_K;
  std::vector<uint8_t> out(static_cast<size_t>(tiles) * chunks * CHUNK_BYTES);
  for (int t = 0; t < tiles; ++t) {
    auto logical_row = [&](int r) {
      if (!interleave) return t * TILE_N + r;
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
            reinterpret_cast<uint32_t*>(blob)[((mbk * (KB / 4) + kb / 4) * 32 + ln) * 4 + kb % 4] = w;
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

static Host make_expert(int N, int K, std::mt19937& rng, bool interleave = false) {
  Host h;
  h.codes.resize(static_cast<size_t>(N) * K);
  h.scales.resize(static_cast<size_t>(N) * (K / 32));
  // e8m0 exponents drawn from MiMo-V2.6's expert scales (60 tensors, layers 3-68): 111..124
  std::uniform_int_distribution<int> nib(0, 15);
  std::discrete_distribution<int> sc({6.61e-06, 3.59e-04, 3.13e-03, 1.43e-02, 3.28e-02, 7.83e-02, 5.18e-02,
                                      1.49e-02, 1.37e-02, 6.70e-01, 1.20e-01, 4.26e-04, 1.55e-05, 5.51e-07});
  for (auto& v : h.codes) v = nib(rng);
  for (auto& v : h.scales) v = 111 + sc(rng);
  h.packed = pack(h.codes, h.scales, N, K, interleave);
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


static void launch(const Params& p, cudaStream_t st) {
  const size_t smem = SMEM_HEAD + static_cast<size_t>(p.stages) * STAGE_BYTES;
  static bool set = false;
  if (!set) {
    CK(cudaFuncSetAttribute(tiered_moe_layer_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    set = true;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(132); cfg.blockDim = dim3(THREADS); cfg.dynamicSmemBytes = smem; cfg.stream = st;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeCooperative; attr[0].val.cooperative = 1;   // w2 waits on other CTAs' w13
  cfg.attrs = attr; cfg.numAttrs = 1;
  CK(cudaLaunchKernelEx(&cfg, tiered_moe_layer_kernel, p));
}

struct Layer {   // device buffers for one layer call
  uint8_t* x13; float* xs13; float* y13; uint8_t* x2; float* xs2; float* y; int* sync;
};
static Layer make_layer(int tokens, int routes, int experts) {
  Layer l;
  CK(cudaMalloc(&l.x13, size_t(tokens) * Phase<0>::K * 2)); CK(cudaMalloc(&l.xs13, tokens * sizeof(float)));
  CK(cudaMalloc(&l.y13, size_t(routes) * Phase<0>::N * 4)); CK(cudaMemset(l.y13, 0, size_t(routes) * Phase<0>::N * 4));
  CK(cudaMalloc(&l.x2, size_t(routes) * Phase<1>::K * 2)); CK(cudaMalloc(&l.xs2, size_t(routes) * 64 * sizeof(float)));
  CK(cudaMalloc(&l.y, size_t(tokens) * Phase<1>::N * 4));
  const size_t n_sync = size_t(experts) * (TILES<0> + 2) + 1;
  CK(cudaMalloc(&l.sync, n_sync * sizeof(int))); CK(cudaMemset(l.sync, 0, n_sync * sizeof(int)));
  return l;
}

static float silu_mul(float g, float u) { return g / (1.f + std::exp(-g)) * u; }

// End to end against fp32: y[tok] = sum over its routes of wt * W2 (silu(W13_g x) * W13_u x).
static int check(int n_hot, int n_cold, int cold_ctas) {
  std::mt19937 rng(4321);
  const int T = 8, n = n_hot + n_cold;
  std::vector<Host> h13, h2;
  for (int i = 0; i < n; ++i) { h13.push_back(make_expert(Phase<0>::N, Phase<0>::K, rng, true)); h2.push_back(make_expert(Phase<1>::N, Phase<1>::K, rng)); }
  std::vector<Expert> ex(n);
  std::uniform_int_distribution<int> ntok(1, 8);
  std::uniform_real_distribution<float> wt(0.05f, 1.f);
  int routes = 0;
  for (int i = 0; i < n; ++i) {
    ex[i].ntok = ntok(rng) <= 4 ? 1 : ntok(rng);
    std::vector<int> perm(T);
    for (int j = 0; j < T; ++j) perm[j] = j;
    std::shuffle(perm.begin(), perm.end(), rng);
    for (int j = 0; j < ex[i].ntok; ++j) { ex[i].tok[j] = perm[j]; ex[i].route[j] = routes++; ex[i].wt[j] = wt(rng); }
    for (int m = 0; m < 2; ++m) {
      const std::vector<uint8_t>& src = m ? h2[i].packed : h13[i].packed;
      uint8_t* d;
      if (i < n_hot) CK(cudaMalloc(&d, src.size())); else d = host_pinned(src.size());
      CK(cudaMemcpy(d, src.data(), src.size(), cudaMemcpyHostToDevice));
      (m ? ex[i].w2 : ex[i].w13) = d;
    }
  }
  // activations: rows at different magnitudes
  std::vector<__nv_bfloat16> xh(size_t(T) * Phase<0>::K);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::uniform_real_distribution<float> mag(-3.f, 3.f);
  for (int r = 0; r < T; ++r) {
    float rs = exp2f(mag(rng));
    for (int k = 0; k < Phase<0>::K; ++k) xh[size_t(r) * Phase<0>::K + k] = __float2bfloat16(nd(rng) * rs * 0.05f);
  }
  __nv_bfloat16* xd; CK(cudaMalloc(&xd, xh.size() * 2)); CK(cudaMemcpy(xd, xh.data(), xh.size() * 2, cudaMemcpyHostToDevice));
  Layer l = make_layer(T, routes, n);
  prep_kernel<<<T, 256>>>(xd, Phase<0>::K, l.x13, l.xs13);
  Expert* ed; CK(cudaMalloc(&ed, sizeof(Expert) * n)); CK(cudaMemcpy(ed, ex.data(), sizeof(Expert) * n, cudaMemcpyHostToDevice));
  Params p{ed, n_hot, ed + n_hot, n_cold, n_cold ? cold_ctas : 0, l.x13, l.xs13, l.y13, l.x2, l.xs2, l.y, l.sync, 4};

  // reference: w13 on the device per route, the rest on the host
  std::vector<std::vector<double>> want(T, std::vector<double>(Phase<1>::N, 0.0));
  uint8_t *cd, *sd; float* yr;
  CK(cudaMalloc(&cd, size_t(Phase<0>::N) * Phase<0>::K)); CK(cudaMalloc(&sd, size_t(Phase<0>::N) * Phase<0>::K / 32));
  CK(cudaMalloc(&yr, sizeof(float) * Phase<0>::N));
  const float lut[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  for (int i = 0; i < n; ++i) {
    CK(cudaMemcpy(cd, h13[i].codes.data(), h13[i].codes.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(sd, h13[i].scales.data(), h13[i].scales.size(), cudaMemcpyHostToDevice));
    for (int j = 0; j < ex[i].ntok; ++j) {
      reference_kernel<<<Phase<0>::N / 128, 128>>>(cd, sd, Phase<0>::N, Phase<0>::K, xd, ex[i].tok[j], yr);
      std::vector<float> gu(Phase<0>::N);
      CK(cudaMemcpy(gu.data(), yr, sizeof(float) * Phase<0>::N, cudaMemcpyDeviceToHost));
      std::vector<float> a(Phase<1>::K);
      for (int k = 0; k < Phase<1>::K; ++k) a[k] = silu_mul(gu[k], gu[Phase<1>::K + k]);
      for (int nn = 0; nn < Phase<1>::N; ++nn) {
        double s = 0;
        const uint8_t* crow = h2[i].codes.data() + size_t(nn) * Phase<1>::K;
        const uint8_t* srow = h2[i].scales.data() + size_t(nn) * (Phase<1>::K / 32);
        for (int k = 0; k < Phase<1>::K; ++k) {
          const int c = crow[k];
          s += double(lut[c & 7] * ((c & 8) ? -1.f : 1.f)) * std::ldexp(1.0, int(srow[k / 32]) - 127) * a[k];
        }
        want[ex[i].tok[j]][nn] += ex[i].wt[j] * s;
      }
    }
  }
  int rc = 0;
  for (int pass = 0; pass < 2; ++pass) {   // twice: the kernel must leave y13 and the sync words clean
    CK(cudaMemset(l.y, 0, size_t(T) * Phase<1>::N * 4));
    launch(p, 0);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    std::vector<float> got(size_t(T) * Phase<1>::N);
    CK(cudaMemcpy(got.data(), l.y, sizeof(float) * got.size(), cudaMemcpyDeviceToHost));
    double max_rel = 0, sum_rel = 0; long cnt = 0; int bad = 0;
    for (int t = 0; t < T; ++t) {
      double scale = 0; for (double v : want[t]) scale = std::max(scale, std::fabs(v));
      if (scale == 0) continue;
      for (int nn = 0; nn < Phase<1>::N; ++nn) {
        double rel = std::fabs(got[size_t(t) * Phase<1>::N + nn] - want[t][nn]) / scale;
        max_rel = std::max(max_rel, rel); sum_rel += rel; ++cnt; bad += rel > 1e-2;
      }
    }
    printf("check layer hot=%d cold=%d cold_ctas=%d pass %d: max rel err %.2e, mean %.2e, %d bad -> %s\n",
           n_hot, n_cold, cold_ctas, pass, max_rel, sum_rel / cnt, bad, bad ? "FAIL" : "ok");
    rc |= bad != 0;
  }
  return rc;
}

static void bench(int n_hot, int n_cold, int cold_ctas, int ntok_mode) {
  const int reps = 20, pool_hot = 40, pool_cold = 20, T = 8;
  std::mt19937 rng(7);
  Host p13 = make_expert(Phase<0>::N, Phase<0>::K, rng, true), p2 = make_expert(Phase<1>::N, Phase<1>::K, rng);
  auto put = [&](const Host& h, bool cold) {
    uint8_t* d;
    if (cold) d = host_pinned(h.packed.size()); else CK(cudaMalloc(&d, h.packed.size()));
    CK(cudaMemcpy(d, h.packed.data(), h.packed.size(), cudaMemcpyHostToDevice));
    return d;
  };
  std::vector<std::pair<uint8_t*, uint8_t*>> hot_pool, cold_pool;
  for (int i = 0; i < pool_hot; ++i) hot_pool.push_back({put(p13, false), put(p2, false)});
  for (int i = 0; i < pool_cold; ++i) cold_pool.push_back({put(p13, true), put(p2, true)});
  std::vector<__nv_bfloat16> xh(size_t(T) * Phase<0>::K);
  std::normal_distribution<float> nd(0.f, 0.05f);
  for (auto& v : xh) v = __float2bfloat16(nd(rng));
  __nv_bfloat16* xd; CK(cudaMalloc(&xd, xh.size() * 2)); CK(cudaMemcpy(xd, xh.data(), xh.size() * 2, cudaMemcpyHostToDevice));
  const int E = n_hot + n_cold;
  Layer l = make_layer(T, 64, E);
  prep_kernel<<<T, 256>>>(xd, Phase<0>::K, l.x13, l.xs13);
  std::uniform_int_distribution<int> tk(0, T - 1);
  std::discrete_distribution<int> tok_dist({0, 74.1, 17.0, 5.4, 2.1, 0.8, 0.4, 0.1, 0.1});
  std::vector<Expert*> lists(reps);
  for (int r = 0; r < reps; ++r) {
    std::vector<Expert> ex(E);
    int routes = 0;
    for (int i = 0; i < E; ++i) {
      auto pr = i < n_hot ? hot_pool[(r * n_hot + i) % pool_hot] : cold_pool[(r * n_cold + i - n_hot) % pool_cold];
      ex[i].w13 = pr.first; ex[i].w2 = pr.second;
      ex[i].ntok = ntok_mode > 0 ? ntok_mode : std::max(1, tok_dist(rng));
      std::vector<int> perm(T);
      for (int j = 0; j < T; ++j) perm[j] = j;
      std::shuffle(perm.begin(), perm.end(), rng);
      for (int j = 0; j < ex[i].ntok; ++j) { ex[i].tok[j] = perm[j]; ex[i].route[j] = routes++ % 64; ex[i].wt[j] = 0.1f; }
    }
    CK(cudaMalloc(&lists[r], sizeof(Expert) * E));
    CK(cudaMemcpy(lists[r], ex.data(), sizeof(Expert) * E, cudaMemcpyHostToDevice));
  }
  cudaStream_t st; CK(cudaStreamCreate(&st));
  auto run = [&](int r) {
    CK(cudaMemsetAsync(l.y, 0, size_t(T) * Phase<1>::N * 4, st));
    Params p{lists[r], n_hot, lists[r] + n_hot, n_cold, n_cold ? cold_ctas : 0, l.x13, l.xs13, l.y13, l.x2, l.xs2,
             l.y, l.sync, 4};
    launch(p, st);
  };
  if (PROF) {
    unsigned long long z[8] = {0, ~0ull, 0, 0, 0, 0, 0, 0}, h[8], h2[8];
    CK(cudaMemcpyToSymbol(g_lprof, z, sizeof(z)));
    unsigned long long z2[8] = {0, ~0ull, 0, 0, 0, 0, 0, 0};
    CK(cudaMemcpyToSymbol(g_lprof2, z2, sizeof(z2)));
    run(0); CK(cudaStreamSynchronize(st));
    CK(cudaMemcpyFromSymbol(h, g_lprof, sizeof(h)));
    CK(cudaMemcpyFromSymbol(h2, g_lprof2, sizeof(h2)));
    printf("  prof: w13 phase ends %.1f..%.1f us after start, kernel end %.1f us, producer ready-wait %.1f us per CTA\n",
           (h2[1] - h[1]) / 1e3, (h2[0] - h[1]) / 1e3, (h2[3] - h[1]) / 1e3, h2[2] / 1e3 / h[6]);
    printf("  prof: per CTA w13 phase %.1f us, whole %.1f us | act_prep x%llu, mean %.1f us, max %.1f us,"
           " last done at %.1f us after start\n", h[0] / 1e3 / h[6], h[2] / 1e3 / h[6], h[4], h[5] / 1e3 / std::max(h[4], 1ull),
           h[3] / 1e3, (h[7] - h[1]) / 1e3);
  }
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
  const double us = best * 1000.0 / reps;
  const double hb = n_hot * double(p13.packed.size() + p2.packed.size()), cb = n_cold * double(p13.packed.size() + p2.packed.size());
  const double sol = std::max(hb / 3.626e12, cb / 0.409e12) * 1e6;
  printf("layer hot=%2d cold=%d cold_ctas=%2d tok=%s: %7.1f us per layer incl. memset (SOL %6.1f us, %3.0f%%)\n",
         n_hot, n_cold, cold_ctas, ntok_mode ? std::to_string(ntok_mode).c_str() : "dist", us, sol, sol / us * 100);
}

int main(int argc, char** argv) {
  if (argc < 2) { printf("usage: tiered_moe_layer check | bench hot cold cold_ctas [ntok]\n"); return 1; }
  if (!strcmp(argv[1], "check")) {
    int rc = 0;
    rc |= check(3, 2, 24);
    rc |= check(9, 2, 24);
    rc |= check(5, 0, 0);
    rc |= check(0, 2, 24);
    return rc;
  }
  bench(atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), argc > 5 ? atoi(argv[5]) : 0);
  return 0;
}

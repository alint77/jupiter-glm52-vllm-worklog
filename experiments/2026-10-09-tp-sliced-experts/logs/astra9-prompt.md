# Astra consult 9: TP-sliced tiered MoE decode kernel (GH200, sm_90a) — where to cut next

Do NOT run commands (your sandbox cannot run on this host). Everything you need is inline.

## Context
One persistent kernel (132 CTAs x 9 warps: 8 consumer + 1 producer, 1 CTA/SM, 4-stage TMA ring of 46 KB) runs
GLM-5.3's W4A16 MoE decode at M = 8..32 tokens for this GPU's 512-wide TP slice of every expert (hidden 6144,
top-8, symmetric INT4 group-32 bf16 scales, fp16 activations with per-row pow2 scale, fp32 accumulate) plus the
bf16 shared expert's slice. Work = units of 128 rows x 512 K (32 KB INT4 + 4 KB scales + ntok activation rows).
Queues S0 (shared w13), R0 (routed w13: entry x 128-row tile, 12 K chunks accumulated in registers, one claim per
12-unit group), S1, R1 (routed w2: one unit per claim, flushed per unit with red.global.add.v4.f32). An entry's w2
units need its w13 complete: the CTA whose done13 count completes the entry runs silu*up (activate_route) and
st.release's ready[q][ei]; w2 producers spin on ready before copying x2 rows.
Hard constraints: same math (no numeric changes; fp32 summation-order changes are OK), only weight TMAs may be
issued before griddepcontrol.wait, no extra quantization.
Floor at M=8 38 hot / 4 cold entries: 61.3 us (HBM ~3.6 TB/s). v32 benches 85-93 us there; M=32 110/12: 254 us vs 167.

## What we just established (fixed trace; numbers are warp 0, mean over 116 hot CTAs, us per CTA, M=8 38/4)
| warp 0, us / CTA | full | compute-only | loads-only |
|---|--:|--:|--:|
| kernel (traced, extra stamps) | 90.0 | 69.8 | 78.4 |
| head (route_prep + PDL) + first fill | 7.1 | 4.6 | 8.4 |
| R0 math (31.4 units) | 28.6 (0.93 / unit) | 23.0 (0.74) | 0.5 |
| R1 math (14.7 units) | 12.7 (0.93) | 10.1 (0.70) | 0.1 |
| loop head: stage full -> math start | 13.5 (0.26 R0 / 0.32 R1) | 11.4 | 7.8 |
| R1 flush / R0 flush | 3.6 / 0.7 | 2.9 / 0.7 | ~0 |
| w13 done count (fence + atomic), 2.6 / CTA | 3.3 (1.3-1.6 each) | 1.6 (0.6) | 7.6 (3.1) |
| activation + ready release (0.3 / CTA) | 1.3 (3.6 each) | 0.9 | 2.2 (5.7) |
| shared expert units | ~4.5 | ~3.3 | ~4.9 |
| waits on loads (all kinds) | ~5 | ~2.6 | ~35 |
| end (last unit + imbalance) | ~4 | ~3.3 | ~1.5 |

- **The full kernel is consumer-bound**: warp 0 consumes 90-96% of the time
  from 16 to 64 us; waits on loads total ~5 us. Loads-only, the same
  delivery finishes at 78 us, so memory is not the limit.
- Per R0 unit the warp-0 cycle is 1.38 us (full) / 1.15 (compute-only);
  the floor needs ~1.2 us per unit for everything (61 us / ~50 units).
- The math slows 25% with loads (0.74 -> 0.93 us / unit); the done-count
  fence + atomic doubles (0.6 -> 1.3-1.6 us) and is on warp 0, which the ring
  waits for (warps 1-7 wait ~22 us vs warp 0 ~15 in the CTA trace).
- ncu (v32 full, base clock): the math regions are issue-bound with
  dependency stalls (selected 33%, wait 25%, not_selected 13%, math 11%,
  dispatch 8-10%, short_sb 7%); per HMMA 13.2 instructions: 4.75 LOP3, 4
  HFMA2, 2 FFMA, 1.1 SHF, 0.84 IMAD, 0.5 LDS. The loop head (27 SASS) is 17%
  of samples, 62% long_sb on the full-barrier try_wait (warps 1-7 idling
  behind warp 0). DRAM bytes 221.6 MB = the model's hot + shared bytes.

Earlier probes, all null or negative: a 5th ring stage (x-row reservation shrunk to fit) = no change; L2 tensor
prefetch of units +2/+4/+8 ahead in a group = +6/+10/+19 us; a 10th "epilogue" warp to take the handoff off the
consumers = much slower (scheduler starvation of a latency-bound warp); rotating the handoff duty over the 8 warps
= no change; R0 groups split into 2-4 smaller groups = slower (more w13 flushes); w2 claims of 2-4 units = noise at
M=8; 16 consumer warps (two groups on alternating stages, v14) = slower (spills).
Ablations (bench, us, M=8 38/4): full ~86-89; loads-only 78; compute-only (no TMA) 60.0; compute-only without
decode 51.5; compute-only without routed MMA 32.6; skeleton 28.
probe_consume.cu (v15-era): the routed consume block alone, 132 CTAs, no loads: 0.72 us per unit at 8 warps.

## Questions
1. Given the breakdown, rank the most promising changes to get warp 0's per-unit cycle from ~1.38 us to <= 1.2 us
   (full) without changing math. Be concrete (code-level).
2. The routed math: 13.2 SASS per HMMA.16816 (decode 1 SHF + 4 LOP3 + 4 HFMA2 per 32-bit word = 8 nibbles; 2 FFMA
   group-scale per HMMA; 0.84 IMAD (movs); 0.75 extra LOP3). Issue-bound with dependency stalls at 2 consumer warps
   per scheduler. Any exact (bit-identical decoded values) decode with fewer instructions, or a restructure
   (e.g. ldmatrix/register reuse, different warp split, interleaving two units, moving the group-scale FFMAs) that
   cuts instructions or stalls? Why might the math slow 25% (0.74 -> 0.93 us/unit) when TMA traffic is on?
3. The loop head costs 0.26-0.32 us per unit between "stage full" and "math start" (desc LDS -> kind -> fs/rows4
   LDS); how to take it off the critical path?
4. The w13 completion handoff (bar.sync 2 + thread-0 __threadfence + atomicAdd on done13, and for the completing CTA
   activate_route + fence.proxy.async + __threadfence + st.release) costs warp 0 1.3-1.6 us per event under load;
   the ring waits for warp 0. How to make it asynchronous or cheaper while keeping the memory-model guarantees?
5. Anything structural we are missing (e.g. the consumer is ~at the floor by itself: 41 us math + 13.5 loop head +
   4.3 flush + 4.6 handoff + 4.5 shared = 68 us of warp-0 work vs a 61 us floor)?

## Code (td_v32.cu excerpts; v35 = v32 + trace/probe switches)
```cuda
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
// ...
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
// ... layer_kernel
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
  const int len[2] = {queue_len(n_tier[0], sh), queue_len(n_tier[1], false)};
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
    const float xs13r =
        lane < T ? ws->xs13[lane] : 0.f;  // token lane's x13 scale
    int q = own, last_ready = -1, it = 0;
    bool stolen = false;
    int gi = 0;
    if (lane == 0) gi = atomicAdd(&ws->next[q], 1);
    gi = __shfl_sync(0xffffffffu, gi, 0);
    while (true) {
      if (gi >= len[q]) {
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
        if (lane == 0) gi = atomicAdd(&ws->next[q], 1);
        gi = __shfl_sync(0xffffffffu, gi, 0);
        continue;
      }
      // claim the next group now: its round trip overlaps this group's issue
      int gn = 0;
#ifndef TD_NO_PREFETCH_CLAIM
      if (lane == 0) gn = atomicAdd(&ws->next[q], 1);
#endif
      const Group gr = group_at(gi, n_tier[q], q == 0 && sh);
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
        TD_V(const uint64_t td_e0 = td_now();)
        if (lane == 0 && it >= STAGES)
          mbar_wait(&empty[s], ((it / STAGES) - 1) & 1);
        TD_V(td_empty += td_now() - td_e0;)
#ifdef TD_UNIT_TRACE
        if (lane == 0) s_issue[s] = td_now();
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
#ifdef TD_CTA_TRACE
            td_record(90 + q, ei, t | own << 16 | gi << 20, ntok, it);
#endif
            desc[s] = make_int4(hdr, ei, t, ntok);
            mbar_expect_tx(&full[s], W_BYTES + S_BYTES + ntok * X2_COPY);
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
            fence_proxy_async();
            last_ready = ei;
          }
          __syncwarp();
#ifdef TD_SPIN_EVERY_UNIT
          while (ld_acquire(&ws->ready[q][ei]) != epoch) __nanosleep(32);
          fence_proxy_async();
#endif
          if (lane < ntok)
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
      gi = __shfl_sync(0xffffffffu, gn, 0);
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
      W_BYTES + S_BYTES + g * XROW_STRIDE + (warp * 16 + tq) * 16;
  const int h1 = warp / 4, k1 = warp % 4;
  const uint32_t wo1 = (k1 * 8) * 1024 + h1 * 512 + lane * 16;
  const uint32_t so1 = W_BYTES + (k1 * 4) * 256 + h1 * 128 + 16 * g;
  const uint32_t xo1 =
      W_BYTES + S_BYTES + g * XROW_STRIDE + (k1 * 16 + tq) * 16;
  int s = 0;
  uint32_t ph = 0;
  for (int it = 0;; ++it) {
    TD_V(const uint64_t td_w0 = td_now();)
    mbar_wait_a(full_u + 8 * s, ph);
    TD_V(td_wait += td_now() - td_w0;)
    const uint4 du = lds_v4(desc_u + 16 * s);
    const int4 d = make_int4(du.x, du.y, du.z, du.w);
    const int kind = d.x & 15;
    if (kind == K_END) break;
#ifdef TD_UNIT_TRACE
    if (warp == 0 && lane == 0)
      td_record(120 + kind, s_issue[s], td_w0, d.w, (d.x >> 4) & 1);
#endif
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
          int done = 0;
          if (lane == 0) {
            __threadfence();
            done = atomicAdd(&ws->done13[q][ei], nch) + nch == UNITS0;
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
            if (lane == 0) st_release(&ws->ready[q][ei], epoch);
          }
        }
        TD_V(td_c0 = td_now();)
      }
    } else if (kind == K_R1) {
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
      __syncwarp();
      if (lane == 0) mbar_arrive_a(empty_s);
      const int t = d.z;
      flush_rows(acc, fs, scr_u, ws->y + t * R1 + h1 * 64, rows4, HIDDEN, ntok,
                 g, tq, lane);
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
```

You reviewed this kernel twice already (td_v56, the TP-sliced tiered MoE decode kernel extended to 64 tokens; the race that turns a call's output into 1e36/NaN garbage). New results, and a candidate fix. Please review the fix critically.

## New measurements (stress: fresh inputs each call, case 64 tokens, cold-heavy routing, pad mask, shared expert on)
- In-kernel check (TD_DEBUGCHK) at the START of consumption of every stage, shared AND routed (R0 int4 weights, scales, x13 rows; R1 weights, scales, x2 rows incl. scale), comparing smem with global sources: zero mismatches across all failing calls (13 garbage calls in 15k).
- Adding the same check again AFTER the MMAs (before empty release): 0 garbage in 2 x 5000 (debug perturbs timing a lot; inconclusive).
- Same node, same case, TD_ACTK=1:
  - control (no change): 25 garbage / 15000
  - TD_SFENCE (both halves below): 0 / 25000 ; also 0/25000 with ACTK default, 0/20000 same_cold T=64, 0/20000 sparse/nohot T=47
  - TD_SFENCE_CONS only: 0 / 20000
  - TD_SFENCE_PROD only: 0 / 20000
  - TD_NO_STEAL (your suggestion, no fence): 0 / 20000 (earlier 0/15000 twice)
  - All remaining failures are the known marginal kind (rel err ~0.005, one element), fp32 atomic order.

## The fix
`fence.proxy.async.shared::cta` (asm volatile, "memory" clobber):
- CONS: in every consumer thread after its last generic read of the stage (ldmatrix / ld.shared of weights, scales, activation rows, flush_meta's reads of the x2 row scale) and before the __syncwarp + lane-0 mbarrier arrive on empty[s]. Three sites: R0 chunk loop, R1, shared.
- PROD: in every producer lane, after lane 0's mbar_wait(empty[s]) + __syncwarp, before writing desc/sd_row/sd_f and issuing tma_3d/tma_2d/cp.async.bulk into stage s.

## Questions
1. Is this the right fix per the PTX memory model (generic-proxy reads of smem, then async-proxy writes (TMA/bulk copy) to the same smem after an mbarrier release(arrive)/acquire(try_wait) chain)? CUTLASS's PipelineTmaAsync consumer_release does not fence; is the WAR edge actually required, and is a fence at either end sufficient (we see both halves alone suffice)? Which placement would you ship: CONS (8 warps x 3 sites, executed by all 256 consumer threads per stage) or PROD (one warp per stage)? I am measuring the cost of each now.
2. If it's not a genuine memory-model requirement, what else could explain both a fence at either end AND no-steal removing it? (I.e. is the fence just a timing perturbation hiding something else?) Give concrete alternative mechanisms and a distinguishing experiment.
3. Why would it appear only at T >= 41..47 (T=40 clean in 20k; 64-token build at T<=32 clean; the shipped 32-token build clean)? Note the layout: stage = W_BYTES 32768 | S_BYTES 4096 | 8 x XROW_STRIDE 1088. Shared units place T token rows of 128 B at W_BYTES + t*128, so at T>32 shared activation rows overlap the routed scale region and at T>40.5 the routed XROW row 1+ region. Also: does the shipped 32-token kernel (same protocol) need the fence too (latent, rarer)?
4. Any other unfenced generic<->async edges in the kernel you can see (smem or global)?

Relevant source excerpts follow.

```cuda
  #define TD_MAX_TOKENS 32  // tokens per call: 32 or 64
#endif
#ifndef TD_COLD_CTAS
  #define TD_COLD_CTAS 16
#endif
constexpr int HIDDEN = 6144, INTER = 512, TOPK = 8;
constexpr int MAX_TOK = 8, MAX_TOKENS = TD_MAX_TOKENS, MAX_ROUTES = MAX_TOKENS * TOPK,
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
constexpr int XROWS = MAX_TOK;
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
static_assert(MAX_TOKENS % 32 == 0, "token tiles; scheduler lanes per token");
constexpr int NT_MAX = MAX_TOKENS / 8;  // shared expert mma token tiles
constexpr int STAGES = TD_STAGES;
constexpr int CONSUMER_WARPS = 8;
constexpr int THREADS = (CONSUMER_WARPS + 3) * 32;  // + producer, scheduler,
                                                    // finisher
constexpr int SMEM_HEAD = 128;
static_assert(STAGES <= 8, "barrier head holds 8 stages");
constexpr int SMEM_BYTES = SMEM_HEAD + 1024 + STAGES * STAGE_BYTES;
static_assert(SMEM_BYTES <= 227 * 1024, "ring exceeds shared memory");
constexpr int PLACEMENT_EXP = 14;  // decoded weights are value * 2^-14
...
  asm volatile("st.release.gpu.global.b32 [%0], %1;" ::"l"(p), "r"(v)
               : "memory");
}
#ifdef TD_SFENCE
  #define TD_SFENCE_CONS
  #define TD_SFENCE_PROD
#endif
#define TD_SFENCE_ASM() \
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory")
#ifdef TD_SFENCE_CONS
  #define TD_SFENCE_C() TD_SFENCE_ASM()
#else
  #define TD_SFENCE_C() \
    do {                \
    } while (0)
#endif
#ifdef TD_SFENCE_PROD
  #define TD_SFENCE_P() TD_SFENCE_ASM()
#else
  #define TD_SFENCE_P() \
    do {                \
    } while (0)
#endif
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
...
      __syncwarp();
      const SchedEntry& e = sq[k];
      if (e.kind == K_END) break;
      const Group gr = {e.kind, e.x, e.c0, e.nch};
#ifdef TD_NOPRE
      const int q = e.q, ei = e.ei, t0 = e.t0, pre_ready = 0;
#else
      const int q = e.q, ei = e.ei, t0 = e.t0, pre_ready = e.ready;
#endif
      int ntok = 0, local = 0, tokl = 0, routel = 0;
      float fl = 0.f;
      if (gr.kind == K_R0 || gr.kind == K_R1) {
        const int l7 = lane & (MAX_TOK - 1);
        ntok = e.ntok;
        local = e.local;
        tokl = e.tok[l7];
        routel = e.route[l7];
        fl = e.f[l7];
      }
      const Tier& tr = p.tier[q];
      for (int ci = 0; ci < gr.nch; ++ci, ++it) {
        const int s = it % STAGES, c = gr.c0 + ci;
        unsigned char* dst = ring + static_cast<size_t>(s) * STAGE_BYTES;
        if (lane == 0 && it >= STAGES)
          mbar_wait(&empty[s], ((it / STAGES) - 1) & 1);
        __syncwarp();
        TD_SFENCE_P();
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
            mbar_expect_tx(&full[s],
                           W_BYTES + S_BYTES + min(ntok, XROWS) * XROW_BYTES0);
            tma_3d(dst, &tr.w[0], t0 * 2 * R0, c * KT0, local, &full[s]);
            tma_3d(dst + W_BYTES, &tr.s[0], t0 * R0, c * G0, local, &full[s]);
          }
          __syncwarp();
          if (lane < ntok && lane < XROWS)
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
            mbar_expect_tx(&full[s],
                           W_BYTES + S_BYTES + min(ntok, XROWS) * X2_COPY);
            tma_3d(dst, &tr.w[1], t * 256, 0, local, &full[s]);
            tma_3d(dst + W_BYTES, &tr.s[1], t * 128, 0, local, &full[s]);
          }
          // weights are in flight; only the activation rows wait for the
          // entry's ready (every copying lane acquires for itself)
          if ((ei | q << 16) != last_ready) {
            if (!pre_ready) {
              while (ld_acquire(&ws->ready[q][ei]) != epoch) __nanosleep(32);
            }
            fence_proxy_async();
            last_ready = ei | q << 16;
          }
          __syncwarp();
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
            while (ld_acquire(&ws->ready_s) != epoch) __nanosleep(32);
            fence_proxy_async();
          }
          if (gr.kind == K_S1) last_ready = -2;
          __syncwarp();
          for (int t = lane; t < T; t += 32)
            bulk_g2s(dst + W_BYTES + t * XS_BYTES,
                     (gr.kind == K_S0
                          ? ws->x13b + static_cast<size_t>(t) * HIDDEN
                          : ws->x2s + static_cast<size_t>(t) * INTER) +
                         c * CKS,
                     XS_BYTES, &full[s]);
        }
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(&sq_empty[k]);
      if (++k == TD_SQ) {
        k = 0;
        kph ^= 1;
      }
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
...
                 sdr0_u = smem_u32(&sd_row[0][0]);
  static_assert(CONSUMER_WARPS == 8,
                "8 K slices (w13), 2 halves x 4 K slices (w2)");
  const int h1 = warp / 4, k1 = warp % 4;
  const uint32_t wo1 = (k1 * 8) * 1024 + h1 * 512 + lane * 16;
  const uint32_t so1 = W_BYTES + (k1 * 4) * 256 + h1 * 128 + 16 * g;
  const uint32_t xo1 =
      W_BYTES + S_BYTES + (g % XROWS) * XROW_STRIDE + (k1 * 16 + tq) * 16;
  int s = 0;
  uint32_t ph = 0;
  for (int it = 0;; ++it) {
    mbar_wait_a(full_u + 8 * s, ph);
    const uint4 du = lds_v4(desc_u + 16 * s);
    const int4 d = make_int4(du.x, du.y, du.z, du.w);
    const int kind = d.x & 15;
    if (kind == K_END) {
      hand_off(make_int4(0, 0, 0, 1));
      break;
    }
    const int q = (d.x >> 4) & 1, last = (d.x >> 9) & 1, nch = d.x >> 16;
    const unsigned char* st = ring + static_cast<size_t>(s) * STAGE_BYTES;
    const uint32_t st_u = ring_u + s * STAGE_BYTES;
    const uint32_t empty_s = empty_u + 8 * s;
    // this lane's two tokens of a routed unit: destination rows and scales
    const int ntok = d.w;
    // flush metadata of stage s (plain shared loads the compiler may schedule
    // into the math), read before the stage is released
    const auto flush_meta = [&](int s_, float* fs_, int* rows_, bool r1) {
      fs_[0] = sd_f[s_][2 * tq];
      fs_[1] = sd_f[s_][2 * tq + 1];
      if (r1) {
        const unsigned char* xr =
            ring + static_cast<size_t>(s_) * STAGE_BYTES + W_BYTES + S_BYTES +
            ((2 * tq) % XROWS) * XROW_STRIDE + XROW_BYTES1;
        fs_[0] *= *reinterpret_cast<const float*>(xr);
        fs_[1] *=
            *reinterpret_cast<const float*>(xr + (XROWS > 1 ? XROW_STRIDE : 0));
      }
#pragma unroll
      for (int j = 0; j < MAX_TOK / 2; ++j)
        rows_[j] = sd_row[s_][(lane >> 4) + 2 * j];
    };
    float fs[2];
    int rows4[MAX_TOK / 2];

    if (kind == K_R0) {
      // the group's nch chunks in a row: its later stages carry nothing new
      // for the consumer (same entry, tile, tokens), so no descriptor reads
      for (int c = 0;; ++c) {
        const uint32_t su = ring_u + s * STAGE_BYTES;
#ifdef TD_DEBUGCHK
        debug_check_routed(p, ws, ring + static_cast<size_t>(s) * STAGE_BYTES,
                           K_R0, q, d.y, d.z, c, ntok, s, ph, it, warp, lane);
#endif
        consume_routed<1>(su + wo1, su + so1, su + xo1, acc);
#ifdef TD_DEBUGCHK
        debug_check_routed(p, ws, ring + static_cast<size_t>(s) * STAGE_BYTES,
                           K_R0, q, d.y, d.z, c, ntok, s, ph, it, warp, lane, 1);
#endif
        if (c == nch - 1) flush_meta(s, fs, rows4, false);
        TD_SFENCE_C();
        __syncwarp();
        if (lane == 0) mbar_arrive_a(empty_u + 8 * s);
        if (c == nch - 1) break;
        if (++s == STAGES) {
          s = 0;
          ph ^= 1;
        }
        mbar_wait_a(full_u + 8 * s, ph);
      }
      {
        const int ei = d.y, t = d.z;
        flush_rows(acc, fs, scr_u, ws->y13 + t * R0 + h1 * 64, rows4, 2 * INTER,
                   ntok, g, tq, lane);
        hand_off(make_int4(q, ei, nch, 0));
      }
    } else if (kind == K_R1) {
#ifdef TD_DEBUGCHK
      debug_check_routed(p, ws, st, K_R1, q, d.y, d.z, 0, ntok, s, ph, it,
                         warp, lane);
#endif
      consume_routed<1>(st_u + wo1, st_u + so1, st_u + xo1, acc);
#ifdef TD_DEBUGCHK
      debug_check_routed(p, ws, st, K_R1, q, d.y, d.z, 0, ntok, s, ph, it,
                         warp, lane, 1);
#endif
      flush_meta(s, fs, rows4, true);
      TD_SFENCE_C();
      __syncwarp();
      if (lane == 0) mbar_arrive_a(empty_s);
      const int t = d.z;
      flush_rows(acc, fs, scr_u, ws->y + t * R1 + h1 * 64, rows4, HIDDEN, ntok,
                 g, tq, lane);
    } else {  // shared expert: 32 rows per warp, 4 k16 steps, nt_n token tiles
      const unsigned char* xa = st + W_BYTES;
#ifdef TD_DEBUGCHK
      const auto dbg_shared = [&](int post) {
      {  // this warp's 32 weight rows and every token row vs their sources
        const int tile = d.y, cch = d.z;
        const bool s0 = kind == K_S0;
        const __nv_bfloat16* src = p.swp[s0 ? 0 : 1];
...
      dbg_shared(0);
#endif
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
        for (int nt = 0; nt < NT_MAX; ++nt) {
          if (nt < nt_n) {
            const uint2 bv = *reinterpret_cast<const uint2*>(
                xa + (nt * 8 + g) * XS_BYTES + ((ks >> 1) * 4 + tq) * 16 +
                (ks & 1) * 8);
            mma_bf16(accs[0][nt], af[0], bv.x, bv.y);
            mma_bf16(accs[1][nt], af[1], bv.x, bv.y);
          }
        }
      }
#ifdef TD_DEBUGCHK
      dbg_shared(1);
#endif
      TD_SFENCE_C();
      __syncwarp();
      if (lane == 0) mbar_arrive_a(empty_s);
      if (last) {
        const int t = d.y;
#pragma unroll
        for (int rb = 0; rb < 2; ++rb)
#pragma unroll
          for (int nt = 0; nt < NT_MAX; ++nt)
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
          __threadfence();  // every thread's own y13s atomics, before the count
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
}

// ---------------------------------------------------------------- 5: finalize
// One float4 per thread: grid [T][HIDDEN / 1024] x 256.
__global__ void finalize_kernel(Workspace* ws,
                                __nv_bfloat16* __restrict__ out) {
  pdl_wait();
  const size_t i = (static_cast<size_t>(blockIdx.y) * HIDDEN +
                    blockIdx.x * 1024 + threadIdx.x * 4);
  float4* y = reinterpret_cast<float4*>(ws->y + i);
  const float4 v = *y;
  *y = make_float4(0.f, 0.f, 0.f, 0.f);
  __nv_bfloat162* o = reinterpret_cast<__nv_bfloat162*>(out + i);
  o[0] = __floats2bfloat162_rn(v.x, v.y);
  o[1] = __floats2bfloat162_rn(v.z, v.w);
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
```

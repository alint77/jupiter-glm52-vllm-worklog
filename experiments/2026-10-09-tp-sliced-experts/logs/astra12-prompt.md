# Astra review 12: the finisher warp's done path (v48-v51 of the sliced MoE layer_kernel)

Do NOT run any commands (your sandbox cannot run here). Review for correctness first (PTX memory
model, proxies), then performance.

Context: same kernel as reviews 10/11. v48 applied your review-11 fixes verbatim (every consumer
warp's lane 0 waits hq_empty before reuse, __syncwarp, lane-0 arrival, hq_full count 8; producer
checks e.kind == K_END before reading other slot fields). Bench-neutral vs v47 (M=8 38/4 87.3 us).

The finisher (one warp, warp 10, sharing an SMSP with 2 MMA-heavy consumer warps) turns each entry's
w13 completion into x2 rows + a ready flag that remote CTAs spin on before their w2 units (the
producer's TMA of x2 rows, after ld.acquire.gpu(ready) + fence.proxy.async in each copying lane, or
via the scheduler probe + smem FIFO you approved in review 11). At M=8 the largest per-CTA bucket in
the kernel is now consumer wait for R1 stages (10-15 us of ~86 us per CTA), i.e. ready latency.

v50 sub-stamps per finished entry (p50, us; M=8 38/4, 1 route unless noted; globaltimer):
  count RT (lane0: fence.sc.gpu + atomicAdd returns)        1.8
  post-count fence (fence.sc.gpu) + __syncwarp              1.5
  y13 loads landed (__ldcg, 32 per lane, all in flight)      1.9   (n=8 routes, K=4 interleave: 7.7)
  silu*up, max-reduce, row_scale, x2 f16 stores + y13 zero   1.0   (n=8: 18.4 !)
  fence.proxy.async + fence.sc.gpu + __syncwarp              1.1
  st.release.gpu (MEMBAR + ST) on lane 0                     0.7
  total ~8-10 us at M=8 (mean 1.5 routes), 11-13 at M=32 (mean 2 routes, up to 30 at 8 routes).
v49 (K=4 route interleave so all routes' loads are in flight; record loaded before the count) was
bench-neutral and did not move the total.

v51 (code below; results pending):
  - TD_ZLATE (default): y13 rows are zeroed AFTER st.release(ready), with float4 stores (they were 2
    of every 3 stores before the release). The next call's red.add's into y13 come after its
    griddepcontrol.wait (PDL), i.e. after this grid completes.
  - TD_FIN_AR (default): fence.acq_rel.gpu instead of __threadfence (fence.sc.gpu).
  - TD_ONEREL (variant): drop the GPU fence between the activation's stores and the release
    __syncwarp; lane 0's st.release.gpu after __syncwarp must cover the other lanes' x2 stores
    (cumulativity through the warp barrier, like your review-11 argument for the count).
  - TD_ATOM_AR (variant): lane 0's count is one atom.acq_rel.gpu.global.add instead of
    fence + relaxed atomicAdd + fence; then __syncwarp distributes the acquire to the loading lanes.
    (SASS: MEMBAR.ALL.CTA; MEMBAR.ALL.GPU; ATOMG.E.ADD.S32.STRONG.GPU; no acquire-side
    instruction visible after it in this build -- is the acquire half implemented by the
    ATOMG.STRONG + dependent use, or is something missing?)

Questions:
1. Correctness of ZLATE (zeroing y13 after the release: anything in THIS grid that could read y13 of
   that route after the release? -- only the finisher reads y13; consumers only red.add into it, and
   all reds for the entry were counted before activation), ONEREL, ATOM_AR, and the writer-side
   fence.proxy.async (x2 is written with generic st by the finisher and read by cp.async.bulk on
   other SMs after their acquire + their own fence.proxy.async: is the writer-side proxy fence
   needed at all?).
2. Why would ~200 instructions of compute + 16 st.global.b16 + 32 st.global.f32 per lane take
   1-1.5 us on this warp (and 18 us for 8 routes)? Consumer warps saturate LDS/ldmatrix (MIO) and
   issue red.global.add.f32 flushes; is MIO/LSU queue contention (st.global through the same MIO
   pipe) the likely cause, and what is the remedy -- fewer, wider stores (shfl-pack x2 halves into
   16 B per lane; I can do the frag_slot permutation via shfl so each lane stores one uint4)?
3. Structural: the done path is a serial chain of ~6 global round trips/fences on one warp. Options:
   (a) the last-arriving CTA's consumers (8 warps) do the activation instead of the finisher (they
       know they are last only after the count RT...),
   (b) split routes between the finisher and the scheduler warp (idle 40-60% of its time in slot
       waits) with an smem barrier before the release,
   (c) per-route ready flags so a w2 unit whose rows are ready can start (each w2 unit uses all of
       the entry's routes, so probably useless),
   (d) skip the post-count acquire fence by having the loads be ld.acquire / ld.relaxed + one fence,
   (e) anything else that shortens count->release. Which would you do first?

## v51 finisher code (kernels/td_v51.cu)
```cpp
template <int K>
__device__ __forceinline__ void activate_routes(Workspace* ws, int routel,
                                                int n, int lane,
                                                uint64_t* tl = nullptr) {
  for (int r0 = 0; r0 < n; r0 += K) {
    float a[K][INTER / 32], m[K];
    float* yr[K];
#pragma unroll
    for (int j = 0; j < K; ++j) {
      const int r = __shfl_sync(0xffffffffu, routel, (r0 + j) & 31);
      yr[j] = ws->y13 + static_cast<size_t>(r) * 2 * INTER;
      m[j] = 0.f;
      if (r0 + j < n) {
#pragma unroll
        for (int q = 0; q < INTER / 32; ++q) {
          const float g = __ldcg(yr[j] + q * 32 + lane),
                      u = __ldcg(yr[j] + INTER + q * 32 + lane);
          a[j][q] = __fdividef(g, 1.f + __expf(-g)) * u;
          m[j] = fmaxf(m[j], fabsf(a[j][q]));
        }
      }
    }
#ifdef TD_UNIT_TRACE
    if (r0 == 0 && tl)
      *tl = td_now_after(__float_as_int(m[0]), __float_as_int(m[K - 1]), 0, 0.f);
#endif
#pragma unroll
    for (int j = 0; j < K; ++j) {
      if (r0 + j >= n) break;
      for (int o = 16; o; o >>= 1)
        m[j] = fmaxf(m[j], __shfl_xor_sync(0xffffffffu, m[j], o));
      float scale;
      const float inv = row_scale(m[j], &scale);
      const int r = __shfl_sync(0xffffffffu, routel, r0 + j);
      if (lane == 0) ws->xs2[r] = scale;
      __half* __restrict__ out = ws->x2 + static_cast<size_t>(r) * X2_LD;
      if (lane == 0) *reinterpret_cast<float*>(out + INTER) = scale;
#pragma unroll
      for (int q = 0; q < INTER / 32; ++q) {
        out[frag_slot(q * 32 + lane)] = __float2half_rn(a[j][q] * inv);
#ifndef TD_ZLATE
        yr[j][q * 32 + lane] = 0.f;
        yr[j][INTER + q * 32 + lane] = 0.f;
#endif
      }
    }
  }
}

// zero the y13 rows of the n routes in lanes [0, n) of routel for the next
// call
__device__ __forceinline__ void zero_routes(Workspace* ws, int routel, int n,
                                            int lane) {
  for (int j = 0; j < n; ++j) {
    const int r = __shfl_sync(0xffffffffu, routel, j);
    float4* __restrict__ yr =
        reinterpret_cast<float4*>(ws->y13 + static_cast<size_t>(r) * 2 * INTER);
#pragma unroll
    for (int q = 0; q < 2 * INTER / 128; ++q)
      yr[q * 32 + lane] = make_float4(0.f, 0.f, 0.f, 0.f);
  }
}

// The shared expert's silu * up for all T tokens (bf16 fragments of x2s);
// y13s is zeroed for the next call.
__device__ __forceinline__ void activate_shared(Workspace* ws, int T, int warp,
                                                int lane) {
  for (int tok = warp; tok < T; tok += CONSUMER_WARPS) {
    float* __restrict__ yr = ws->y13s + static_cast<size_t>(tok) * 2 * INTER;
    __nv_bfloat16* __restrict__ out =
```

```cpp
  if (warp == CONSUMER_WARPS + 2) {  // finisher warp
#ifdef TD_UNIT_TRACE
    unsigned td_fs = 0;  // lane 0: this CTA's finisher records
    if (lane == 0) td_fs = atomicAdd(&td_trace_n, 512u);
#endif
    int k = 0;
    uint32_t kph = 0;
    while (true) {
      TD_V(const uint64_t td_w = td_now();)
      if (lane == 0) mbar_wait(&hq_full[k], kph);
      __syncwarp();
      const int4 h = hq_info[k];
      __syncwarp();
      if (lane == 0) mbar_arrive(&hq_empty[k]);
      if (++k == TD_HQ) {
        k = 0;
        kph ^= 1;
      }
      if (h.w) break;
      const int q = h.x, ei = h.y, nch = h.z;
      TD_V(const uint64_t td_hc = td_now();)
#if TD_ACTK > 1
      const Expert& e = ws->lists[q][ei];
      const int n = e.ntok, routel = e.route[lane & (MAX_TOK - 1)];
#endif
      int done = 0;
      if (lane == 0) {
#ifdef TD_ATOM_AR
        int old;
        asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;"
                     : "=r"(old)
                     : "l"(&ws->done13[q][ei]), "r"(nch)
                     : "memory");
        done = old + nch == UNITS0;
#else
        TD_FIN_FENCE();
        done = atomicAdd(&ws->done13[q][ei], nch) + nch == UNITS0;
#endif
      }
      done = __shfl_sync(0xffffffffu, done, 0);
      TD_V(const uint64_t td_hd = td_now();)
      if (done) {
#ifndef TD_ATOM_AR
        TD_FIN_FENCE();
#endif
        __syncwarp();  // lane 0's acquire (the count) ordered before every
                       // lane's y13 reads
#if TD_ACTK > 1
#ifdef TD_UNIT_TRACE
        const uint64_t td_a0 = td_now();
        uint64_t td_al = 0;
        activate_routes<TD_ACTK>(ws, routel, n, lane, &td_al);
        const uint64_t td_a1 = td_now();
#else
        activate_routes<TD_ACTK>(ws, routel, n, lane);
#endif
#else
        const Expert& e = ws->lists[q][ei];
        const int n = e.ntok;
        for (int r = 0; r < n; ++r) activate_route(ws, e.route[r], lane);
#endif
        fence_proxy_async();
#ifndef TD_ONEREL
        TD_FIN_FENCE();
#endif
        __syncwarp();
#if defined(TD_UNIT_TRACE) && TD_ACTK > 1
        const uint64_t td_a2 = td_now();
#endif
        if (lane == 0) st_release(&ws->ready[q][ei], epoch);
#if defined(TD_ZLATE) && TD_ACTK > 1
        zero_routes(ws, routel, n, lane);
#endif
#if defined(TD_UNIT_TRACE) && TD_ACTK > 1
        if (lane == 0) {
          td_record_at(td_fs++, 179, td_a0, td_al, td_a1, n, 0);
          td_record_at(td_fs++, 143, td_a1, td_a2, td_now(), n, 0);
        }
#endif
      }
#ifdef TD_UNIT_TRACE
      if (lane == 0) {
        td_record_at(td_fs++, 178, td_w, td_hc, 0, 0, 0);
        td_record_at(td_fs++, 141 + done, td_hc, td_hd, td_now(), 0, 0);
      }
#endif
    }
    return;
  }
#endif
#ifdef TD_SCHED
  if (warp == CONSUMER_WARPS + 1) {  // scheduler warp
#ifdef TD_UNIT_TRACE
    unsigned td_ss = 0;  // lane 0: this CTA's scheduler records
    if (lane == 0) td_ss = atomicAdd(&td_trace_n, 512u);
#endif
    const float xs13r = lane < T ? ws->xs13[lane] : 0.f;
```

## consumer hand_off (v48+)
```cpp
  float accs[2][4][4] = {};
  int acq = -1;  // (tier, entry) whose ready this warp has acquired
#ifdef TD_FIN
  int hk = 0, hn = 0;  // handoff slot and count (same in every consumer)
  const auto hand_off = [&](int4 h) {
    // every warp waits for the slot's previous use to be taken before it
    // arrives (no arrival may enter a phase whose predecessor nobody waited
    // on); __syncwarp gathers the warp's reds and warp 0's info for lane 0's
    // release
    if (lane == 0) {
      if (hn >= TD_HQ) mbar_wait(&hq_empty[hk], ((hn / TD_HQ) - 1) & 1);
      if (warp == 0) hq_info[hk] = h;
    }
    __syncwarp();
    if (lane == 0) mbar_arrive(&hq_full[hk]);
    if (++hk == TD_HQ) hk = 0;
    ++hn;
  };
#endif
  TD_V(uint64_t td_c0 = 0;)
  TD_V(uint64_t td_wait = 0; const uint64_t td_q0 = td_now();)
```

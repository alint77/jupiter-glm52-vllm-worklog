"""td_v49.cu = td_v48.cu + TD_ACTK=K (default 4): the finisher activates K
routes at once (all their y13 loads in flight together; v47's trace has one
route per L2 round trip, 8-11 us per finished entry) and loads the entry's
record before the done count instead of after it (the lists are read-only in
the kernel). TD_ACTK=1 keeps the per-route loop."""
from pathlib import Path

here = Path(__file__).parent
src = (here / "td_v48.cu").read_text()


def sub(old, new, count=1):
    global src
    n = src.count(old)
    assert n == count, (n, old[:90])
    src = src.replace(old, new)


sub('''#if defined(TD_SCHED) && !defined(TD_NO_SPIPE)''', '''#ifndef TD_ACTK
  #define TD_ACTK 4
#endif
#if defined(TD_SCHED) && !defined(TD_NO_SPIPE)''')
sub('''// The shared expert's silu * up for all T tokens''', '''// activate_route for the n routes in lanes [0, n) of routel, K at a time
template <int K>
__device__ __forceinline__ void activate_routes(Workspace* ws, int routel,
                                                int n, int lane) {
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
        yr[j][q * 32 + lane] = 0.f;
        yr[j][INTER + q * 32 + lane] = 0.f;
      }
    }
  }
}

// The shared expert's silu * up for all T tokens''')
sub('''      TD_V(const uint64_t td_hc = td_now();)
      int done = 0;
      if (lane == 0) {
        TD_FIN_FENCE();''', '''      TD_V(const uint64_t td_hc = td_now();)
#if TD_ACTK > 1
      const Expert& e = ws->lists[q][ei];
      const int n = e.ntok, routel = e.route[lane & (MAX_TOK - 1)];
#endif
      int done = 0;
      if (lane == 0) {
        TD_FIN_FENCE();''')
sub('''        const Expert& e = ws->lists[q][ei];
        const int n = e.ntok;
        for (int r = 0; r < n; ++r) activate_route(ws, e.route[r], lane);
        fence_proxy_async();
        TD_FIN_FENCE();''', '''#if TD_ACTK > 1
        activate_routes<TD_ACTK>(ws, routel, n, lane);
#else
        const Expert& e = ws->lists[q][ei];
        const int n = e.ntok;
        for (int r = 0; r < n; ++r) activate_route(ws, e.route[r], lane);
#endif
        fence_proxy_async();
        TD_FIN_FENCE();''')
(here / "td_v49.cu").write_text(src)
print("ok")

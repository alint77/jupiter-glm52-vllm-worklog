"""td_v51.cu = td_v50.cu with the finisher's done path shortened (v50's
sub-stamps: ~8-10 us per finished entry = count RT 1.8 + post-count fence 1.5
+ loads 1.9 + compute/stores 1.0 (18 at 8 routes) + fences 1.1 + release 0.7):
- TD_ZLATE (default; TD_NO_ZLATE off): y13 is zeroed after the ready release
  instead of before it (2/3 of the activation's stores leave the path).
- TD_FIN_AR on by default (fence.acq_rel.gpu, not MEMBAR.SC).
- TD_ONEREL: no GPU fence before the release __syncwarp; lane 0's
  st.release.gpu is cumulative over the lanes' x2 stores ordered by it.
- TD_ATOM_AR: the done count is one atom.acq_rel.gpu instead of fence +
  relaxed atomicAdd + fence."""
from pathlib import Path

here = Path(__file__).parent
src = (here / "td_v50.cu").read_text()


def sub(old, new, count=1):
    global src
    n = src.count(old)
    assert n == count, (n, old[:90])
    src = src.replace(old, new)


sub('''#ifndef TD_ACTK''', '''#ifndef TD_NO_ZLATE
  #define TD_ZLATE
#endif
#ifndef TD_NO_FIN_AR
  #define TD_FIN_AR
#endif
#ifndef TD_ACTK''')
sub('''      for (int q = 0; q < INTER / 32; ++q) {
        out[frag_slot(q * 32 + lane)] = __float2half_rn(a[j][q] * inv);
        yr[j][q * 32 + lane] = 0.f;
        yr[j][INTER + q * 32 + lane] = 0.f;
      }''', '''      for (int q = 0; q < INTER / 32; ++q) {
        out[frag_slot(q * 32 + lane)] = __float2half_rn(a[j][q] * inv);
#ifndef TD_ZLATE
        yr[j][q * 32 + lane] = 0.f;
        yr[j][INTER + q * 32 + lane] = 0.f;
#endif
      }''')
sub('''// The shared expert's silu * up for all T tokens''', '''// zero the y13 rows of the n routes in lanes [0, n) of routel for the next
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

// The shared expert's silu * up for all T tokens''')
sub('''      int done = 0;
      if (lane == 0) {
        TD_FIN_FENCE();
        done = atomicAdd(&ws->done13[q][ei], nch) + nch == UNITS0;
      }
      done = __shfl_sync(0xffffffffu, done, 0);
      TD_V(const uint64_t td_hd = td_now();)
      if (done) {
        TD_FIN_FENCE();
        __syncwarp();''', '''      int done = 0;
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
        __syncwarp();''')
sub('''        fence_proxy_async();
        TD_FIN_FENCE();
        __syncwarp();
#if defined(TD_UNIT_TRACE) && TD_ACTK > 1''', '''        fence_proxy_async();
#ifndef TD_ONEREL
        TD_FIN_FENCE();
#endif
        __syncwarp();
#if defined(TD_UNIT_TRACE) && TD_ACTK > 1''')
sub('''        if (lane == 0) st_release(&ws->ready[q][ei], epoch);
#if defined(TD_UNIT_TRACE) && TD_ACTK > 1''', '''        if (lane == 0) st_release(&ws->ready[q][ei], epoch);
#if defined(TD_ZLATE) && TD_ACTK > 1
        zero_routes(ws, routel, n, lane);
#endif
#if defined(TD_UNIT_TRACE) && TD_ACTK > 1''')
assert "TD_ZLATE" in src
(here / "td_v51.cu").write_text(src)
print("ok")

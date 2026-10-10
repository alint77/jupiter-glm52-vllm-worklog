"""td_v37.cu = td_v35.cu + switches from Astra consult 9 (logs/codex-review9.md):
TD_MMA_NV (non-volatile mma asm), TD_MB2 (two row blocks interleaved in the
routed math), TD_GLOOP (R0 group consumed in an inner loop; flush metadata
read as plain shared loads only where needed), TD_ACQREL (acq_rel done
count, no __threadfence in the w13 handoff)."""
from pathlib import Path

here = Path(__file__).parent
src = (here / "td_v35.cu").read_text()


def sub(old, new, count=1):
    global src
    n = src.count(old)
    assert n == count, (n, old[:80])
    src = src.replace(old, new)


# --- TD_MMA_NV
sub('''__device__ __forceinline__ void mma_f16(float* d, const uint32_t* a,
                                        uint32_t b0, uint32_t b1,
                                        const float* c) {
  asm volatile(''', '''#ifdef TD_MMA_NV  // pure arithmetic: let the compiler schedule it
  #define TD_MMA_ASM asm
#else
  #define TD_MMA_ASM asm volatile
#endif
__device__ __forceinline__ void mma_f16(float* d, const uint32_t* a,
                                        uint32_t b0, uint32_t b1,
                                        const float* c) {
  TD_MMA_ASM(''')

# --- TD_MB2
old_mb = '''#pragma unroll
    for (int mb = 0; mb < 4; ++mb) {
      const float zero[4] = {0.f, 0.f, 0.f, 0.f};
      float d[4];
      uint32_t a[4];
#ifdef TD_ABL_NODECODE'''
new_mb = '''#ifdef TD_MB2  // two row blocks at a time: independent decode / MMA chains
    const float zero[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int mb = 0; mb < 4; mb += 2) {
      float d0[4], d1[4];
      uint32_t a0[4], a1[4];
      decode_int4_fast(w0s[mb], a0);
      decode_int4_fast(w0s[mb + 1], a1);
      mma_f16(d0, a0, xv.x, xv.y, zero);
      mma_f16(d1, a1, xv.x, xv.y, zero);
      decode_int4_fast(w1s[mb], a0);
      decode_int4_fast(w1s[mb + 1], a1);
      mma_f16(d0, a0, xv.z, xv.w, d0);
      mma_f16(d1, a1, xv.z, xv.w, d1);
      const float s00 = __uint_as_float(sws[mb] << 16),
                  s01 = __uint_as_float(sws[mb] & 0xFFFF0000u),
                  s10 = __uint_as_float(sws[mb + 1] << 16),
                  s11 = __uint_as_float(sws[mb + 1] & 0xFFFF0000u);
      acc[mb][0] = fmaf(s00, d0[0], acc[mb][0]);
      acc[mb][1] = fmaf(s00, d0[1], acc[mb][1]);
      acc[mb][2] = fmaf(s01, d0[2], acc[mb][2]);
      acc[mb][3] = fmaf(s01, d0[3], acc[mb][3]);
      acc[mb + 1][0] = fmaf(s10, d1[0], acc[mb + 1][0]);
      acc[mb + 1][1] = fmaf(s10, d1[1], acc[mb + 1][1]);
      acc[mb + 1][2] = fmaf(s11, d1[2], acc[mb + 1][2]);
      acc[mb + 1][3] = fmaf(s11, d1[3], acc[mb + 1][3]);
    }
#else
#pragma unroll
    for (int mb = 0; mb < 4; ++mb) {
      const float zero[4] = {0.f, 0.f, 0.f, 0.f};
      float d[4];
      uint32_t a[4];
#ifdef TD_ABL_NODECODE'''
sub(old_mb, new_mb)
sub('''      acc[mb][3] = fmaf(s1, d[3], acc[mb][3]);
    }
  }
}''', '''      acc[mb][3] = fmaf(s1, d[3], acc[mb][3]);
    }
#endif
  }
}''')

# --- TD_GLOOP: metadata reads
sub('''    const Expert* experts = ws->lists[q];
    // this lane's two tokens of a routed unit: destination rows and scales
    const int ntok = d.w;
    // read unconditionally (stale past ntok, masked at the flush)
    float fs[2] =''', '''    const Expert* experts = ws->lists[q];
    // this lane's two tokens of a routed unit: destination rows and scales
    const int ntok = d.w;
#ifdef TD_GLOOP
    // flush metadata of stage s (plain shared loads the compiler may schedule
    // into the math), read before the stage is released
    const auto flush_meta = [&](int s_, float* fs_, int* rows_, bool r1) {
      fs_[0] = sd_f[s_][2 * tq];
      fs_[1] = sd_f[s_][2 * tq + 1];
      if (r1) {
        const unsigned char* xr = ring + static_cast<size_t>(s_) * STAGE_BYTES +
                                  W_BYTES + S_BYTES +
                                  ((2 * tq) % XROWS) * XROW_STRIDE + XROW_BYTES1;
        fs_[0] *= *reinterpret_cast<const float*>(xr);
        fs_[1] *= *reinterpret_cast<const float*>(
            xr + (XROWS > 1 ? XROW_STRIDE : 0));
      }
#pragma unroll
      for (int j = 0; j < MAX_TOK / 2; ++j)
        rows_[j] = sd_row[s_][(lane >> 4) + 2 * j];
    };
    float fs[2];
    int rows4[MAX_TOK / 2];
#else
    // read unconditionally (stale past ntok, masked at the flush)
    float fs[2] =''')
sub('''      rows4[j] = static_cast<int>(
          lds_u32(sdr0_u + (s * MAX_TOK + (lane >> 4) + 2 * j) * 4));
''', '''      rows4[j] = static_cast<int>(
          lds_u32(sdr0_u + (s * MAX_TOK + (lane >> 4) + 2 * j) * 4));
#endif
''')

# --- TD_GLOOP: R0 inner loop
sub('''    if (kind == K_R0) {
#ifdef TD_UNIT_TRACE
      const uint64_t td_cs = td_now();
#endif
      consume_routed<1>(st_u + wo1, st_u + so1, st_u + xo1, acc);
      __syncwarp();
      if (lane == 0) mbar_arrive_a(empty_s);
#ifdef TD_UNIT_TRACE
      if (warp == 0 && lane == 0)
        td_record_at(td_slot++, 151, td_cs, td_now(), td_f, 0, 0);
#endif
      if (last) {''', '''    if (kind == K_R0) {
#ifdef TD_GLOOP
      // the group's nch chunks in a row: its later stages carry nothing new
      // for the consumer (same entry, tile, tokens), so no descriptor reads
#ifdef TD_UNIT_TRACE
      uint64_t td_fc = td_f;
#endif
      for (int c = 0;; ++c) {
        const uint32_t su = ring_u + s * STAGE_BYTES;
#ifdef TD_UNIT_TRACE
        const uint64_t td_cs = td_now();
#endif
        consume_routed<1>(su + wo1, su + so1, su + xo1, acc);
        if (c == nch - 1) flush_meta(s, fs, rows4, false);
        __syncwarp();
        if (lane == 0) mbar_arrive_a(empty_u + 8 * s);
#ifdef TD_UNIT_TRACE
        if (warp == 0 && lane == 0)
          td_record_at(td_slot++, 151, td_cs, td_now(), td_fc, 0, 0);
#endif
        if (c == nch - 1) break;
        if (++s == STAGES) {
          s = 0;
          ph ^= 1;
        }
        TD_V(const uint64_t td_w1 = td_now();)
        mbar_wait_a(full_u + 8 * s, ph);
        TD_V(td_wait += td_now() - td_w1;)
#ifdef TD_UNIT_TRACE
        td_fc = td_now();
        if (warp == 0 && lane == 0)
          td_record_at(td_slot++, 120 + K_R0, s_issue[s], td_w1, td_fc, ntok,
                       q);
#endif
      }
      {
#else
#ifdef TD_UNIT_TRACE
      const uint64_t td_cs = td_now();
#endif
      consume_routed<1>(st_u + wo1, st_u + so1, st_u + xo1, acc);
      __syncwarp();
      if (lane == 0) mbar_arrive_a(empty_s);
#ifdef TD_UNIT_TRACE
      if (warp == 0 && lane == 0)
        td_record_at(td_slot++, 151, td_cs, td_now(), td_f, 0, 0);
#endif
      if (last) {
#endif''')

# R1: metadata after the math (before the release)
sub('''      consume_routed<1>(st_u + wo1, st_u + so1, st_u + xo1, acc);
      __syncwarp();
      if (lane == 0) mbar_arrive_a(empty_s);
      const int t = d.z;''', '''      consume_routed<1>(st_u + wo1, st_u + so1, st_u + xo1, acc);
#ifdef TD_GLOOP
      flush_meta(s, fs, rows4, true);
#endif
      __syncwarp();
      if (lane == 0) mbar_arrive_a(empty_s);
      const int t = d.z;''')

# --- TD_ACQREL handoff
sub('''          if (lane == 0) {
            __threadfence();
            done = atomicAdd(&ws->done13[q][ei], nch) + nch == UNITS0;
          }''', '''          if (lane == 0) {
#ifdef TD_ACQREL
            // the CTA barrier carries every warp's y13 reds to lane 0; its
            // acq_rel count publishes them and, for the winner, acquires all
            // other CTAs' (cumulativity), with no full fence
            int old;
            asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;"
                         : "=r"(old)
                         : "l"(&ws->done13[q][ei]), "r"(nch)
                         : "memory");
            done = old + nch == UNITS0;
#else
            __threadfence();
            done = atomicAdd(&ws->done13[q][ei], nch) + nch == UNITS0;
#endif
          }''')
sub('''          if (__shfl_sync(0xffffffffu, done, 0)) {
            __threadfence();
            __syncwarp();''', '''          if (__shfl_sync(0xffffffffu, done, 0)) {
#ifndef TD_ACQREL
            __threadfence();
#endif
            __syncwarp();''')
sub('''            for (int r = 0; r < n; ++r) activate_route(ws, e.route[r], lane);
            fence_proxy_async();
            __threadfence();
            __syncwarp();''', '''            for (int r = 0; r < n; ++r) activate_route(ws, e.route[r], lane);
            fence_proxy_async();
#ifndef TD_ACQREL
            __threadfence();
#endif
            __syncwarp();''')

(here / "td_v37.cu").write_text(src)
print("ok")

"""td_v50.cu = td_v49.cu + finisher sub-stamps in trace builds (no change
otherwise): record 179 {post-count fence done, first K routes' loads landed,
activation stores issued; n routes}, record 143 {stores issued, proxy + GPU
fences done, ready released}."""
from pathlib import Path

here = Path(__file__).parent
src = (here / "td_v49.cu").read_text()


def sub(old, new, count=1):
    global src
    n = src.count(old)
    assert n == count, (n, old[:90])
    src = src.replace(old, new)


sub('''__device__ __forceinline__ void activate_routes(Workspace* ws, int routel,
                                                int n, int lane) {''',
    '''__device__ __forceinline__ void activate_routes(Workspace* ws, int routel,
                                                int n, int lane,
                                                uint64_t* tl = nullptr) {''')
sub('''          m[j] = fmaxf(m[j], fabsf(a[j][q]));
        }
      }
    }
''', '''          m[j] = fmaxf(m[j], fabsf(a[j][q]));
        }
      }
    }
#ifdef TD_UNIT_TRACE
    if (r0 == 0 && tl)
      *tl = td_now_after(__float_as_int(m[0]), __float_as_int(m[K - 1]), 0, 0.f);
#endif
''')
sub('''        TD_FIN_FENCE();
        __syncwarp();  // lane 0's acquire (the count) ordered before every
                       // lane's y13 reads
#if TD_ACTK > 1
        activate_routes<TD_ACTK>(ws, routel, n, lane);
#else''', '''        TD_FIN_FENCE();
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
#else''')
sub('''        fence_proxy_async();
        TD_FIN_FENCE();
        __syncwarp();
        if (lane == 0) st_release(&ws->ready[q][ei], epoch);
      }''', '''        fence_proxy_async();
        TD_FIN_FENCE();
        __syncwarp();
#if defined(TD_UNIT_TRACE) && TD_ACTK > 1
        const uint64_t td_a2 = td_now();
#endif
        if (lane == 0) st_release(&ws->ready[q][ei], epoch);
#if defined(TD_UNIT_TRACE) && TD_ACTK > 1
        if (lane == 0) {
          td_record_at(td_fs++, 179, td_a0, td_al, td_a1, n, 0);
          td_record_at(td_fs++, 143, td_a1, td_a2, td_now(), n, 0);
        }
#endif
      }''')
(here / "td_v50.cu").write_text(src)
print("ok")

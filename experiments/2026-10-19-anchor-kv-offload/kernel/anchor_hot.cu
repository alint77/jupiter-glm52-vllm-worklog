// Anchor hot cache (v3e): an HBM buffer of N rows in front of an anchor
// layer's fp8_ds_mla main cache on Grace. Per decode step and anchor, the
// converted top-k (rank-local slots) becomes buffer rows in one launch, one
// thread per entry:
//   lookup  hits stamp their row and set its CLOCK reference bit (one packed
//           status word, epoch << 1 | ref); misses claim their slot (where[s]:
//           -1 -> -2, one winner per slot) and are listed; this step's tokens
//           map to reserved rows N + j and their slots are invalidated (the
//           KV write changes the slot's content).
//   sweep   the last block to arrive gives every claim a victim in CLOCK hand
//           order (block scan over a window of candidates: a row read this
//           step is skipped, a set reference bit is cleared and passed),
//           publishes where[s], advances the step and releases the others.
//           The launch is cooperative, so every block is resident and the
//           waiting blocks cannot keep a later block from starting.
//   copy    the other blocks: once the last block arrives the claim list is
//           final, so they read the missed rows from Grace (16 B words over
//           their threads) while it sweeps, and store them to their rows
//           after the release; each entry that missed reads its row.
// Entries keep position, multiplicity and -1, so FlashMLA reads the same
// bytes in the same order as from the full cache.
#include <cub/block/block_scan.cuh>
#include <cuda/atomic>
#include <cuda_runtime.h>
#include <stdint.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

namespace {

constexpr int ROW_WORDS = 656 / 16;  // 41 x 16 B
constexpr int THREADS = 1024;
constexpr int SWEEP_V = 4;     // candidates per thread per sweep round
constexpr int SH_MISS = 4096;  // claims the sweep stages in shared memory

// ctl: [0] step (epoch; status holds it << 1, so < 2^30 steps), [1] CLOCK
// hand, [2 + 2 * (step & 1)] claims of the call at that step, [6] overflow
// (claim without a victim: must stay 0), [7] candidates swept, [8] claims so
// far ([7], [8] wrap: telemetry), [9] blocks arrived, [10..15] and [24..26] AHC_PROF timers,
// [16] released step, [20] step whose claim list is final, [21] its sweeping
// block.
constexpr int CTL = 32;
constexpr int REL = 16;
constexpr int ARR = 20;
constexpr int LASTB = 21;

struct State {
  int32_t* where;     // [S] slot -> row, -1 absent, -2 claimed this step
  int32_t* slot_of;   // [N] row -> slot, -1 free
  int32_t* status;    // [N] (step last read or filled) << 1 | CLOCK ref bit
  int32_t* ctl;
  int32_t* miss;      // [MAXE] claimed slots
  int32_t* miss_row;  // [MAXE] their rows
};

// Accesses that race with other threads of the launch.
__device__ __forceinline__ int ld_rlx(const int32_t* p) {
  return cuda::atomic_ref<int32_t, cuda::thread_scope_device>(*const_cast<int32_t*>(p))
      .load(cuda::memory_order_relaxed);
}
__device__ __forceinline__ void st_rlx(int32_t* p, int32_t v) {
  cuda::atomic_ref<int32_t, cuda::thread_scope_device>(*p).store(
      v, cuda::memory_order_relaxed);
}

__device__ __forceinline__ int warp_append(int32_t* counter, bool want) {
  unsigned m = __ballot_sync(0xffffffffu, want);
  if (!m) return -1;
  int lane = threadIdx.x & 31;
  int leader = __ffs(m) - 1;
  int base = 0;
  if (lane == leader) base = atomicAdd(counter, __popc(m));
  base = __shfl_sync(0xffffffffu, base, leader);
  return want ? base + __popc(m & ((1u << lane) - 1)) : -1;
}

__device__ __forceinline__ int64_t gtime() {
  int64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
// read the timer only once dep is available
__device__ __forceinline__ int64_t gtime_after(int dep) {
  int64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t) : "r"(dep));
  return t;
}

// One block gives every claim a victim, in hand order. Candidate p of a round
// is row (hand + p) % N; the hand stops right after the candidate taken by
// the last claim, so exactly the rows it passed lose their reference bit.
// Everything a candidate needs is loaded before the scan; after it, stores.
// The other blocks wait at the barrier, so plain (L2) loads see final values.
__device__ void sweep(State st, int step, int nrows, int nmiss, int hand) {
  using Scan = cub::BlockScan<int, THREADS>;
  __shared__ typename Scan::TempStorage scan;
  __shared__ int sh_stop, sh_taken;
  __shared__ int32_t sh_miss[SH_MISS];
#ifdef AHC_PROF
  int64_t t_scan = 0, t_enter = gtime();
#endif
#ifdef AHC_PROF
  int64_t t_pre = gtime(), t_cand = 0;
#endif
  int assigned = 0;
  const int max_rounds = 2 * (nrows / (THREADS * SWEEP_V) + 1) + 1;
  for (int round = 0; assigned < nmiss && round < max_rounds; ++round) {
    const int need = nmiss - assigned;
    // thread t owns candidates p = t * V + v (contiguous, scan in hand order)
    int c[SWEEP_V], stv[SWEEP_V], old[SWEEP_V];
    bool ok[SWEEP_V], refd[SWEEP_V];
#pragma unroll
    for (int v = 0; v < SWEEP_V; ++v) {
      c[v] = (hand + threadIdx.x * SWEEP_V + v) % nrows;
      stv[v] = __ldcg(&st.status[c[v]]);
      old[v] = __ldcg(&st.slot_of[c[v]]);
    }
    if (round == 0) {
      for (int i = threadIdx.x; i < min(nmiss, SH_MISS); i += THREADS)
        sh_miss[i] = __ldcg(&st.miss[i]);
    }
    int cnt = 0;
#pragma unroll
    for (int v = 0; v < SWEEP_V; ++v) {
      bool fresh = (stv[v] >> 1) != step;
      refd[v] = fresh && (stv[v] & 1);
      ok[v] = fresh && !refd[v];
      cnt += ok[v];
    }
#ifdef AHC_PROF
    if (round == 0) t_cand = gtime_after(cnt + old[0] + old[SWEEP_V - 1]);
#endif
    int off, total;
    Scan(scan).ExclusiveSum(cnt, off, total);
#ifdef AHC_PROF
    if (round == 0 && threadIdx.x == 0) t_scan = gtime();
#endif
    if (threadIdx.x == 0) {
      sh_stop = THREADS * SWEEP_V;
      sh_taken = min(total, need);
    }
    __syncthreads();
    if (total >= need && off < need && off + cnt >= need) {
      int r = off;
#pragma unroll
      for (int v = 0; v < SWEEP_V; ++v) {
        r += ok[v];
        if (ok[v] && r == need) sh_stop = threadIdx.x * SWEEP_V + v + 1;
      }
    }
    __syncthreads();
    const int stop = sh_stop;
    int r = off;
#pragma unroll
    for (int v = 0; v < SWEEP_V; ++v) {
      int p = threadIdx.x * SWEEP_V + v;
      if (p >= stop) break;
      if (refd[v]) st.status[c[v]] = stv[v] & ~1;
      if (ok[v]) {
        int i = assigned + r;
        ++r;
        int s = i < SH_MISS ? sh_miss[i] : __ldcg(&st.miss[i]);
        // where/slot_of stay inverse (every writer clears both), so the
        // victim's slot maps to it
        if (old[v] >= 0) st.where[old[v]] = -1;
        st.slot_of[c[v]] = s;
        st.status[c[v]] = step << 1;
        st.where[s] = c[v];
        st.miss_row[i] = c[v];
      }
    }
    hand = (hand + stop) % nrows;
    assigned += sh_taken;
    if (threadIdx.x == 0) st.ctl[7] = (int32_t)((uint32_t)st.ctl[7] + stop);
#ifdef AHC_PROF
    if (threadIdx.x == 0) st.ctl[14] += 1;
#endif
    __syncthreads();
  }
  for (int i = assigned + threadIdx.x; i < nmiss; i += THREADS) {
    st.where[i < SH_MISS ? sh_miss[i] : __ldcg(&st.miss[i])] = -1;
    st.miss_row[i] = -1;
  }
  if (threadIdx.x == 0) {
    st.ctl[1] = hand;
    st.ctl[6] += nmiss - assigned;
    st.ctl[8] = (int32_t)((uint32_t)st.ctl[8] + nmiss);
#ifdef AHC_PROF
    st.ctl[12] += (int32_t)(t_scan ? gtime() - t_scan : 0);
    st.ctl[24] += (int32_t)(t_pre - t_enter);
    st.ctl[25] += (int32_t)(t_cand - t_pre);
    st.ctl[26] += (int32_t)(t_scan - t_cand);
#endif
  }
}

template <int UNROLL>
__global__ void __launch_bounds__(THREADS)
map_kernel(State st, const int32_t* __restrict__ idx, int n,
           const int64_t* __restrict__ new_slots, int t, int32_t* __restrict__ out,
           const int4* __restrict__ src, int4* __restrict__ dst, int nrows) {
  __shared__ int32_t sh_new[64];
  __shared__ bool last;
  if (threadIdx.x < 64) sh_new[threadIdx.x] = threadIdx.x < t ? (int32_t)new_slots[threadIdx.x] : -1;
  __syncthreads();
  const int step = st.ctl[0] + 1;
  const int hand = st.ctl[1];  // written only by the previous launch's sweep
  int32_t* cnt = st.ctl + 2 + 2 * (step & 1);
#ifdef AHC_PROF
  const int64_t t_start = gtime();
  if (threadIdx.x == 0) atomicMin((unsigned long long*)(st.ctl + 10), (unsigned long long)t_start);
#endif
  if (blockIdx.x == 0 && threadIdx.x < t) {
    // no entry reads where[] for these slots (they map to reserved rows)
    int s = sh_new[threadIdx.x];
    if (s >= 0) {
      int r = ld_rlx(&st.where[s]);
      if (r >= 0) {
        st_rlx(&st.where[s], -1);
        st_rlx(&st.slot_of[r], -1);
        st_rlx(&st.status[r], ld_rlx(&st.status[r]) & ~1);
      }
    }
  }
  const int e = blockIdx.x * THREADS + threadIdx.x;
  const int s = e < n ? idx[e] : -1;
  bool pend = false, claim = false;
  if (s >= 0) {
    int j = -1;
    for (int k = 0; k < t; ++k) j = (sh_new[k] == s) ? k : j;
    int o;
    if (j >= 0) {
      o = nrows + j;
    } else {
      o = ld_rlx(&st.where[s]);
      if (o >= 0) {
        st_rlx(&st.status[o], (step << 1) | 1);
      } else {
        pend = true;
        if (o == -1) claim = atomicCAS(&st.where[s], -1, -2) == -1;
      }
    }
    if (!pend) out[e] = o;
  } else if (e < n) {
    out[e] = -1;
  }
  int m = warp_append(&cnt[0], claim);
  if (claim) st.miss[m] = s;

  // grid barrier: the last block to arrive sweeps, the others wait for it
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) last = atomicAdd(&st.ctl[9], 1) == (int)gridDim.x - 1;
  __syncthreads();
  cuda::atomic_ref<int32_t, cuda::thread_scope_device> arr(st.ctl[ARR]), rel(st.ctl[REL]);
  if (last) {
    __threadfence();
    if (threadIdx.x == 0) {
      st.ctl[LASTB] = blockIdx.x;
      arr.store(step, cuda::memory_order_release);
    }
#ifdef AHC_PROF
    const int64_t t_last = gtime();
#endif
    sweep(st, step, nrows, __ldcg(&cnt[0]), hand);
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
      // the previous call (stream-ordered before this launch) is done
      st.ctl[2 + 2 * ((step + 1) & 1)] = 0;
      st.ctl[9] = 0;
      st.ctl[0] = step;
#ifdef AHC_PROF
      int64_t t0 = (int64_t)*(unsigned long long*)(st.ctl + 10);
      st.ctl[13] += (int32_t)(t_last - t0);
      st.ctl[15] += (int32_t)(gtime() - t_last);
      *(unsigned long long*)(st.ctl + 10) = ~0ull;
#endif
      rel.store(step, cuda::memory_order_release);
    }
  } else if (threadIdx.x == 0) {
    while (arr.load(cuda::memory_order_acquire) != step) __nanosleep(32);
  }
  __syncthreads();

  // claim list final: the copiers (all blocks but the sweeping one, unless it
  // is alone) read their words from Grace now, store them after the release
  const int nmiss = __ldcg(&cnt[0]);
  const int lastb = __ldcg(&st.ctl[LASTB]);
  const bool alone = gridDim.x == 1;
  const bool copier = alone || !last;
  const int ncp = alone ? 1 : gridDim.x - 1;
  const int cb = alone || (int)blockIdx.x < lastb ? blockIdx.x : blockIdx.x - 1;
  const int ce = copier ? cb * THREADS + threadIdx.x : 0;
  const int stride = ncp * THREADS;
  const int total = copier ? nmiss * ROW_WORDS : 0;
  int4 pre[UNROLL];
#pragma unroll
  for (int u = 0; u < UNROLL; ++u) {
    int f = ce + u * stride;
    if (f < total) {
      int i = f / ROW_WORDS;
      pre[u] = src[(int64_t)__ldcg(&st.miss[i]) * ROW_WORDS + (f - i * ROW_WORDS)];
    }
  }
  if (!last && threadIdx.x == 0) {
    while (rel.load(cuda::memory_order_acquire) != step) __nanosleep(32);
  }
  __syncthreads();

  if (pend) out[e] = __ldcg(&st.where[s]);
#pragma unroll
  for (int u = 0; u < UNROLL; ++u) {
    int f = ce + u * stride;
    if (f < total) {
      int i = f / ROW_WORDS;
      int r = __ldcg(&st.miss_row[i]);
      if (r >= 0) dst[(int64_t)r * ROW_WORDS + (f - i * ROW_WORDS)] = pre[u];
    }
  }
  for (int f0 = ce + UNROLL * stride; f0 < total; f0 += stride * UNROLL) {
    int4 v[UNROLL];
    int64_t to[UNROLL];
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      int f = f0 + u * stride;
      to[u] = -1;
      if (f < total) {
        int i = f / ROW_WORDS, w = f - i * ROW_WORDS;
        int r = __ldcg(&st.miss_row[i]);
        if (r >= 0) {
          v[u] = src[(int64_t)__ldcg(&st.miss[i]) * ROW_WORDS + w];
          to[u] = (int64_t)r * ROW_WORDS + w;
        }
      }
    }
#pragma unroll
    for (int u = 0; u < UNROLL; ++u)
      if (to[u] >= 0) dst[to[u]] = v[u];
  }
}

}  // namespace

void anchor_hot_map(torch::Tensor idx, torch::Tensor new_slots, torch::Tensor out,
                    torch::Tensor src, torch::Tensor dst, torch::Tensor where,
                    torch::Tensor slot_of, torch::Tensor status, torch::Tensor ctl,
                    torch::Tensor miss, torch::Tensor miss_row) {
  for (auto* x : {&idx, &out, &where, &slot_of, &status, &ctl, &miss, &miss_row}) {
    TORCH_CHECK(x->is_cuda() && x->is_contiguous() && x->scalar_type() == at::kInt);
  }
  // this step's slot mapping, as the KV write takes it
  TORCH_CHECK(new_slots.is_cuda() && new_slots.is_contiguous() &&
              new_slots.scalar_type() == at::kLong);
  TORCH_CHECK(dst.is_cuda() && dst.is_contiguous() && src.is_contiguous());
  const int64_t n = idx.numel();
  const int t = new_slots.numel();
  const int nrows = slot_of.numel();
  const int64_t nslots = where.numel();
  // A sweep window holds distinct rows. Every claim gets a victim within two
  // laps when hit rows + claims <= nrows, which n <= nrows guarantees.
  TORCH_CHECK(t <= 64 && n <= nrows && nrows >= THREADS * SWEEP_V);
  TORCH_CHECK(out.numel() == n && miss.numel() >= n && miss_row.numel() >= n &&
              status.numel() == nrows && ctl.numel() >= CTL);
  TORCH_CHECK(src.nbytes() >= nslots * ROW_WORDS * 16 &&
              dst.nbytes() >= (int64_t)(nrows + 64) * ROW_WORDS * 16);
  TORCH_CHECK(nslots < (1ll << 31));
  const auto dev = idx.device();
  for (auto* x : {&new_slots, &out, &dst, &where, &slot_of, &status, &ctl, &miss, &miss_row}) {
    TORCH_CHECK(x->device() == dev);
  }
  TORCH_CHECK(reinterpret_cast<uintptr_t>(src.data_ptr()) % 16 == 0 &&
              reinterpret_cast<uintptr_t>(dst.data_ptr()) % 16 == 0);
  State st{where.data_ptr<int32_t>(), slot_of.data_ptr<int32_t>(),
           status.data_ptr<int32_t>(), ctl.data_ptr<int32_t>(),
           miss.data_ptr<int32_t>(), miss_row.data_ptr<int32_t>()};
  auto stream = at::cuda::getCurrentCUDAStream();
  const int blocks = std::max<int>((n + THREADS - 1) / THREADS, 1);
  static int resident = 0;  // blocks the device can hold at once
  if (!resident) {
    int per_sm = 0, sms = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, map_kernel<4>,
                                                                 THREADS, 0));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev.index()));
    resident = per_sm * sms;
  }
  TORCH_CHECK(blocks <= resident, "anchor_hot_map: grid exceeds co-resident blocks");
  // the grid barrier needs every block resident: a cooperative launch
  // guarantees it (an ordinary launch can leave a block waiting for an SM
  // that the spinning blocks hold)
  const int32_t* idx_p = idx.data_ptr<int32_t>();
  const int64_t* new_p = new_slots.data_ptr<int64_t>();
  int32_t* out_p = out.data_ptr<int32_t>();
  const int4* src_p = reinterpret_cast<const int4*>(src.data_ptr());
  int4* dst_p = reinterpret_cast<int4*>(dst.data_ptr());
  int n32 = (int)n, t32 = t, nrows32 = nrows;
  void* args[] = {&st, &idx_p, &n32, &new_p, &t32, &out_p, &src_p, &dst_p, &nrows32};
  C10_CUDA_CHECK(cudaLaunchCooperativeKernel((void*)map_kernel<4>, dim3(blocks), dim3(THREADS),
                                             args, 0, stream));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("anchor_hot_map", &anchor_hot_map); }

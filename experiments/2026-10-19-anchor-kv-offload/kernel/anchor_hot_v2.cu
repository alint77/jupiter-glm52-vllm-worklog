// Anchor hot cache (v2): an HBM buffer of N rows in front of an anchor
// layer's fp8_ds_mla main cache on Grace. Per decode step and anchor, the
// converted top-k (rank-local slots) becomes buffer rows in two launches:
//   lookup  hits stamp their row and set its CLOCK reference bit (one packed
//           status word, epoch << 1 | ref); misses claim
//           their slot (where[s]: -1 -> -2, one winner per slot) and are
//           listed; this step's tokens map to reserved rows N + j and their
//           slots are invalidated (the KV write changes the slot's content).
//           The last block to finish sweeps the CLOCK hand once for all
//           claims (block scan over a window of candidates: a row read this
//           step is skipped, a set reference bit is cleared and passed) and
//           publishes where[s].
//   copy    missed rows Grace -> HBM, one 16 B word per thread; entries that
//           missed read their row. The last block advances the step.
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
constexpr int LOOKUP_THREADS = 1024;
constexpr int SWEEP_V = 4;  // candidates per thread per sweep round

// ctl: [0] step (epoch; status holds it << 1, so < 2^30 steps), [1] CLOCK hand, [2] claimed slots, [3] pending entries,
// [4] overflow (claim without a victim: must stay 0), [5] candidates swept,
// [6] claims so far, [7] lookup blocks done, [8] copy blocks done.
// [4], [5], [6] are cumulative; [2], [3] per call.
struct State {
  int32_t* where;     // [S] slot -> row, -1 absent, -2 claimed this step
  int32_t* slot_of;   // [N] row -> slot, -1 free
  int32_t* status;    // [N] (step last read or filled) << 1 | CLOCK ref bit
  int32_t* ctl;
  int32_t* miss;      // [MAXE] claimed slots
  int32_t* miss_row;  // [MAXE] their rows
  int32_t* pending;   // [MAXE] entries waiting for their row
};

// Accesses that race with other threads of the same launch.
__device__ __forceinline__ int ld_rlx(int32_t* p) {
  return cuda::atomic_ref<int32_t, cuda::thread_scope_device>(*p).load(
      cuda::memory_order_relaxed);
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

// True in exactly one block: the last to arrive, after every block's writes.
__device__ __forceinline__ bool last_block(int32_t* done) {
  __shared__ bool last;
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    last = atomicAdd(done, 1) == (int)gridDim.x - 1;
  }
  __syncthreads();
  if (last) __threadfence();
  return last;
}

// One block gives every claim a victim, in hand order. Candidate p of a round
// is row (hand + p) % N; the hand stops right after the candidate taken by
// the last claim, so exactly the rows it passed lose their reference bit.
__device__ void sweep(State st, int step, int nrows) {
  using Scan = cub::BlockScan<int, LOOKUP_THREADS>;
  __shared__ typename Scan::TempStorage scan;
  __shared__ int sh_stop, sh_taken;
  const int nmiss = ld_rlx(&st.ctl[2]);
  int hand = ld_rlx(&st.ctl[1]);
  int assigned = 0;
  const int max_rounds = 2 * (nrows / (LOOKUP_THREADS * SWEEP_V) + 1) + 1;
  for (int round = 0; assigned < nmiss && round < max_rounds; ++round) {
    const int need = nmiss - assigned;
    // thread t owns candidates p = t * V + v (contiguous, scan in hand order)
    int c[SWEEP_V], stv[SWEEP_V];
    bool ok[SWEEP_V], refd[SWEEP_V];
    int cnt = 0;
#pragma unroll
    for (int v = 0; v < SWEEP_V; ++v) {
      int p = threadIdx.x * SWEEP_V + v;
      c[v] = (hand + p) % nrows;
      stv[v] = ld_rlx(&st.status[c[v]]);
      bool fresh = (stv[v] >> 1) != step;
      refd[v] = fresh && (stv[v] & 1);
      ok[v] = fresh && !refd[v];
      cnt += ok[v];
    }
    int off, total;
    Scan(scan).ExclusiveSum(cnt, off, total);
    if (threadIdx.x == 0) {
      sh_stop = LOOKUP_THREADS * SWEEP_V;
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
        int s = ld_rlx(&st.miss[i]);
        int old = ld_rlx(&st.slot_of[c[v]]);
        if (old >= 0 && ld_rlx(&st.where[old]) == c[v]) st.where[old] = -1;
        st.slot_of[c[v]] = s;
        st.status[c[v]] = step << 1;
        st.where[s] = c[v];
        st.miss_row[i] = c[v];
      }
    }
    hand = (hand + stop) % nrows;
    assigned += sh_taken;
    if (threadIdx.x == 0) atomicAdd(&st.ctl[5], stop);
    __syncthreads();
  }
  for (int i = assigned + threadIdx.x; i < nmiss; i += LOOKUP_THREADS) {
    st.where[ld_rlx(&st.miss[i])] = -1;
    st.miss_row[i] = -1;
  }
  if (threadIdx.x == 0) {
    st.ctl[1] = hand;
    if (assigned < nmiss) atomicAdd(&st.ctl[4], nmiss - assigned);
  }
}

__global__ void __launch_bounds__(LOOKUP_THREADS)
lookup_kernel(State st, const int32_t* __restrict__ idx, int n,
              const int32_t* __restrict__ new_slots, int t,
              int32_t* __restrict__ out, int nrows) {
  __shared__ int32_t sh_new[64];
  if (threadIdx.x < 64) sh_new[threadIdx.x] = threadIdx.x < t ? new_slots[threadIdx.x] : -1;
  __syncthreads();
  const int step = st.ctl[0] + 1;
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
  for (int base = blockIdx.x * blockDim.x; base < n; base += gridDim.x * blockDim.x) {
    int e = base + threadIdx.x;
    int s = e < n ? idx[e] : -1;
    int o = -1;
    bool pend = false, claim = false;
    if (s >= 0) {
      int j = -1;
      for (int k = 0; k < t; ++k) j = (sh_new[k] == s) ? k : j;
      if (j >= 0) {
        o = nrows + j;
      } else {
        int r = ld_rlx(&st.where[s]);
        if (r >= 0) {
          o = r;
          st_rlx(&st.status[r], (step << 1) | 1);
        } else {
          pend = true;
          if (r == -1) claim = atomicCAS(&st.where[s], -1, -2) == -1;
        }
      }
    }
    if (e < n) out[e] = o;
    int p = warp_append(&st.ctl[3], pend);
    if (pend) st.pending[p] = e;
    int m = warp_append(&st.ctl[2], claim);
    if (claim) st.miss[m] = s;
  }
  if (last_block(&st.ctl[7])) {
    sweep(st, step, nrows);
    if (threadIdx.x == 0) st.ctl[7] = 0;
  }
}

template <int UNROLL>
__global__ void copy_kernel(State st, const int4* __restrict__ src, int4* __restrict__ dst,
                            const int32_t* __restrict__ idx, int32_t* __restrict__ out) {
  const int nmiss = st.ctl[2];
  const int np = st.ctl[3];
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  const int total = nmiss * ROW_WORDS;
  for (int f0 = tid; f0 < total; f0 += stride * UNROLL) {
    int4 v[UNROLL];
    int64_t to[UNROLL];
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      int f = f0 + u * stride;
      to[u] = -1;
      if (f < total) {
        int i = f / ROW_WORDS, w = f - i * ROW_WORDS;
        int r = st.miss_row[i];
        if (r >= 0) {
          v[u] = src[(int64_t)st.miss[i] * ROW_WORDS + w];
          to[u] = (int64_t)r * ROW_WORDS + w;
        }
      }
    }
#pragma unroll
    for (int u = 0; u < UNROLL; ++u)
      if (to[u] >= 0) dst[to[u]] = v[u];
  }
  for (int p = tid; p < np; p += stride) {
    int e = st.pending[p];
    out[e] = st.where[idx[e]];
  }
  if (last_block(&st.ctl[8]) && threadIdx.x == 0) {
    st.ctl[0] += 1;
    st.ctl[6] += nmiss;
    st.ctl[2] = 0;
    st.ctl[3] = 0;
    st.ctl[8] = 0;
  }
}

}  // namespace

void anchor_hot_map(torch::Tensor idx, torch::Tensor new_slots, torch::Tensor out,
                    torch::Tensor src, torch::Tensor dst, torch::Tensor where,
                    torch::Tensor slot_of, torch::Tensor status, torch::Tensor ctl, torch::Tensor miss, torch::Tensor miss_row,
                    torch::Tensor pending, int64_t select_ctas, int64_t copy_ctas) {
  for (auto* x : {&idx, &new_slots, &out, &where, &slot_of, &status, &ctl, &miss,
                  &miss_row, &pending}) {
    TORCH_CHECK(x->is_cuda() && x->is_contiguous() && x->scalar_type() == at::kInt);
  }
  TORCH_CHECK(dst.is_cuda() && dst.is_contiguous() && src.is_contiguous());
  const int64_t n = idx.numel();
  const int t = new_slots.numel();
  const int nrows = slot_of.numel();
  const int64_t nslots = where.numel();
  // A sweep window holds distinct rows. Every claim gets a victim within two
  // laps when hit rows + claims <= nrows, which n <= nrows guarantees.
  TORCH_CHECK(t <= 64 && n <= nrows && nrows >= LOOKUP_THREADS * SWEEP_V);
  TORCH_CHECK(out.numel() == n && miss.numel() >= n && miss_row.numel() >= n &&
              pending.numel() >= n && status.numel() == nrows && ctl.numel() >= 9);
  TORCH_CHECK(src.nbytes() >= nslots * ROW_WORDS * 16 &&
              dst.nbytes() >= (int64_t)(nrows + 64) * ROW_WORDS * 16);
  TORCH_CHECK(copy_ctas > 0 && nslots < (1ll << 31));
  State st{where.data_ptr<int32_t>(), slot_of.data_ptr<int32_t>(),
           status.data_ptr<int32_t>(), ctl.data_ptr<int32_t>(),
           miss.data_ptr<int32_t>(), miss_row.data_ptr<int32_t>(),
           pending.data_ptr<int32_t>()};
  auto stream = at::cuda::getCurrentCUDAStream();
  const int lk = std::max<int>(std::min<int64_t>((n + LOOKUP_THREADS - 1) / LOOKUP_THREADS, 132), 1);
  lookup_kernel<<<lk, LOOKUP_THREADS, 0, stream>>>(st, idx.data_ptr<int32_t>(), n,
                                                    new_slots.data_ptr<int32_t>(), t,
                                                    out.data_ptr<int32_t>(), nrows);
  copy_kernel<4><<<copy_ctas, 256, 0, stream>>>(
      st, reinterpret_cast<const int4*>(src.data_ptr()),
      reinterpret_cast<int4*>(dst.data_ptr()), idx.data_ptr<int32_t>(),
      out.data_ptr<int32_t>());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("anchor_hot_map", &anchor_hot_map); }

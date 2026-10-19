// Anchor hot cache: an HBM buffer of N rows in front of an anchor layer's
// fp8_ds_mla main cache on Grace. Per decode step and anchor, the converted
// top-k (rank-local slots) becomes buffer rows:
//   lookup  hits stamp their row and set its CLOCK reference bit; misses claim
//           their slot (where[s]: -1 -> -2, one winner per slot) and are
//           listed; this step's tokens map to reserved rows N + j and their
//           slots are invalidated (the KV write changes the slot's content)
//   select  each claimed slot takes a CLOCK victim not read this step and
//           publishes where[s]
//   copy    missed rows Grace -> HBM, every 16 B word its own thread; entries
//           that missed read their row
//   finish  next step
// Entries keep position, multiplicity and -1, so FlashMLA reads the same
// bytes in the same order as from the full cache.
#include <cuda_runtime.h>
#include <stdint.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

namespace {

constexpr int ROW_WORDS = 656 / 16;  // 41 x 16 B

// ctl: [0] step, [1] CLOCK hand, [2] claimed slots, [3] pending entries,
// [4] overflow (a claim without a victim: must stay 0), [5] candidates swept,
// [6] claims so far
struct State {
  int32_t* where;     // [S] slot -> row, -1 absent, -2 claimed this step
  int32_t* slot_of;   // [N] row -> slot, -1 free
  int32_t* stamp;     // [N] step the row was last read or filled
  int8_t* ref;        // [N] CLOCK reference bit
  int32_t* ctl;
  int32_t* miss;      // [MAXE] claimed slots
  int32_t* miss_row;  // [MAXE] their rows
  int32_t* pending;   // [MAXE] entries waiting for their row
};

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

__global__ void lookup_kernel(State st, const int32_t* __restrict__ idx, int n,
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
      int r = st.where[s];
      if (r >= 0) {
        st.where[s] = -1;
        st.slot_of[r] = -1;
        st.ref[r] = 0;
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
        int r = st.where[s];
        if (r >= 0) {
          o = r;
          st.stamp[r] = step;
          st.ref[r] = 1;
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
}

// A warp serves up to 32 claims: per round it takes as many CLOCK candidates
// as it still needs with one atomic on the hand; a candidate read this step is
// skipped, one with its reference bit set gets a second chance. A victim is
// owned through its stamp (atomicExch to this step), so a hand that wrapped
// within the step can never give one row to two slots.
__global__ void select_kernel(State st, int nrows) {
  const int step = st.ctl[0] + 1;
  const int nmiss = st.ctl[2];
  const int lane = threadIdx.x & 31;
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int nwarps = (gridDim.x * blockDim.x) >> 5;
  for (int b = warp * 32; b < nmiss; b += nwarps * 32) {
    const int i = b + lane;
    bool need = i < nmiss;
    int victim = -1;
    // 32 candidates per round; a full sweep clears every reference bit, so
    // two sweeps always reach a row not read this step (N exceeds the rows a
    // step can read)
    const int max_rounds = 2 * (nrows / 32 + 1);
    for (int round = 0; round < max_rounds; ++round) {
      unsigned needm = __ballot_sync(0xffffffffu, need);
      if (!needm) break;
      const int k = __popc(needm);
      unsigned c0 = 0;
      if (lane == 0) {
        c0 = atomicAdd(reinterpret_cast<unsigned*>(&st.ctl[1]), 32u);
        atomicAdd(&st.ctl[5], 32);
      }
      c0 = __shfl_sync(0xffffffffu, c0, 0);
      const int c = (int)((c0 + lane) % (unsigned)nrows);
      bool ok = st.stamp[c] != step;
      if (ok && st.ref[c]) {
        st.ref[c] = 0;
        ok = false;
      }
      // claim only as many as the warp needs; the rest are passed over
      unsigned okm = __ballot_sync(0xffffffffu, ok);
      ok = ok && __popc(okm & ((1u << lane) - 1)) < k;
      if (ok) ok = atomicExch(&st.stamp[c], step) != step;
      okm = __ballot_sync(0xffffffffu, ok);
      while (okm && needm) {
        int from = __ffs(okm) - 1;
        int to = __ffs(needm) - 1;
        int cv = __shfl_sync(0xffffffffu, c, from);
        if (lane == to) {
          victim = cv;
          need = false;
        }
        okm &= okm - 1;
        needm &= needm - 1;
      }
    }
    if (i < nmiss) {
      int s = st.miss[i];
      if (victim >= 0) {
        int old = st.slot_of[victim];
        if (old >= 0) st.where[old] = -1;
        st.slot_of[victim] = s;
        st.ref[victim] = 0;
        st.where[s] = victim;
      } else {
        atomicAdd(&st.ctl[4], 1);
        st.where[s] = -1;
      }
      st.miss_row[i] = victim;
    }
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
}

__global__ void finish_kernel(State st) {
  st.ctl[0] += 1;
  st.ctl[6] += st.ctl[2];
  st.ctl[2] = 0;
  st.ctl[3] = 0;
}

}  // namespace

void anchor_hot_map(torch::Tensor idx, torch::Tensor new_slots, torch::Tensor out,
                    torch::Tensor src, torch::Tensor dst, torch::Tensor where,
                    torch::Tensor slot_of, torch::Tensor stamp, torch::Tensor ref,
                    torch::Tensor ctl, torch::Tensor miss, torch::Tensor miss_row,
                    torch::Tensor pending, int64_t select_ctas, int64_t copy_ctas) {
  State st{where.data_ptr<int32_t>(), slot_of.data_ptr<int32_t>(),
           stamp.data_ptr<int32_t>(), ref.data_ptr<int8_t>(),
           ctl.data_ptr<int32_t>(),   miss.data_ptr<int32_t>(),
           miss_row.data_ptr<int32_t>(), pending.data_ptr<int32_t>()};
  auto stream = at::cuda::getCurrentCUDAStream();
  const int n = idx.numel();
  const int t = new_slots.numel();
  TORCH_CHECK(t <= 64 && n <= miss.numel());
  const int nrows = slot_of.numel();
  const int lk = std::max(std::min((n + 255) / 256, 132), 1);
  lookup_kernel<<<lk, 256, 0, stream>>>(st, idx.data_ptr<int32_t>(), n,
                                         new_slots.data_ptr<int32_t>(), t,
                                         out.data_ptr<int32_t>(), nrows);
  select_kernel<<<select_ctas, 128, 0, stream>>>(st, nrows);
  copy_kernel<4><<<copy_ctas, 256, 0, stream>>>(
      st, reinterpret_cast<const int4*>(src.data_ptr()),
      reinterpret_cast<int4*>(dst.data_ptr()), idx.data_ptr<int32_t>(),
      out.data_ptr<int32_t>());
  finish_kernel<<<1, 1, 0, stream>>>(st);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("anchor_hot_map", &anchor_hot_map); }

// Push-mode, barrier-free one-shot collectives (included inside namespace
// one_shot). Every rank owns a symmetric buffer:
//   [flags: FlagType[kPushMaxBlocks][8]][half 0][half 1]
// and each half holds one slot of slot_bytes per source rank. A call takes
// epoch e = *epoch + 1 and uses half e & 1. Block b of the sender stores its
// chunk into every peer's slot, then raises flags[b][rank] = e on the peer
// (release); block b of the receiver waits for flags[b][r] >= e (acquire)
// from every peer, then reads its chunk of the slots locally. No barriers:
// a rank writes half (e + 1) & 1 only in call e + 1, after it has received
// every peer's call-e data, so every peer has finished call e - 1 (the last
// reader of that half). Flags only grow, so no resets.

constexpr int kPushMaxBlocks = 132;
constexpr int64_t kPushFlagBytes = 64 * 1024;
static_assert(kPushMaxBlocks * 8 * sizeof(vllm::FlagType) <= kPushFlagBytes);

struct __align__(16) PushBufs {
  char* base[8];  // every rank's buffer (flags at 0, halves after)
};

struct PushState {
  PushBufs bufs;
  int64_t slot_bytes = 0;
  uint32_t* epoch = nullptr;  // [0] epoch, [1] blocks done (this rank)
  bool ready = false;
};
static PushState g_push;

void push_setup(std::vector<int64_t> ptrs, int64_t slot_bytes,
                int64_t epoch_ptr) {
  TORCH_CHECK(ptrs.size() <= 8, "at most 8 ranks");
  for (size_t i = 0; i < ptrs.size(); ++i)
    g_push.bufs.base[i] = reinterpret_cast<char*>(ptrs[i]);
  g_push.slot_bytes = slot_bytes;
  g_push.epoch = reinterpret_cast<uint32_t*>(epoch_ptr);
  g_push.ready = true;
}

int64_t push_buffer_bytes(int64_t slot_bytes, int64_t world) {
  return kPushFlagBytes + 2 * world * slot_bytes;
}

DINLINE vllm::FlagType* push_flags(char* base) {
  return reinterpret_cast<vllm::FlagType*>(base);
}

// Slot of source rank src in half h of a rank's buffer.
DINLINE char* push_slot(char* base, int64_t slot_bytes, int ngpus, uint32_t h,
                        int src) {
  return base + kPushFlagBytes + (int64_t(h) * ngpus + src) * slot_bytes;
}

template <int ngpus>
DINLINE void push_signal(const PushBufs& bufs, int rank, uint32_t e) {
  __syncthreads();
  if (threadIdx.x < ngpus && threadIdx.x != rank)
    vllm::st_flag_release(
        push_flags(bufs.base[threadIdx.x]) + blockIdx.x * 8 + rank, e);
}

template <int ngpus>
DINLINE void push_wait(char* self, int rank, uint32_t e) {
  if (threadIdx.x < ngpus && threadIdx.x != rank) {
    vllm::FlagType* f = push_flags(self) + blockIdx.x * 8 + threadIdx.x;
    while (static_cast<int32_t>(vllm::ld_flag_acquire(f) - e) < 0);
  }
  __syncthreads();
}

// The last block to finish advances the epoch for the next call.
DINLINE void push_done(uint32_t* epoch, uint32_t e) {
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    if (atomicAdd(epoch + 1, 1u) == gridDim.x - 1) {
      epoch[1] = 0;
      epoch[0] = e;
      __threadfence();
    }
  }
}

// a [T, H, Da], b [T, H, Db] (strided rows) -> out [T, ngpus * H, Da + Db].
// Units are uint4; this rank's units are numbered u = (t * H + h) * row + j,
// block b owns units [b * chunk, (b + 1) * chunk).
template <int ngpus, int U>
__global__ void __launch_bounds__(kThreads, 1)
    push_gather_cat_kernel(PushBufs bufs, int64_t slot_bytes,
                           uint32_t* epoch, const uint4* __restrict__ a,
                           const uint4* __restrict__ b, uint4* __restrict__ out,
                           int rank, uint32_t T, uint32_t H, uint32_t a_row,
                           uint32_t sa_t, uint32_t sa_h, uint32_t b_row,
                           uint32_t sb_t, uint32_t sb_h, uint32_t chunk) {
  const uint32_t e = __ldcg(epoch) + 1;
  const uint32_t row = a_row + b_row;
  const uint32_t total = T * H * row;
  const uint32_t lo = blockIdx.x * chunk;
  const uint32_t hi = min(total, lo + chunk);
  uint4* dst[ngpus];
#pragma unroll
  for (int r = 0; r < ngpus; ++r)
    dst[r] = reinterpret_cast<uint4*>(
        push_slot(bufs.base[r], slot_bytes, ngpus, e & 1, rank));
  // send: this rank's units to every peer's slot, and into out directly
  for (uint32_t u0 = lo + threadIdx.x; u0 < hi; u0 += U * kThreads) {
    uint4 v[U];
#pragma unroll
    for (int k = 0; k < U; ++k) {
      const uint32_t u = u0 + k * kThreads;
      if (u < hi) {
        const uint32_t th = u / row, j = u - th * row;
        const uint32_t t = th / H, h = th - t * H;
        v[k] = j < a_row ? a[t * sa_t + h * sa_h + j]
                         : b[t * sb_t + h * sb_h + (j - a_row)];
      }
    }
#pragma unroll
    for (int k = 0; k < U; ++k) {
      const uint32_t u = u0 + k * kThreads;
      if (u < hi) {
#pragma unroll
        for (int r = 0; r < ngpus; ++r)
          if (r != rank) dst[r][u] = v[k];
        const uint32_t th = u / row, j = u - th * row;
        const uint32_t t = th / H, h = th - t * H;
        out[(t * ngpus * H + rank * H + h) * row + j] = v[k];
      }
    }
  }
  push_signal<ngpus>(bufs, rank, e);
  push_wait<ngpus>(bufs.base[rank], rank, e);
  // receive: peers' units of this chunk from the local slots into out
  const uint32_t n = hi > lo ? hi - lo : 0;
  for (uint32_t i0 = threadIdx.x; i0 < (ngpus - 1) * n; i0 += U * kThreads) {
    uint4 v[U];
    uint32_t o[U];
#pragma unroll
    for (int k = 0; k < U; ++k) {
      const uint32_t i = i0 + k * kThreads;
      if (i < (ngpus - 1) * n) {
        const uint32_t q = i / n;
        const int r = q + (q >= rank);
        const uint32_t u = lo + (i - q * n);
        v[k] = __ldcg(reinterpret_cast<const uint4*>(push_slot(
                          bufs.base[rank], slot_bytes, ngpus, e & 1, r)) +
                      u);
        const uint32_t th = u / row, j = u - th * row;
        const uint32_t t = th / H, h = th - t * H;
        o[k] = (t * ngpus * H + r * H + h) * row + j;
      }
    }
#pragma unroll
    for (int k = 0; k < U; ++k)
      if (i0 + k * kThreads < (ngpus - 1) * n) out[o[k]] = v[k];
  }
  push_done(epoch, e);
}

// DCP combine, push: rank s sends each peer r its out[t, r's heads, :] and
// lse[t, r's heads]; the receiver combines with the same arithmetic and rank
// order as lse_reduce_scatter_kernel. Pairs p = t * H_local + hl, block b
// owns pairs [b * chunk, (b + 1) * chunk). Slot: [pairs * D_PACKS packs]
// [pairs floats].
template <typename T, int ngpus, bool base_e, int D_PACKS, int U>
__global__ void __launch_bounds__(kThreads, 1)
    push_lse_rs_kernel(PushBufs bufs, int64_t slot_bytes, uint32_t* epoch,
                       const T* __restrict__ outp, const float* __restrict__ lsep,
                       T* __restrict__ result, int rank, uint32_t n_pairs,
                       uint32_t H, uint32_t H_local, int64_t lse_st,
                       int64_t lse_sh, uint32_t chunk) {
  using P = typename vllm::packed_t<T>::P;
  using A = typename vllm::packed_t<T>::A;
  constexpr int kPack = 16 / sizeof(T);
  constexpr int PER_LANE = D_PACKS / 32;
  const uint32_t e = __ldcg(epoch) + 1;
  const uint32_t lo = blockIdx.x * chunk;
  const uint32_t hi = min(n_pairs, lo + chunk);
  const uint32_t n = hi > lo ? hi - lo : 0;
  const P* src = reinterpret_cast<const P*>(outp);
  const int64_t lse_off = int64_t(n_pairs) * D_PACKS * 16;
  // send: (peer, pair, pack) items of this chunk
  const uint32_t items = (ngpus - 1) * n * D_PACKS;
  for (uint32_t i0 = threadIdx.x; i0 < items; i0 += U * kThreads) {
    P v[U];
#pragma unroll
    for (int k = 0; k < U; ++k) {
      const uint32_t i = i0 + k * kThreads;
      if (i < items) {
        const uint32_t g = i / D_PACKS, j = i % D_PACKS;
        const uint32_t q = g % (ngpus - 1);
        const int r = q + (q >= rank);
        const uint32_t p = lo + g / (ngpus - 1);
        const uint32_t t = p / H_local, hl = p - t * H_local;
        v[k] = src[(t * H + r * H_local + hl) * D_PACKS + j];
      }
    }
#pragma unroll
    for (int k = 0; k < U; ++k) {
      const uint32_t i = i0 + k * kThreads;
      if (i < items) {
        const uint32_t g = i / D_PACKS, j = i % D_PACKS;
        const uint32_t q = g % (ngpus - 1);
        const int r = q + (q >= rank);
        const uint32_t p = lo + g / (ngpus - 1);
        reinterpret_cast<P*>(push_slot(bufs.base[r], slot_bytes, ngpus, e & 1,
                                       rank))[p * D_PACKS + j] = v[k];
      }
    }
  }
  for (uint32_t i = threadIdx.x; i < (ngpus - 1) * n; i += kThreads) {
    const uint32_t q = i / n;
    const int r = q + (q >= rank);
    const uint32_t p = lo + (i - q * n);
    const uint32_t t = p / H_local, hl = p - t * H_local;
    reinterpret_cast<float*>(push_slot(bufs.base[r], slot_bytes, ngpus, e & 1,
                                       rank) +
                             lse_off)[p] =
        lsep[t * lse_st + (r * H_local + hl) * lse_sh];
  }
  push_signal<ngpus>(bufs, rank, e);
  push_wait<ngpus>(bufs.base[rank], rank, e);
  // receive: one warp per pair
  const int lane = threadIdx.x % 32;
  for (uint32_t p = lo + threadIdx.x / 32; p < hi; p += kThreads / 32) {
    const uint32_t t = p / H_local, hl = p - t * H_local;
    const uint32_t h = rank * H_local + hl;
    P x[ngpus][PER_LANE];
#pragma unroll
    for (int r = 0; r < ngpus; ++r) {
      const P* s = r == rank
                       ? src + (t * H + h) * D_PACKS
                       : reinterpret_cast<const P*>(push_slot(
                             bufs.base[rank], slot_bytes, ngpus, e & 1, r)) +
                             p * D_PACKS;
#pragma unroll
      for (int i = 0; i < PER_LANE; ++i) {
        if (r == rank)
          x[r][i] = s[i * 32 + lane];
        else {
          uint4 w = __ldcg(reinterpret_cast<const uint4*>(s + i * 32 + lane));
          x[r][i] = *reinterpret_cast<P*>(&w);
        }
      }
    }
    float mine = 0.f;
    if (lane < ngpus)
      mine = lane == rank
                 ? lsep[t * lse_st + h * lse_sh]
                 : __ldcg(reinterpret_cast<const float*>(
                              push_slot(bufs.base[rank], slot_bytes, ngpus,
                                        e & 1, lane) +
                              lse_off) +
                          p);
    float raw[ngpus], l[ngpus];
    float m = -INFINITY;
#pragma unroll
    for (int r = 0; r < ngpus; ++r) {
      raw[r] = __shfl_sync(0xffffffffu, mine, r);
      l[r] = nan_inf_to_neg_inf(raw[r]);
      m = fmaxf(m, l[r]);
    }
    if (m == -INFINITY) m = 0.f;
    float ex[ngpus];
#pragma unroll
    for (int r = 0; r < ngpus; ++r) ex[r] = cp_exp<base_e>(l[r] - m);
#pragma unroll
    for (int stride = ngpus / 2; stride >= 1; stride /= 2)
#pragma unroll
      for (int r = 0; r < stride; ++r) ex[r] += ex[r + stride];
    const float lse = cp_log<base_e>(ex[0]) + m;
    float f[ngpus];
#pragma unroll
    for (int r = 0; r < ngpus; ++r)
      f[r] = cp_exp<base_e>(nan_inf_to_neg_inf(raw[r] - lse));
#pragma unroll
    for (int i = 0; i < PER_LANE; ++i) {
      A acc;
#pragma unroll
      for (int r = 0; r < ngpus; ++r) {
        A v = vllm::upcast(x[r][i]);
#pragma unroll
        for (int k = 0; k < kPack; ++k)
          v.data[k] = f[r] == 0.f ? 0.f : v.data[k] * f[r];
        v = vllm::upcast(vllm::downcast<P>(v));
        if (r == 0) {
          acc = v;
        } else {
          vllm::packed_assign_add(acc, v);
        }
      }
      reinterpret_cast<P*>(result)[p * D_PACKS + i * 32 + lane] =
          vllm::downcast<P>(acc);
    }
  }
  push_done(epoch, e);
}

static int g_push_blocks = 64;
static int g_push_unroll = 4;
void push_config(int64_t blocks, int64_t unroll) {
  TORCH_CHECK(blocks >= 1 && blocks <= kPushMaxBlocks, "blocks out of range");
  g_push_blocks = static_cast<int>(blocks);
  g_push_unroll = static_cast<int>(unroll);
}

void push_all_gather_cat(torch::Tensor a, torch::Tensor b, torch::Tensor out,
                         int64_t world, int64_t rank) {
  TORCH_CHECK(g_push.ready, "push_setup first");
  TORCH_CHECK(world == 4, "bench: 4 ranks");
  const int64_t es = a.element_size(), u = 16 / es;
  const uint32_t T = a.size(0), H = a.size(1);
  const uint32_t a_row = a.size(2) / u, b_row = b.size(2) / u;
  const uint32_t total = T * H * (a_row + b_row);
  TORCH_CHECK(int64_t(total) * 16 <= g_push.slot_bytes, "slot too small");
  const int blocks = g_push_blocks;
  const uint32_t chunk = (total + blocks - 1) / blocks;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
#define PGC(UU)                                                               \
  push_gather_cat_kernel<4, UU><<<blocks, kThreads, 0, stream>>>(             \
      g_push.bufs, g_push.slot_bytes, g_push.epoch,                           \
      reinterpret_cast<const uint4*>(a.data_ptr()),                           \
      reinterpret_cast<const uint4*>(b.data_ptr()),                           \
      reinterpret_cast<uint4*>(out.data_ptr()), rank, T, H, a_row,            \
      a.stride(0) / u, a.stride(1) / u, b_row, b.stride(0) / u,               \
      b.stride(1) / u, chunk)
  switch (g_push_unroll) {
    case 1: PGC(1); break;
    case 2: PGC(2); break;
    case 4: PGC(4); break;
    case 8: PGC(8); break;
    default: TORCH_CHECK(false, "unroll 1/2/4/8");
  }
#undef PGC
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void push_lse_reduce_scatter(torch::Tensor out, torch::Tensor lse,
                             torch::Tensor result, bool base_e, int64_t world,
                             int64_t rank) {
  TORCH_CHECK(g_push.ready, "push_setup first");
  TORCH_CHECK(world == 4 && out.is_contiguous(), "bench: 4 ranks, contiguous");
  const uint32_t T = out.size(0), H = out.size(1), D = out.size(2);
  const uint32_t H_local = H / world;
  const uint32_t D_packs = D * out.element_size() / 16;
  TORCH_CHECK(D_packs == 64, "bench: 64 packs");
  const uint32_t n_pairs = T * H_local;
  TORCH_CHECK(int64_t(n_pairs) * (D_packs * 16 + 4) <= g_push.slot_bytes,
              "slot too small");
  const int blocks = g_push_blocks;
  const uint32_t chunk = (n_pairs + blocks - 1) / blocks;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto* o = reinterpret_cast<const nv_bfloat16*>(out.data_ptr());
  auto* l = lse.data_ptr<float>();
  auto* res = reinterpret_cast<nv_bfloat16*>(result.data_ptr());
#define PLR(BE, UU)                                                          \
  push_lse_rs_kernel<nv_bfloat16, 4, BE, 64, UU>                             \
      <<<blocks, kThreads, 0, stream>>>(g_push.bufs, g_push.slot_bytes,      \
                                        g_push.epoch, o, l, res, rank,       \
                                        n_pairs, H, H_local, lse.stride(0),  \
                                        lse.stride(1), chunk)
  TORCH_CHECK(base_e, "bench: base e");
  switch (g_push_unroll) {
    case 1: PLR(true, 1); break;
    case 2: PLR(true, 2); break;
    case 4: PLR(true, 4); break;
    case 8: PLR(true, 8); break;
    default: TORCH_CHECK(false, "unroll 1/2/4/8");
  }
#undef PLR
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

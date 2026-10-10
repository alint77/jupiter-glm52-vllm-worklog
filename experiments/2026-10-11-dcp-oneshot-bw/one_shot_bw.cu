// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// One-shot all-gather and reduce-scatter over the IPC buffers of vLLM's
// custom all-reduce (csrc/custom_all_reduce.cuh), for small collectives where
// NCCL's ring sits at its ~7-12 us latency floor -- decode context
// parallelism's per-layer query gather, LSE gather and output reduce-scatter.
//
// Header-only reuse: the CustomAllreduce object built by the custom
// all-reduce extension is taken by pointer, so these kernels share its peer
// pointers, graph-buffer registration and barrier flags. Every rank issues
// the same sequence of launches with the same grids, which keeps the
// per-block flags in lockstep exactly as for the all-reduce kernels.
//
// Layout: a rank's input is [outer, inner] (inner contiguous, 16-byte
// multiple). all_gather writes out[t, r * inner + j] = in_r[t, j], which is a
// gather along the dimension that `inner` ends at; reduce_scatter writes
// out[t, j] = sum_r in_r[t, rank * inner + j] from inputs of [outer,
// world * inner].

#include <vector>

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

#include "custom_all_reduce.cuh"

namespace one_shot {

using vllm::CustomAllreduce;
using vllm::RankData;
using vllm::RankSignals;
using vllm::Signal;

constexpr int kThreads = 512;

template <int ngpus>
__global__ void __launch_bounds__(kThreads, 1)
    gather_kernel(RankData* _dp, RankSignals sg, Signal* self_sg,
                  uint4* __restrict__ out, int rank, int64_t outer,
                  int64_t inner) {
  auto dp = *_dp;
  vllm::barrier_at_start<ngpus>(sg, self_sg, rank);
  const int64_t per_rank = outer * inner;
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < ngpus * per_rank;
       i += gridDim.x * blockDim.x) {
    const int64_t r = i / per_rank, rem = i - r * per_rank;
    const int64_t t = rem / inner, j = rem - t * inner;
    out[(t * ngpus + r) * inner + j] =
        reinterpret_cast<const uint4*>(dp.ptrs[r])[rem];
  }
  vllm::barrier_at_end<ngpus, true>(sg, self_sg, rank);
}

template <typename T, int ngpus>
__global__ void __launch_bounds__(kThreads, 1)
    reduce_scatter_kernel(RankData* _dp, RankSignals sg, Signal* self_sg,
                          T* __restrict__ result, int rank, int64_t outer,
                          int64_t inner) {
  using P = typename vllm::packed_t<T>::P;
  using A = typename vllm::packed_t<T>::A;
  auto dp = *_dp;
  vllm::barrier_at_start<ngpus>(sg, self_sg, rank);
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < outer * inner;
       i += gridDim.x * blockDim.x) {
    const int64_t t = i / inner, j = i - t * inner;
    const int64_t src = (t * ngpus + rank) * inner + j;
    A acc = vllm::upcast(reinterpret_cast<const P*>(dp.ptrs[0])[src]);
#pragma unroll
    for (int r = 1; r < ngpus; ++r)
      vllm::packed_assign_add(
          acc, vllm::upcast(reinterpret_cast<const P*>(dp.ptrs[r])[src]));
    reinterpret_cast<P*>(result)[i] = vllm::downcast<P>(acc);
  }
  vllm::barrier_at_end<ngpus, true>(sg, self_sg, rank);
}

// Two inputs gathered as one: out[t, r * H + h, :] = concat(a_r[t, h, :],
// b_r[t, h, :]), i.e. torch.cat([a, b], -1) followed by the all-gather along
// heads, without materialising the concatenation. Rows are uint4 units; a and
// b may be strided in t and h (the inner row is contiguous).
template <int ngpus>
__global__ void __launch_bounds__(kThreads, 1)
    gather_cat_kernel(RankData* _da, RankData* _db, int64_t off_a,
                      int64_t off_b, RankSignals sg, Signal* self_sg,
                      uint4* __restrict__ out, int rank, int64_t T, int64_t H,
                      int64_t a_row, int64_t sa_t, int64_t sa_h, int64_t b_row,
                      int64_t sb_t, int64_t sb_h) {
  auto da = *_da;
  auto db = *_db;
  vllm::barrier_at_start<ngpus>(sg, self_sg, rank);
  const int64_t row = a_row + b_row;
  const int64_t per_rank = T * H * row;
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < ngpus * per_rank;
       i += gridDim.x * blockDim.x) {
    const int64_t r = i / per_rank, rem = i - r * per_rank;
    const int64_t th = rem / row, j = rem - th * row;
    const int64_t t = th / H, h = th - t * H;
    uint4 v;
    if (j < a_row) {
      v = reinterpret_cast<const uint4*>(
          reinterpret_cast<const char*>(da.ptrs[r]) +
          off_a)[t * sa_t + h * sa_h + j];
    } else {
      v = reinterpret_cast<const uint4*>(
          reinterpret_cast<const char*>(db.ptrs[r]) +
          off_b)[t * sb_t + h * sb_h + (j - a_row)];
    }
    out[(t * ngpus * H + r * H + h) * row + j] = v;
  }
  vllm::barrier_at_end<ngpus, true>(sg, self_sg, rank);
}

DINLINE float nan_inf_to_neg_inf(float x) {
  return (x != x || x == INFINITY) ? -INFINITY : x;
}

// exp / exp2 as Triton lowers them (ex2.approx), log / log2 via libdevice.
template <bool base_e>
DINLINE float cp_exp(float x) {
  float y;
  if constexpr (base_e) x *= 1.4426950408889634f;
  asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
template <bool base_e>
DINLINE float cp_log(float x) {
  if constexpr (base_e) return logf(x);
  return log2f(x);
}

// Decode context parallelism's attention combine in one launch: every rank
// holds out [T, H, D] (all heads over its KV shard) and lse [T, H]; this rank
// gets result [T, H / ngpus, D] for its heads,
//   sum_r bf16(out_r * exp(lse_r - lse)),  lse = logsumexp_r(lse_r),
// with the same arithmetic and summation order as the LSE all-gather,
// _correct_attn_cp_out_kernel and reduce-scatter it replaces.
template <typename T, int ngpus, bool base_e>
__global__ void __launch_bounds__(kThreads, 1)
    lse_reduce_scatter_kernel(RankData* _dout, RankData* _dlse, int64_t off_out,
                              int64_t off_lse, RankSignals sg, Signal* self_sg,
                              T* __restrict__ result, int rank, int64_t n_pairs,
                              int64_t H, int64_t H_local, int64_t D_packs,
                              int64_t lse_st, int64_t lse_sh) {
  using P = typename vllm::packed_t<T>::P;
  using A = typename vllm::packed_t<T>::A;
  constexpr int kPack = 16 / sizeof(T);
  __shared__ float factor[kThreads / 8][ngpus];
  auto dout = *_dout;
  auto dlse = *_dlse;
  const int pairs_per_block = kThreads / D_packs;
  vllm::barrier_at_start<ngpus>(sg, self_sg, rank);
  for (int64_t p0 = blockIdx.x * pairs_per_block; p0 < n_pairs;
       p0 += gridDim.x * pairs_per_block) {
    if (threadIdx.x < pairs_per_block && p0 + threadIdx.x < n_pairs) {
      const int64_t p = p0 + threadIdx.x;
      const int64_t t = p / H_local;
      const int64_t h = rank * H_local + (p - t * H_local);
      float raw[ngpus], l[ngpus];
      float m = -INFINITY;
#pragma unroll
      for (int r = 0; r < ngpus; ++r) {
        raw[r] = reinterpret_cast<const float*>(
            reinterpret_cast<const char*>(dlse.ptrs[r]) +
            off_lse)[t * lse_st + h * lse_sh];
        l[r] = nan_inf_to_neg_inf(raw[r]);
        m = fmaxf(m, l[r]);
      }
      if (m == -INFINITY) m = 0.f;
      // butterfly order, as Triton's tl.sum over the gathered ranks
      float e[ngpus];
#pragma unroll
      for (int r = 0; r < ngpus; ++r) e[r] = cp_exp<base_e>(l[r] - m);
#pragma unroll
      for (int stride = ngpus / 2; stride >= 1; stride /= 2)
#pragma unroll
        for (int r = 0; r < stride; ++r) e[r] += e[r + stride];
      const float lse = cp_log<base_e>(e[0]) + m;
#pragma unroll
      for (int r = 0; r < ngpus; ++r)
        factor[threadIdx.x][r] =
            cp_exp<base_e>(nan_inf_to_neg_inf(raw[r] - lse));
    }
    __syncthreads();
    const int64_t local = threadIdx.x / D_packs;
    const int64_t p = p0 + local;
    if (local < pairs_per_block && p < n_pairs) {
      const int64_t j = threadIdx.x - local * D_packs;
      const int64_t t = p / H_local;
      const int64_t h = rank * H_local + (p - t * H_local);
      const int64_t src = (t * H + h) * D_packs + j;
      A acc;
#pragma unroll
      for (int r = 0; r < ngpus; ++r) {
        const float f = factor[local][r];
        A x = vllm::upcast(reinterpret_cast<const P*>(
            reinterpret_cast<const char*>(dout.ptrs[r]) + off_out)[src]);
#pragma unroll
        for (int k = 0; k < kPack; ++k)
          x.data[k] = f == 0.f ? 0.f : x.data[k] * f;
        // the corrected partial is stored at T's precision before the sum
        x = vllm::upcast(vllm::downcast<P>(x));
        if (r == 0) {
          acc = x;
        } else {
          vllm::packed_assign_add(acc, x);
        }
      }
      reinterpret_cast<P*>(result)[p * D_packs + j] = vllm::downcast<P>(acc);
    }
    __syncthreads();
  }
  vllm::barrier_at_end<ngpus, true>(sg, self_sg, rank);
}

// ------------------------------------------------------------ bench variants
// g_variant selects the kernel the entry points launch:
//   gather_cat: 0 original; 1xU (U = 1..16 loads in flight per thread) with
//     32-bit index math; 2xU the same with 64-bit index math.
//   lse_rs: 0 original; 1 one warp per (token, local head) pair, every
//     rank's packs loaded before the LSE math (as lse_combine_kernel).
static int g_gc_variant = 0;
static int g_lse_variant = 0;
// > 0: launch the variant kernels on this many blocks; blocks past
// kMaxBlocks skip the barriers (timing probe only, not safe).
static int g_blocks = 0;
static int g_interleave = 0;
void set_interleave(int64_t v) { g_interleave = static_cast<int>(v); }
void set_blocks(int64_t b) { g_blocks = static_cast<int>(b); }
void set_variant(int64_t gc, int64_t lse) {
  g_gc_variant = static_cast<int>(gc);
  g_lse_variant = static_cast<int>(lse);
}

template <int ngpus, int U, typename I>
__global__ void __launch_bounds__(kThreads, 1)
    gather_cat_u_kernel(RankData* _da, RankData* _db, int64_t off_a,
                        int64_t off_b, RankSignals sg, Signal* self_sg,
                        uint4* __restrict__ out, int rank, int64_t T_,
                        int64_t H_, int64_t a_row_, int64_t sa_t, int64_t sa_h,
                        int64_t b_row_, int64_t sb_t, int64_t sb_h,
                        int il) {
  auto da = *_da;
  auto db = *_db;
  if (blockIdx.x < vllm::kMaxBlocks)
    vllm::barrier_at_start<ngpus>(sg, self_sg, rank);
  const I H = H_, a_row = a_row_;
  const I row = a_row_ + b_row_;
  const I per_rank = T_ * H_ * row;
  // il: ranks interleaved per 32 units, so the grid reads every peer at once
  const I per_rank_r = il ? (per_rank + 31) / 32 * 32 : per_rank;
  const I total = ngpus * per_rank_r;
  const I stride = gridDim.x * blockDim.x;
  for (I i0 = blockIdx.x * blockDim.x + threadIdx.x; i0 < total;
       i0 += U * stride) {
    uint4 v[U];
    I dst[U];
#pragma unroll
    for (int k = 0; k < U; ++k) {
      const I i = i0 + k * stride;
      I r, rem;
      if (il) {
        const I w = i / 32;
        r = w % ngpus;
        rem = (w / ngpus) * 32 + (i % 32);
      } else {
        r = i / per_rank;
        rem = i - r * per_rank;
      }
      if (i < total && rem < per_rank) {
        const I th = rem / row, j = rem - th * row;
        const I t = th / H, h = th - t * H;
        const uint4* src =
            j < a_row ? reinterpret_cast<const uint4*>(
                            reinterpret_cast<const char*>(da.ptrs[r]) + off_a) +
                            (t * sa_t + h * sa_h + j)
                      : reinterpret_cast<const uint4*>(
                            reinterpret_cast<const char*>(db.ptrs[r]) + off_b) +
                            (t * sb_t + h * sb_h + (j - a_row));
        v[k] = *src;
        dst[k] = (t * ngpus * H + r * H + h) * row + j;
      }
    }
#pragma unroll
    for (int k = 0; k < U; ++k) {
      const I i = i0 + k * stride;
      const I rem = il ? ((i / 32) / ngpus) * 32 + (i % 32) : 0;
      if (i < total && rem < per_rank) out[dst[k]] = v[k];
    }
  }
  if (blockIdx.x < vllm::kMaxBlocks)
    vllm::barrier_at_end<ngpus, true>(sg, self_sg, rank);
}

// One warp per (token, local head) pair, PAIRS pairs per warp per iteration:
// lanes load every rank's packs of the rows first, the LSEs ride a shuffle.
// Same arithmetic and order as lse_reduce_scatter_kernel.
template <typename T, int ngpus, bool base_e, int D_PACKS, int PAIRS>
__global__ void __launch_bounds__(kThreads, 1)
    lse_rs_warp_kernel(RankData* _dout, RankData* _dlse, int64_t off_out,
                       int64_t off_lse, RankSignals sg, Signal* self_sg,
                       T* __restrict__ result, int rank, int64_t n_pairs,
                       int64_t H, int64_t H_local, int64_t lse_st,
                       int64_t lse_sh) {
  using P = typename vllm::packed_t<T>::P;
  using A = typename vllm::packed_t<T>::A;
  constexpr int kPack = 16 / sizeof(T);
  constexpr int PER_LANE = D_PACKS / 32;
  static_assert(D_PACKS % 32 == 0, "whole packs per lane");
  auto dout = *_dout;
  auto dlse = *_dlse;
  const int lane = threadIdx.x % 32;
  const int64_t warps = int64_t(gridDim.x) * (kThreads / 32);
  if (blockIdx.x < vllm::kMaxBlocks)
    vllm::barrier_at_start<ngpus>(sg, self_sg, rank);
  for (int64_t p0 = (blockIdx.x * (kThreads / 32) + threadIdx.x / 32) * PAIRS;
       p0 < n_pairs; p0 += warps * PAIRS) {
    P x[PAIRS][ngpus][PER_LANE];
    float mine[PAIRS];
#pragma unroll
    for (int q = 0; q < PAIRS; ++q) {
      const int64_t p = p0 + q;
      if (p < n_pairs) {
        const int64_t t = p / H_local;
        const int64_t h = rank * H_local + (p - t * H_local);
        const int64_t src = (t * H + h) * D_PACKS;
#pragma unroll
        for (int r = 0; r < ngpus; ++r)
#pragma unroll
          for (int i = 0; i < PER_LANE; ++i)
            x[q][r][i] = reinterpret_cast<const P*>(
                reinterpret_cast<const char*>(dout.ptrs[r]) +
                off_out)[src + i * 32 + lane];
        mine[q] = 0.f;
        if (lane < ngpus)
          mine[q] = reinterpret_cast<const float*>(
              reinterpret_cast<const char*>(dlse.ptrs[lane]) +
              off_lse)[t * lse_st + h * lse_sh];
      }
    }
#pragma unroll
    for (int q = 0; q < PAIRS; ++q) {
      const int64_t p = p0 + q;
      if (p >= n_pairs) break;
      float raw[ngpus], l[ngpus];
      float m = -INFINITY;
#pragma unroll
      for (int r = 0; r < ngpus; ++r) {
        raw[r] = __shfl_sync(0xffffffffu, mine[q], r);
        l[r] = nan_inf_to_neg_inf(raw[r]);
        m = fmaxf(m, l[r]);
      }
      if (m == -INFINITY) m = 0.f;
      float e[ngpus];
#pragma unroll
      for (int r = 0; r < ngpus; ++r) e[r] = cp_exp<base_e>(l[r] - m);
#pragma unroll
      for (int stride = ngpus / 2; stride >= 1; stride /= 2)
#pragma unroll
        for (int r = 0; r < stride; ++r) e[r] += e[r + stride];
      const float lse = cp_log<base_e>(e[0]) + m;
      float f[ngpus];
#pragma unroll
      for (int r = 0; r < ngpus; ++r)
        f[r] = cp_exp<base_e>(nan_inf_to_neg_inf(raw[r] - lse));
#pragma unroll
      for (int i = 0; i < PER_LANE; ++i) {
        A acc;
#pragma unroll
        for (int r = 0; r < ngpus; ++r) {
          A v = vllm::upcast(x[q][r][i]);
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
  }
  if (blockIdx.x < vllm::kMaxBlocks)
    vllm::barrier_at_end<ngpus, true>(sg, self_sg, rank);
}

// The input's rank data: registered (captured, or a buffer registered at
// init) exactly as CustomAllreduce::allreduce resolves it.
static RankData* rank_data(CustomAllreduce* fa, void* input,
                           cudaStream_t stream) {
  cudaStreamCaptureStatus status;
  CUDACHECK(cudaStreamIsCapturing(stream, &status));
  if (status == cudaStreamCaptureStatusActive) {
    fa->check_rank_data_capacity(fa->graph_unreg_buffers_.size() + 1);
    RankData* ptrs = fa->d_rank_data_base_ + fa->graph_unreg_buffers_.size();
    fa->graph_unreg_buffers_.push_back(input);
    return ptrs;
  }
  auto it = fa->buffers_.find(input);
  TORCH_CHECK(it != fa->buffers_.end(),
              "one-shot collective input is not registered");
  return it->second;
}

static void* stage(const torch::Tensor& inp, int64_t reg_buffer,
                   int64_t reg_bytes, cudaStream_t stream) {
  if (!reg_buffer) return inp.data_ptr();
  const int64_t bytes = inp.numel() * inp.element_size();
  TORCH_CHECK(bytes <= reg_bytes,
              "one-shot collective input exceeds the buffer");
  CUDACHECK(cudaMemcpyAsync(reinterpret_cast<void*>(reg_buffer), inp.data_ptr(),
                            bytes, cudaMemcpyDeviceToDevice, stream));
  return reinterpret_cast<void*>(reg_buffer);
}

static int blocks_for(int64_t work) {
  return static_cast<int>(
      std::min<int64_t>(vllm::kMaxBlocks, (work + kThreads - 1) / kThreads));
}

// inp [outer, inner_bytes] per rank -> out [outer, world * inner_bytes]
void all_gather(int64_t fa_ptr, torch::Tensor inp, torch::Tensor out,
                int64_t outer, int64_t inner_bytes, int64_t reg_buffer,
                int64_t reg_bytes) {
  auto* fa = reinterpret_cast<CustomAllreduce*>(fa_ptr);
  TORCH_CHECK(inp.is_contiguous() && out.is_contiguous(),
              "contiguous tensors only");
  TORCH_CHECK(inner_bytes % 16 == 0,
              "the gathered block must be a multiple of 16 bytes");
  TORCH_CHECK(
      out.numel() * out.element_size() == fa->world_size_ * outer * inner_bytes,
      "output size mismatch");
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  RankData* ptrs =
      rank_data(fa, stage(inp, reg_buffer, reg_bytes, stream), stream);
  const int64_t inner = inner_bytes / 16;
  const int blocks = blocks_for(fa->world_size_ * outer * inner);
  auto* o = reinterpret_cast<uint4*>(out.data_ptr());
  switch (fa->world_size_) {
    case 2:
      gather_kernel<2><<<blocks, kThreads, 0, stream>>>(
          ptrs, fa->sg_, fa->self_sg_, o, fa->rank_, outer, inner);
      break;
    case 4:
      gather_kernel<4><<<blocks, kThreads, 0, stream>>>(
          ptrs, fa->sg_, fa->self_sg_, o, fa->rank_, outer, inner);
      break;
    case 8:
      gather_kernel<8><<<blocks, kThreads, 0, stream>>>(
          ptrs, fa->sg_, fa->self_sg_, o, fa->rank_, outer, inner);
      break;
    default:
      TORCH_CHECK(false, "one-shot collectives take 2, 4 or 8 GPUs");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename T>
static void launch_rs(CustomAllreduce* fa, RankData* ptrs, T* out,
                      int64_t outer, int64_t inner, cudaStream_t stream) {
  const int blocks = blocks_for(outer * inner);
  switch (fa->world_size_) {
    case 2:
      reduce_scatter_kernel<T, 2><<<blocks, kThreads, 0, stream>>>(
          ptrs, fa->sg_, fa->self_sg_, out, fa->rank_, outer, inner);
      break;
    case 4:
      reduce_scatter_kernel<T, 4><<<blocks, kThreads, 0, stream>>>(
          ptrs, fa->sg_, fa->self_sg_, out, fa->rank_, outer, inner);
      break;
    case 8:
      reduce_scatter_kernel<T, 8><<<blocks, kThreads, 0, stream>>>(
          ptrs, fa->sg_, fa->self_sg_, out, fa->rank_, outer, inner);
      break;
    default:
      TORCH_CHECK(false, "one-shot collectives take 2, 4 or 8 GPUs");
  }
}

// inp [outer, world * inner] -> out [outer, inner], summed over ranks
void reduce_scatter(int64_t fa_ptr, torch::Tensor inp, torch::Tensor out,
                    int64_t outer, int64_t inner, int64_t reg_buffer,
                    int64_t reg_bytes) {
  auto* fa = reinterpret_cast<CustomAllreduce*>(fa_ptr);
  TORCH_CHECK(inp.is_contiguous() && out.is_contiguous(),
              "contiguous tensors only");
  TORCH_CHECK(inp.scalar_type() == out.scalar_type(), "dtype mismatch");
  TORCH_CHECK((inner * inp.element_size()) % 16 == 0,
              "each rank's block must be a multiple of 16 bytes");
  TORCH_CHECK(inp.numel() == fa->world_size_ * outer * inner &&
                  out.numel() == outer * inner,
              "size mismatch");
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  RankData* ptrs =
      rank_data(fa, stage(inp, reg_buffer, reg_bytes, stream), stream);
  const int64_t pack = 16 / inp.element_size();
  switch (inp.scalar_type()) {
    case at::ScalarType::BFloat16:
      launch_rs(fa, ptrs, reinterpret_cast<nv_bfloat16*>(out.data_ptr()), outer,
                inner / pack, stream);
      break;
    case at::ScalarType::Half:
      launch_rs(fa, ptrs, reinterpret_cast<half*>(out.data_ptr()), outer,
                inner / pack, stream);
      break;
    case at::ScalarType::Float:
      launch_rs(fa, ptrs, reinterpret_cast<float*>(out.data_ptr()), outer,
                inner / pack, stream);
      break;
    default:
      TORCH_CHECK(false, "one-shot reduce-scatter takes bf16, fp16 or fp32");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Two inputs staged into the registered buffer (eager) or registered as
// captured inputs; returns their rank data and byte offsets.
static void stage_two(CustomAllreduce* fa, const torch::Tensor& a,
                      const torch::Tensor& b, int64_t reg_buffer,
                      int64_t reg_bytes, cudaStream_t stream, RankData** da,
                      RankData** db, int64_t* off_a, int64_t* off_b) {
  if (!reg_buffer) {
    *da = rank_data(fa, a.data_ptr(), stream);
    *db = rank_data(fa, b.data_ptr(), stream);
    *off_a = *off_b = 0;
    return;
  }
  const int64_t a_bytes = a.numel() * a.element_size();
  const int64_t b_bytes = b.numel() * b.element_size();
  *off_a = 0;
  *off_b = (a_bytes + 255) / 256 * 256;
  TORCH_CHECK(*off_b + b_bytes <= reg_bytes,
              "one-shot collective inputs exceed the buffer");
  auto* base = reinterpret_cast<char*>(reg_buffer);
  CUDACHECK(cudaMemcpyAsync(base, a.data_ptr(), a_bytes,
                            cudaMemcpyDeviceToDevice, stream));
  CUDACHECK(cudaMemcpyAsync(base + *off_b, b.data_ptr(), b_bytes,
                            cudaMemcpyDeviceToDevice, stream));
  *da = *db = rank_data(fa, base, stream);
}

template <int N, typename I>
static void launch_gc_u_i(int U, int blocks, cudaStream_t stream, RankData* da,
                          RankData* db, int64_t off_a, int64_t off_b,
                          CustomAllreduce* fa, uint4* o, int64_t T, int64_t H,
                          int64_t a_row, int64_t sa_t, int64_t sa_h,
                          int64_t b_row, int64_t sb_t, int64_t sb_h) {
#define GCU(UU)                                                            \
  case UU:                                                                 \
    gather_cat_u_kernel<N, UU, I><<<blocks, kThreads, 0, stream>>>(        \
        da, db, off_a, off_b, fa->sg_, fa->self_sg_, o, fa->rank_, T, H,   \
        a_row, sa_t, sa_h, b_row, sb_t, sb_h, g_interleave);               \
    break;
  switch (U) {
    GCU(1) GCU(2) GCU(4) GCU(8) GCU(16)
    default:
      TORCH_CHECK(false, "unroll must be 1, 2, 4, 8 or 16");
  }
#undef GCU
}

template <int N>
static void launch_gc_u(int variant, int blocks, cudaStream_t stream,
                        RankData* da, RankData* db, int64_t off_a,
                        int64_t off_b, CustomAllreduce* fa, uint4* o,
                        int64_t T, int64_t H, int64_t a_row, int64_t sa_t,
                        int64_t sa_h, int64_t b_row, int64_t sb_t,
                        int64_t sb_h) {
  TORCH_CHECK(N * T * H * (a_row + b_row) < (int64_t(1) << 31), "too large");
  if (g_blocks > 0) blocks = g_blocks;
  if (variant > 0)
    launch_gc_u_i<N, uint32_t>(variant, blocks, stream, da, db, off_a, off_b,
                               fa, o, T, H, a_row, sa_t, sa_h, b_row, sb_t,
                               sb_h);
  else
    launch_gc_u_i<N, int64_t>(-variant, blocks, stream, da, db, off_a, off_b,
                              fa, o, T, H, a_row, sa_t, sa_h, b_row, sb_t,
                              sb_h);
}

// a [T, H, Da], b [T, H, Db] (rows contiguous; t / h strides in elements,
// 16-byte multiples) -> out [T, world * H, Da + Db]. With reg_buffer set
// (eager) a and b must be contiguous.
void all_gather_cat(int64_t fa_ptr, torch::Tensor a, torch::Tensor b,
                    torch::Tensor out, int64_t reg_buffer, int64_t reg_bytes) {
  auto* fa = reinterpret_cast<CustomAllreduce*>(fa_ptr);
  TORCH_CHECK(a.dim() == 3 && b.dim() == 3 && a.size(0) == b.size(0) &&
                  a.size(1) == b.size(1),
              "a and b must be [T, H, *] with matching T and H");
  TORCH_CHECK(a.scalar_type() == b.scalar_type() &&
                  a.scalar_type() == out.scalar_type(),
              "dtype mismatch");
  TORCH_CHECK(a.stride(2) == 1 && b.stride(2) == 1 && out.is_contiguous(),
              "rows must be contiguous");
  TORCH_CHECK(!reg_buffer || (a.is_contiguous() && b.is_contiguous()),
              "eager inputs must be contiguous");
  const int64_t es = a.element_size();
  for (int64_t v : {a.size(2) * es, b.size(2) * es, a.stride(0) * es,
                    a.stride(1) * es, b.stride(0) * es, b.stride(1) * es})
    TORCH_CHECK(v % 16 == 0, "rows and strides must be 16-byte multiples");
  const int64_t T = a.size(0), H = a.size(1);
  TORCH_CHECK(out.numel() == T * fa->world_size_ * H * (a.size(2) + b.size(2)),
              "output size mismatch");
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  RankData *da, *db;
  int64_t off_a, off_b;
  stage_two(fa, a, b, reg_buffer, reg_bytes, stream, &da, &db, &off_a, &off_b);
  const int64_t u = 16 / es;
  const int64_t a_row = a.size(2) / u, b_row = b.size(2) / u;
  const int64_t sa_t = a.stride(0) / u, sa_h = a.stride(1) / u;
  const int64_t sb_t = b.stride(0) / u, sb_h = b.stride(1) / u;
  const int blocks = blocks_for(fa->world_size_ * T * H * (a_row + b_row));
  auto* o = reinterpret_cast<uint4*>(out.data_ptr());
#define GATHER_CAT(N)                                                         \
  if (g_gc_variant != 0) {                                                    \
    launch_gc_u<N>(g_gc_variant, blocks, stream, da, db, off_a, off_b, fa, o, \
                   T, H, a_row, sa_t, sa_h, b_row, sb_t, sb_h);               \
  } else                                                                      \
    gather_cat_kernel<N><<<blocks, kThreads, 0, stream>>>(                    \
        da, db, off_a, off_b, fa->sg_, fa->self_sg_, o, fa->rank_, T, H,      \
        a_row, sa_t, sa_h, b_row, sb_t, sb_h)
  switch (fa->world_size_) {
    case 2:
      GATHER_CAT(2);
      break;
    case 4:
      GATHER_CAT(4);
      break;
    case 8:
      GATHER_CAT(8);
      break;
    default:
      TORCH_CHECK(false, "one-shot collectives take 2, 4 or 8 GPUs");
  }
#undef GATHER_CAT
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename T, int ngpus>
static void launch_lse_rs(CustomAllreduce* fa, RankData* dout, RankData* dlse,
                          int64_t off_out, int64_t off_lse, T* result,
                          int64_t n_pairs, int64_t H, int64_t H_local,
                          int64_t D_packs, int64_t lse_st, int64_t lse_sh,
                          bool base_e, cudaStream_t stream) {
  if (g_lse_variant != 0) {
    TORCH_CHECK(D_packs == 64, "warp variant takes 64 packs");
    const int pairs = g_lse_variant;
    const int64_t wpb = kThreads / 32;
    const int blocks = static_cast<int>(std::min<int64_t>(
        vllm::kMaxBlocks, (n_pairs + wpb * pairs - 1) / (wpb * pairs)));
    const int nb = g_blocks > 0 ? g_blocks : blocks;
#define LW(BE, PP)                                                           \
  lse_rs_warp_kernel<T, ngpus, BE, 64, PP><<<nb, kThreads, 0, stream>>>( \
      dout, dlse, off_out, off_lse, fa->sg_, fa->self_sg_, result, fa->rank_, \
      n_pairs, H, H_local, lse_st, lse_sh)
    if (pairs == 1) {
      if (base_e) LW(true, 1); else LW(false, 1);
    } else if (pairs == 2) {
      if (base_e) LW(true, 2); else LW(false, 2);
    } else {
      TORCH_CHECK(false, "pairs must be 1 or 2");
    }
#undef LW
    return;
  }
  const int64_t ppb = kThreads / D_packs;
  const int blocks = static_cast<int>(
      std::min<int64_t>(vllm::kMaxBlocks, (n_pairs + ppb - 1) / ppb));
  if (base_e) {
    lse_reduce_scatter_kernel<T, ngpus, true><<<blocks, kThreads, 0, stream>>>(
        dout, dlse, off_out, off_lse, fa->sg_, fa->self_sg_, result, fa->rank_,
        n_pairs, H, H_local, D_packs, lse_st, lse_sh);
  } else {
    lse_reduce_scatter_kernel<T, ngpus, false><<<blocks, kThreads, 0, stream>>>(
        dout, dlse, off_out, off_lse, fa->sg_, fa->self_sg_, result, fa->rank_,
        n_pairs, H, H_local, D_packs, lse_st, lse_sh);
  }
}

// out [T, H, D] and lse [T, H] (fp32, any non-negative strides, e.g. the
// transposed view of a kernel's [H, T] LSE; contiguous when staged) per rank
// -> result [T, H / world, D].
void lse_reduce_scatter(int64_t fa_ptr, torch::Tensor out, torch::Tensor lse,
                        torch::Tensor result, bool base_e, int64_t reg_buffer,
                        int64_t reg_bytes) {
  auto* fa = reinterpret_cast<CustomAllreduce*>(fa_ptr);
  const int world = fa->world_size_;
  TORCH_CHECK(out.is_contiguous() && result.is_contiguous(),
              "contiguous out and result only");
  TORCH_CHECK(!reg_buffer || lse.is_contiguous(),
              "eager (staged) lse must be contiguous");
  TORCH_CHECK(out.dim() == 3 && lse.dim() == 2 && lse.size(0) == out.size(0) &&
                  lse.size(1) == out.size(1),
              "out must be [T, H, D] and lse [T, H]");
  TORCH_CHECK(lse.stride(0) >= 0 && lse.stride(1) >= 0, "negative lse stride");
  TORCH_CHECK(lse.scalar_type() == at::ScalarType::Float, "lse must be fp32");
  TORCH_CHECK(out.scalar_type() == result.scalar_type(), "dtype mismatch");
  const int64_t T = out.size(0), H = out.size(1), D = out.size(2);
  TORCH_CHECK(H % world == 0, "heads must split evenly over ranks");
  const int64_t pack = 16 / out.element_size();
  TORCH_CHECK(D % pack == 0, "head dim must be a 16-byte multiple");
  const int64_t D_packs = D / pack;
  TORCH_CHECK(D_packs <= kThreads && kThreads % D_packs == 0 &&
                  kThreads / D_packs <= kThreads / 8,
              "unsupported head dim");
  const int64_t H_local = H / world;
  TORCH_CHECK(result.numel() == T * H_local * D, "result size mismatch");
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  RankData *dout, *dlse;
  int64_t off_out, off_lse;
  stage_two(fa, out, lse, reg_buffer, reg_bytes, stream, &dout, &dlse, &off_out,
            &off_lse);
  const int64_t n_pairs = T * H_local;
#define LSE_RS(TYPE, N)                                                       \
  launch_lse_rs<TYPE, N>(fa, dout, dlse, off_out, off_lse,                    \
                         reinterpret_cast<TYPE*>(result.data_ptr()), n_pairs, \
                         H, H_local, D_packs, lse.stride(0), lse.stride(1),   \
                         base_e, stream)
#define LSE_RS_WORLD(TYPE)                                            \
  switch (world) {                                                    \
    case 2:                                                           \
      LSE_RS(TYPE, 2);                                                \
      break;                                                          \
    case 4:                                                           \
      LSE_RS(TYPE, 4);                                                \
      break;                                                          \
    case 8:                                                           \
      LSE_RS(TYPE, 8);                                                \
      break;                                                          \
    default:                                                          \
      TORCH_CHECK(false, "one-shot collectives take 2, 4 or 8 GPUs"); \
  }
  switch (out.scalar_type()) {
    case at::ScalarType::BFloat16:
      LSE_RS_WORLD(nv_bfloat16);
      break;
    case at::ScalarType::Half:
      LSE_RS_WORLD(half);
      break;
    default:
      TORCH_CHECK(false, "one-shot LSE reduce-scatter takes bf16 or fp16");
  }
#undef LSE_RS_WORLD
#undef LSE_RS
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---------------------------------------------------------------- prefill
// The same combine at prefill size (an out of hundreds of MB): out and lse sit
// in a buffer registered at init (the attention writes out there directly, so
// nothing is staged), and the kernel runs on the whole GPU. Its peers are
// synced by one-block barrier launches before and after it, since only
// kMaxBlocks blocks have barrier flags. One warp per (token, local head) pair:
// lanes load all ranks' 16 B packs of the row before the LSE math, so every
// load is in flight at once. result is head-major [H / ngpus, T, D].
template <int ngpus, bool end>
__global__ void barrier_kernel(RankSignals sg, Signal* self_sg, int rank) {
  if constexpr (end)
    vllm::barrier_at_end<ngpus, true>(sg, self_sg, rank);
  else
    vllm::barrier_at_start<ngpus>(sg, self_sg, rank);
}

constexpr int kCombineThreads = 256;

template <typename T, int ngpus, bool base_e, int D_PACKS>
__global__ void __launch_bounds__(kCombineThreads)
    lse_combine_kernel(RankData* _dbuf, int64_t off_lse, T* __restrict__ result,
                       int rank, int64_t T_, int64_t H, int64_t H_local) {
  using P = typename vllm::packed_t<T>::P;
  using A = typename vllm::packed_t<T>::A;
  constexpr int kPack = 16 / sizeof(T);
  constexpr int PER_LANE = D_PACKS / 32;
  static_assert(D_PACKS % 32 == 0, "whole packs per lane");
  auto dbuf = *_dbuf;
  const int lane = threadIdx.x % 32;
  const int64_t warps = int64_t(gridDim.x) * (kCombineThreads / 32);
  const int64_t n_pairs = T_ * H_local;
  for (int64_t p = blockIdx.x * (kCombineThreads / 32) + threadIdx.x / 32;
       p < n_pairs; p += warps) {
    // heads fastest: a token's local heads are one contiguous span per rank
    const int64_t t = p / H_local, hl = p - t * H_local;
    const int64_t h = rank * H_local + hl;
    const int64_t src = (t * H + h) * D_PACKS;
    P x[ngpus][PER_LANE];
#pragma unroll
    for (int r = 0; r < ngpus; ++r)
#pragma unroll
      for (int i = 0; i < PER_LANE; ++i)
        x[r][i] = reinterpret_cast<const P*>(dbuf.ptrs[r])[src + i * 32 + lane];
    float mine = 0.f;
    if (lane < ngpus)
      mine = reinterpret_cast<const float*>(
          reinterpret_cast<const char*>(dbuf.ptrs[lane]) + off_lse)[t * H + h];
    float raw[ngpus], l[ngpus];
    float m = -INFINITY;
#pragma unroll
    for (int r = 0; r < ngpus; ++r) {
      raw[r] = __shfl_sync(0xffffffffu, mine, r);
      l[r] = nan_inf_to_neg_inf(raw[r]);
      m = fmaxf(m, l[r]);
    }
    if (m == -INFINITY) m = 0.f;
    float e[ngpus];
#pragma unroll
    for (int r = 0; r < ngpus; ++r) e[r] = cp_exp<base_e>(l[r] - m);
#pragma unroll
    for (int stride = ngpus / 2; stride >= 1; stride /= 2)
#pragma unroll
      for (int r = 0; r < stride; ++r) e[r] += e[r + stride];
    const float lse = cp_log<base_e>(e[0]) + m;
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
      reinterpret_cast<P*>(result)[(hl * T_ + t) * D_PACKS + i * 32 + lane] =
          vllm::downcast<P>(acc);
    }
  }
}

template <typename T, int ngpus>
static void launch_combine(CustomAllreduce* fa, RankData* dbuf, int64_t off_lse,
                           T* result, int64_t T_, int64_t H, int64_t D_packs,
                           bool base_e, cudaStream_t stream) {
  static int sms = 0;
  if (sms == 0) {
    int dev;
    CUDACHECK(cudaGetDevice(&dev));
    CUDACHECK(
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
  }
  const int64_t H_local = H / ngpus;
  const int64_t wpb = kCombineThreads / 32;
  const int blocks = static_cast<int>(
      std::min<int64_t>(sms * 8, (T_ * H_local + wpb - 1) / wpb));
  barrier_kernel<ngpus, false>
      <<<1, 32, 0, stream>>>(fa->sg_, fa->self_sg_, fa->rank_);
  TORCH_CHECK(D_packs == 64, "prefill combine takes a 512-wide bf16 head dim");
  if (base_e)
    lse_combine_kernel<T, ngpus, true, 64>
        <<<blocks, kCombineThreads, 0, stream>>>(dbuf, off_lse, result,
                                                 fa->rank_, T_, H, H_local);
  else
    lse_combine_kernel<T, ngpus, false, 64>
        <<<blocks, kCombineThreads, 0, stream>>>(dbuf, off_lse, result,
                                                 fa->rank_, T_, H, H_local);
  barrier_kernel<ngpus, true>
      <<<1, 32, 0, stream>>>(fa->sg_, fa->self_sg_, fa->rank_);
}

// buf: a registered buffer holding this rank's out [T, H, D] at offset 0 and
// lse [T, H] (fp32, contiguous) at off_lse -> result, head-major [H / world,
// T, D]. Eager only.
void lse_combine(int64_t fa_ptr, int64_t buf, int64_t off_lse, int64_t T_,
                 int64_t H, int64_t D, torch::Tensor result, bool base_e) {
  auto* fa = reinterpret_cast<CustomAllreduce*>(fa_ptr);
  const int world = fa->world_size_;
  TORCH_CHECK(result.is_contiguous() && result.numel() == T_ * H / world * D,
              "result must be contiguous [H / world, T, D]");
  TORCH_CHECK(H % world == 0, "heads must split evenly over ranks");
  const int64_t D_packs = D * result.element_size() / 16;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto it = fa->buffers_.find(reinterpret_cast<void*>(buf));
  TORCH_CHECK(it != fa->buffers_.end(), "prefill buffer is not registered");
  RankData* dbuf = it->second;
#define COMBINE(TYPE, N)                                                     \
  launch_combine<TYPE, N>(fa, dbuf, off_lse,                                 \
                          reinterpret_cast<TYPE*>(result.data_ptr()), T_, H, \
                          D_packs, base_e, stream)
#define COMBINE_WORLD(TYPE)                                           \
  switch (world) {                                                    \
    case 2:                                                           \
      COMBINE(TYPE, 2);                                               \
      break;                                                          \
    case 4:                                                           \
      COMBINE(TYPE, 4);                                               \
      break;                                                          \
    case 8:                                                           \
      COMBINE(TYPE, 8);                                               \
      break;                                                          \
    default:                                                          \
      TORCH_CHECK(false, "one-shot collectives take 2, 4 or 8 GPUs"); \
  }
  switch (result.scalar_type()) {
    case at::ScalarType::BFloat16:
      COMBINE_WORLD(nv_bfloat16);
      break;
    case at::ScalarType::Half:
      COMBINE_WORLD(half);
      break;
    default:
      TORCH_CHECK(false, "prefill combine takes bf16 or fp16");
  }
#undef COMBINE_WORLD
#undef COMBINE
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

#include "push.cuh"

// A uint8 tensor over nbytes of device memory at ptr (not owned).
torch::Tensor view(int64_t ptr, int64_t nbytes) {
  int dev;
  CUDACHECK(cudaGetDevice(&dev));
  return torch::from_blob(
      reinterpret_cast<void*>(ptr), {nbytes},
      torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, dev));
}

}  // namespace one_shot

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("all_gather", &one_shot::all_gather, "One-shot IPC all-gather");
  m.def("reduce_scatter", &one_shot::reduce_scatter,
        "One-shot IPC reduce-scatter");
  m.def("all_gather_cat", &one_shot::all_gather_cat,
        "One-shot IPC all-gather of two inputs, concatenated per row");
  m.def("lse_reduce_scatter", &one_shot::lse_reduce_scatter,
        "One-shot IPC LSE-weighted reduce-scatter (DCP attention combine)");
  m.def("lse_combine", &one_shot::lse_combine,
        "Prefill-size DCP attention combine over a registered buffer");
  m.def("view", &one_shot::view, "uint8 tensor over raw device memory");
  m.def("push_setup", &one_shot::push_setup, "push: buffers");
  m.def("push_buffer_bytes", &one_shot::push_buffer_bytes, "push: size");
  m.def("push_config", &one_shot::push_config, "push: blocks, unroll");
  m.def("push_all_gather_cat", &one_shot::push_all_gather_cat, "push gc");
  m.def("push_lse_reduce_scatter", &one_shot::push_lse_reduce_scatter,
        "push lse rs");
  m.def("set_interleave", &one_shot::set_interleave, "bench");
  m.def("set_blocks", &one_shot::set_blocks, "bench: block override");
  m.def("set_variant", &one_shot::set_variant, "bench: kernel variants");
}

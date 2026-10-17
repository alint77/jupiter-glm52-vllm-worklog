// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// y[M, N] = x[M, K] @ W[N, K]^T for bf16 at decode sizes (M <= 8) on sm_90a.
//
// At a handful of tokens a linear layer is a weight read: the job is to stream
// W at HBM speed on every SM. Library GEMMs at M = 8 pick 64x8 tiles, split-K
// with a separate reduce kernel, or under-fill the GPU for small N; this
// kernel spreads (128-row tile, 128-wide K chunk) units evenly over all SMs
// (stream-K). A producer warp feeds a TMA ring (128-byte swizzle, so the
// ldmatrix reads below are bank-conflict free); eight consumer warps each own
// 16 rows of the tile and run mma.m16n8k16 with W as A and the tokens as B.
//
// A CTA that owns a whole tile writes bf16 directly. Otherwise it adds its
// fp32 partial into a workspace and counts its chunks into the tile's
// counter; the CTA that completes the count converts the tile, zeroes the
// workspace rows and resets the counter, so every call finds them clean and
// there is no second launch.

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>
#include <cstdint>

#ifndef MUT
#define MUT 0
#endif
namespace skinny {

constexpr int HALF_K = 64;  // one 128-byte swizzled box width, in bf16
constexpr int MAX_M = 8;
constexpr int GRID = 132;
constexpr int SMEM_LIMIT = 227 * 1024;

// TILE_N rows per tile (16 per consumer warp), CHUNK_K columns per unit
template <int TILE_N, int CHUNK_K>
struct Cfg {
  static constexpr int CONSUMERS = TILE_N / 16;
  static constexpr int THREADS = (CONSUMERS + 1) * 32;
  static constexpr int BOXES = CHUNK_K / HALF_K;
  static constexpr int W_BYTES = TILE_N * CHUNK_K * 2;
  static constexpr int X_BYTES = MAX_M * CHUNK_K * 2;
  static constexpr int STAGE_BYTES = W_BYTES + X_BYTES;
  static constexpr int STAGES = std::min(8, (SMEM_LIMIT - 2048) / STAGE_BYTES);
  static constexpr int SMEM_BYTES = 1024 + STAGES * STAGE_BYTES;  // 1 KB swizzle align
};

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(count));
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(bar)),
               "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
  asm volatile(
      "{\n .reg .pred p;\n WAIT_%=:\n mbarrier.try_wait.parity.shared::cta.b64 "
      "p, [%0], %1;\n @!p bra WAIT_%=;\n}\n" ::"r"(smem_u32(bar)), "r"(parity)
      : "memory");
}
__device__ __forceinline__ void tma_2d(void* dst, const CUtensorMap* map, int x, int y,
                                       uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3}], [%4];" ::"r"(smem_u32(dst)),
      "l"(reinterpret_cast<uint64_t>(map)), "r"(x), "r"(y), "r"(smem_u32(bar))
      : "memory");
}
__device__ __forceinline__ void ldmatrix_x4(uint32_t* r, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
__device__ __forceinline__ void ldmatrix_x2(uint32_t* r, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
               : "=r"(r[0]), "=r"(r[1]) : "r"(addr));
}
__device__ __forceinline__ void mma(float* d, const uint32_t* a, const uint32_t* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
// Byte offset of (row, 16-byte column c) in a 128-byte-wide box written with
// CU_TENSOR_MAP_SWIZZLE_128B: the column is XORed with the row within 8.
__device__ __forceinline__ uint32_t swz(int row, int c) {
  return row * 128 + ((c ^ (row & 7)) << 4);
}

struct Params {
  int M, N, K, tiles, chunks;
  __nv_bfloat16* out;
  float* ws;      // [MAX_M, N] fp32, zero between calls
  int* counters;  // [tiles], zero between calls
};

template <int TILE_N, int CHUNK_K>
__global__ void __launch_bounds__(Cfg<TILE_N, CHUNK_K>::THREADS, 1)
    skinny_kernel(const __grid_constant__ CUtensorMap wmap,
                  const __grid_constant__ CUtensorMap xmap, const Params p) {
  using C = Cfg<TILE_N, CHUNK_K>;
  constexpr int STAGES = C::STAGES, CONSUMERS = C::CONSUMERS, BOXES = C::BOXES;
  constexpr int W_BYTES = C::W_BYTES, X_BYTES = C::X_BYTES, STAGE_BYTES = C::STAGE_BYTES;
  extern __shared__ __align__(1024) unsigned char raw[];
  unsigned char* ring = reinterpret_cast<unsigned char*>(
      (reinterpret_cast<uintptr_t>(raw) + 1023) & ~uintptr_t(1023));
  __shared__ uint64_t full[STAGES], empty[STAGES];
  __shared__ int last;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int64_t units = static_cast<int64_t>(p.tiles) * p.chunks;
  const int u0 = static_cast<int>(units * blockIdx.x / gridDim.x);
  const int u1 = static_cast<int>(units * (blockIdx.x + 1) / gridDim.x);

  if (threadIdx.x == 0) {
    for (int s = 0; s < STAGES; ++s) {
      mbar_init(&full[s], 1);
      mbar_init(&empty[s], MUT == 9 ? CONSUMERS - 1 : CONSUMERS);
    }
    if (MUT != 11) asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (u0 >= u1) return;

  if (warp == CONSUMERS) {  // producer
    if (lane == 0) {
      asm volatile("prefetch.tensormap [%0];" ::"l"(reinterpret_cast<uint64_t>(&wmap))
                   : "memory");
      asm volatile("prefetch.tensormap [%0];" ::"l"(reinterpret_cast<uint64_t>(&xmap))
                   : "memory");
      for (int u = u0, it = 0; u < u1; ++u, ++it) {
        const int t = u / p.chunks, c = u - t * p.chunks, s = it % STAGES;
        if (it >= STAGES) mbar_wait(&empty[s], (MUT == 8 ? (it / STAGES) : (it / STAGES) - 1) & 1);
        unsigned char* st = ring + (MUT == 12 ? (s + 1) % STAGES : s) * STAGE_BYTES;
        mbar_expect_tx(&full[s], STAGE_BYTES + (MUT == 4 ? -1024 : MUT == 5 ? 1024 : 0));
        for (int h = 0; h < BOXES; ++h) {
          const int k = c * CHUNK_K + h * HALF_K;
          tma_2d(st + h * (W_BYTES / BOXES), &wmap, k, t * TILE_N, &full[s]);
          if (!(MUT == 6 && h == BOXES - 1))
            tma_2d(st + W_BYTES + h * (X_BYTES / BOXES), &xmap, k, 0, &full[s]);
        }
      }
    }
    return;
  }

  // consumer: rows [warp*16, warp*16+16) of the tile
  const int g = lane / 4, tq = lane % 4;
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  int first_chunk = -1;
  for (int u = u0, it = 0; u < u1; ++u, ++it) {
    const int t = u / p.chunks, c = u - t * p.chunks, s = it % STAGES;
    if (first_chunk < 0) first_chunk = c;
    if (!(MUT == 1 && it == 0)) mbar_wait(&full[s], ((it / STAGES) + (MUT == 7)) & 1); if (MUT == 2 && lane == 0) mbar_arrive(&empty[s]);
    const uint32_t w = smem_u32(ring + s * STAGE_BYTES);
    const uint32_t x = w + W_BYTES;
#pragma unroll
    for (int h = 0; h < BOXES; ++h) {
#pragma unroll
      for (int kk = 0; kk < HALF_K / 16; ++kk) {
        // A: rows warp*16 + (lane%16), 16-byte column 2kk + lane/16
        uint32_t a[4], b[2];
        const int arow = warp * 16 + (lane & 15);
        ldmatrix_x4(a, w + h * (W_BYTES / BOXES) + swz(arow, 2 * kk + (lane >> 4)));
        // B: token rows lane%8, 16-byte column 2kk + (lane/8)%2
        const int brow = lane & 7;
        ldmatrix_x2(b, x + h * (X_BYTES / BOXES) + swz(brow, 2 * kk + ((lane >> 3) & 1)));
        mma(acc, a, b);
      }
    }
    if (MUT != 10) __syncwarp();
    if (MUT != 2 && lane == 0) mbar_arrive(&empty[s]);

    const bool tile_end = c == p.chunks - 1 || u == u1 - 1;
    if (!tile_end) continue;
    // flush this CTA's share of tile t: chunks [first_chunk, c]
    const int done = c - first_chunk + 1;
    const bool whole = done == p.chunks;
    const int n0 = t * TILE_N + warp * 16 + g;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int n = n0 + (i >> 1) * 8, m = 2 * tq + (i & 1);
      if (n < p.N && m < p.M) {
        if (whole)
          p.out[static_cast<int64_t>(m) * p.N + n] = __float2bfloat16(acc[i]);
        else
          atomicAdd(&p.ws[static_cast<int64_t>(m) * p.N + n], acc[i]);
      }
      acc[i] = 0.f;
    }
    first_chunk = -1;
    if (whole) continue;
    // count this share in; the CTA completing the tile finishes it
    asm volatile("bar.sync 1, %0;" ::"n"(CONSUMERS * 32));
    if (threadIdx.x == 0) {
      __threadfence();
      last = atomicAdd(&p.counters[t], done) + done == p.chunks;
    }
    if (MUT != 3) asm volatile("bar.sync 1, %0;" ::"n"(CONSUMERS * 32));
    if (last) {
      __threadfence();
      for (int e = threadIdx.x; e < TILE_N * p.M; e += CONSUMERS * 32) {
        const int m = e / TILE_N, n = t * TILE_N + e % TILE_N;
        if (n < p.N) {
          float* src = &p.ws[static_cast<int64_t>(m) * p.N + n];
          p.out[static_cast<int64_t>(m) * p.N + n] = __float2bfloat16(__ldcg(src));
          *src = 0.f;
        }
      }
      if (threadIdx.x == 0) p.counters[t] = 0;
    }
  }
}

static CUtensorMap make_map(const torch::Tensor& t, int box_rows) {
  // t is [rows, K] bf16 row-major; boxes are 64 elements (128 B) x box_rows
  CUtensorMap map;
  const cuuint64_t dims[2] = {static_cast<cuuint64_t>(t.size(1)),
                              static_cast<cuuint64_t>(t.size(0))};
  const cuuint64_t strides[1] = {static_cast<cuuint64_t>(t.stride(0)) * 2};
  const cuuint32_t box[2] = {HALF_K, static_cast<cuuint32_t>(box_rows)};
  const cuuint32_t unit[2] = {1, 1};
  const CUresult r = cuTensorMapEncodeTiled(
      &map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, t.data_ptr(), dims, strides, box, unit,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", static_cast<int>(r));
  return map;
}

template <int TILE_N, int CHUNK_K>
static void launch(Params p, const torch::Tensor& x, const torch::Tensor& w, int grid_cap) {
  using C = Cfg<TILE_N, CHUNK_K>;
  p.tiles = (p.N + TILE_N - 1) / TILE_N;
  p.chunks = (p.K + CHUNK_K - 1) / CHUNK_K;
  const CUtensorMap wmap = make_map(w, TILE_N);
  const CUtensorMap xmap = make_map(x, MAX_M);
  static bool attr = false;
  if (!attr) {
    cudaFuncSetAttribute(skinny_kernel<TILE_N, CHUNK_K>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM_BYTES);
    attr = true;
  }
  const int64_t units = static_cast<int64_t>(p.tiles) * p.chunks;
  const int grid = static_cast<int>(std::min<int64_t>(grid_cap, units));
  skinny_kernel<TILE_N, CHUNK_K><<<grid, C::THREADS, C::SMEM_BYTES,
                                   at::cuda::getCurrentCUDAStream()>>>(wmap, xmap, p);
}

// out[M, N] = x[M, K] @ w[N, K]^T; ws fp32 [>= MAX_M * N] and counters int32
// [>= N / 64] must be zero on entry and are left zero. config selects
// (tile rows, K chunk): 0 = 128x128, 1 = 64x128, 2 = 128x256, 3 = 64x256;
// grid caps the CTA count (0 = one per SM).
void gemm(torch::Tensor out, torch::Tensor x, torch::Tensor w, torch::Tensor ws,
          torch::Tensor counters, int64_t config, int64_t grid) {
  const int M = static_cast<int>(x.size(0)), K = static_cast<int>(x.size(1));
  const int N = static_cast<int>(w.size(0));
  TORCH_CHECK(M >= 1 && M <= MAX_M, "skinny gemm takes 1..8 rows");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && w.scalar_type() == at::kBFloat16 &&
                  out.scalar_type() == at::kBFloat16, "bf16 only");
  TORCH_CHECK(w.size(1) == K && x.stride(1) == 1 && w.stride(1) == 1 && out.is_contiguous(),
              "shape / layout");
  TORCH_CHECK(K % 8 == 0 && (x.stride(0) * 2) % 16 == 0 && (w.stride(0) * 2) % 16 == 0,
              "rows must be 16-byte aligned");
  Params p{};
  p.M = M;
  p.N = N;
  p.K = K;
  TORCH_CHECK(ws.numel() >= static_cast<int64_t>(MAX_M) * N &&
                  counters.numel() >= (N + 63) / 64, "workspace too small");
  p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  p.ws = ws.data_ptr<float>();
  p.counters = counters.data_ptr<int>();
  const int cap = grid > 0 ? static_cast<int>(grid) : GRID;
  switch (config) {
    case 0: launch<128, 128>(p, x, w, cap); break;
    case 1: launch<64, 128>(p, x, w, cap); break;
    case 2: launch<128, 256>(p, x, w, cap); break;
    case 3: launch<64, 256>(p, x, w, cap); break;
    default: TORCH_CHECK(false, "unknown config");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace skinny

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemm", &skinny::gemm, "bf16 x[M<=8, K] @ w[N, K]^T");
}

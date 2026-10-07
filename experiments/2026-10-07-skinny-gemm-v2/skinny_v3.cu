// skinny_v2's operand mapping (16 B loads straight into m16n8k16, K permuted
// identically for weight and activations), made persistent: grid = SMs x
// CTAS, and every warp streams an equal contiguous slice of the flattened
// (16-row tile, 32-wide K step) space, crossing tile boundaries as needed.
// A warp's partial for a tile goes to a fixed slot (its rank among the warps
// touching that tile, fixed by the static partition); the warp that completes
// a tile's step count sums the slots in order (deterministic, no float atomics),
// writes bf16 and resets the tile's counter for the next launch.
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

__device__ __forceinline__ void mma(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                    uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
      "{%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

struct Args {
  const __nv_bfloat16* W;
  const __nv_bfloat16* x;
  __nv_bfloat16* y;
  float* part;     // [tiles][slots][32][4]
  int* count;      // [tiles], zero between launches
  int M, N, K, slots;
  long long total;  // tiles * steps_per_tile
  int nwarps;
};

__device__ __forceinline__ long long warp_begin(long long w, long long total, int nwarps) {
  return w * total / nwarps;
}

template <int WARPS, int UNROLL, int MINB>
__global__ void __launch_bounds__(WARPS * 32, MINB) skinny_v3(Args p) {
  const int lane = threadIdx.x % 32, g = lane >> 2, t = lane & 3;
  const int gw = blockIdx.x * WARPS + threadIdx.x / 32;
  const int spt = p.K / 32;  // steps per tile
  long long s = warp_begin(gw, p.total, p.nwarps);
  const long long e = warp_begin(gw + 1, p.total, p.nwarps);
  const bool tok = g < p.M;
  while (s < e) {
    const int tile = int(s / spt);
    const int k_lo = int(s - (long long)tile * spt);
    const int k_hi = int(min(e - (long long)tile * spt, (long long)spt));
    const int n0 = tile * 16;
    const uint4* w0 = reinterpret_cast<const uint4*>(p.W + size_t(n0 + g) * p.K) + t;
    const uint4* w1 = reinterpret_cast<const uint4*>(p.W + size_t(n0 + g + 8) * p.K) + t;
    const uint4* xr = reinterpret_cast<const uint4*>(p.x + size_t(g) * p.K) + t;
    float c[4] = {0.f, 0.f, 0.f, 0.f};
    for (int k = k_lo; k < k_hi; k += UNROLL) {
      uint4 a[UNROLL], b[UNROLL], v[UNROLL];
#pragma unroll
      for (int u = 0; u < UNROLL; ++u) {
        const bool ok = k + u < k_hi;
        a[u] = ok ? __ldcs(w0 + (k + u) * 4) : make_uint4(0, 0, 0, 0);
        b[u] = ok ? __ldcs(w1 + (k + u) * 4) : make_uint4(0, 0, 0, 0);
        v[u] = ok && tok ? __ldg(xr + (k + u) * 4) : make_uint4(0, 0, 0, 0);
      }
#pragma unroll
      for (int u = 0; u < UNROLL; ++u) {
        mma(c, a[u].x, b[u].x, a[u].y, b[u].y, v[u].x, v[u].y);
        mma(c, a[u].z, b[u].z, a[u].w, b[u].w, v[u].z, v[u].w);
      }
    }
    // slot = this warp's rank among the warps that touch the tile
    const long long tile_lo = (long long)tile * spt;
    int first = int((tile_lo * p.nwarps) / p.total);
    while (first > 0 && warp_begin(first, p.total, p.nwarps) > tile_lo) --first;
    while (warp_begin(first + 1, p.total, p.nwarps) <= tile_lo) ++first;
    const int slot = gw - first;
    float4* dst = reinterpret_cast<float4*>(p.part) + (size_t(tile) * p.slots + slot) * 32 + lane;
    __stcg(dst, make_float4(c[0], c[1], c[2], c[3]));
    __threadfence();
    __syncwarp();
    int done = 0;
    if (lane == 0) done = atomicAdd(p.count + tile, k_hi - k_lo) + (k_hi - k_lo);
    done = __shfl_sync(0xffffffffu, done, 0);
    if (done == spt) {
      __threadfence();
      int last = first;
      while (warp_begin(last + 1, p.total, p.nwarps) < tile_lo + spt) ++last;
      float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
      for (int sl = 0; sl <= last - first; ++sl) {
        const float4 q = __ldcg(reinterpret_cast<const float4*>(p.part) +
                                (size_t(tile) * p.slots + sl) * 32 + lane);
        acc.x += q.x; acc.y += q.y; acc.z += q.z; acc.w += q.w;
      }
      if (2 * t < p.M) {
        p.y[size_t(2 * t) * p.N + n0 + g] = __float2bfloat16(acc.x);
        p.y[size_t(2 * t) * p.N + n0 + g + 8] = __float2bfloat16(acc.z);
      }
      if (2 * t + 1 < p.M) {
        p.y[size_t(2 * t + 1) * p.N + n0 + g] = __float2bfloat16(acc.y);
        p.y[size_t(2 * t + 1) * p.N + n0 + g + 8] = __float2bfloat16(acc.w);
      }
      if (lane == 0) p.count[tile] = 0;
    }
    s = tile_lo + k_hi;
  }
}

template <int WARPS, int UNROLL, int MINB>
void launch(Args& a, int grid) {
  skinny_v3<WARPS, UNROLL, MINB><<<grid, WARPS * 32, 0, c10::cuda::getCurrentCUDAStream()>>>(a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// part / count: workspaces (count zero-initialised once, left zero by the kernel)
torch::Tensor gemm(torch::Tensor x, torch::Tensor w, torch::Tensor part, torch::Tensor count,
                   int64_t warps, int64_t unroll, int64_t ctas_per_sm) {
  const int M = x.size(0), K = x.size(1), N = w.size(0);
  TORCH_CHECK(M <= 8 && N % 16 == 0 && K % 32 == 0, "shape");
  int sms;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, x.device().index());
  auto y = torch::empty({M, N}, x.options());
  Args a;
  a.W = reinterpret_cast<const __nv_bfloat16*>(w.data_ptr());
  a.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  a.y = reinterpret_cast<__nv_bfloat16*>(y.data_ptr());
  a.part = part.data_ptr<float>();
  a.count = count.data_ptr<int>();
  a.M = M; a.N = N; a.K = K;
  const int grid = sms * int(ctas_per_sm);
  a.nwarps = grid * int(warps);
  a.total = (long long)(N / 16) * (K / 32);
  const long long per = (a.total + a.nwarps - 1) / a.nwarps;
  a.slots = int((K / 32 + per - 1) / per) + 2;
  TORCH_CHECK(part.numel() >= (int64_t)(N / 16) * a.slots * 128 && count.numel() >= N / 16, "ws");
#define CASE(W_, U_, B_) \
  if (warps == W_ && unroll == U_ && ctas_per_sm == B_) { launch<W_, U_, B_>(a, grid); return y; }
  CASE(16, 2, 1) CASE(16, 4, 1) CASE(16, 8, 1) CASE(8, 4, 2) CASE(8, 8, 2) CASE(16, 2, 2)
  CASE(16, 4, 2) CASE(32, 2, 1) CASE(32, 4, 1)
#undef CASE
  TORCH_CHECK(false, "unsupported config");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("gemm", &gemm); }

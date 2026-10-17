// Prototype: the glue between fused_qkv_a and q_b in one PDL-aware kernel.
// Replaces inductor's triton_poi_fused_0 (k_pe interleaved RoPE) and
// triton_red_fused_1 (kv_a + q_a RMSNorm) for the GLM-5.3 MLA layout:
// qkv[M, 2624] = [q_c 2048 | kv_c 512 | k_pe 64]. One CTA per token. It lets
// its dependent (q_b's decode_gemm) launch at once, so q_b's first weight
// batch loads while this runs, then waits for qkv_a before reading.
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cfloat>
#include <torch/extension.h>

constexpr int QL = 2048, KVL = 512, PE = 64, LD = QL + KVL + PE;

template <int N>
__device__ __forceinline__ float block_sum(float v, float* sh) {
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
  if (threadIdx.x % 32 == 0) sh[threadIdx.x / 32] = v;
  __syncthreads();
  float s = 0.f;
#pragma unroll
  for (int w = 0; w < N; ++w) s += sh[w];
  return s;
}

__global__ void __launch_bounds__(256)
    qkv_glue_kernel(const __nv_bfloat16* __restrict__ qkv,
                    const int64_t* __restrict__ pos,
                    const __nv_bfloat16* __restrict__ cos_sin,
                    const __nv_bfloat16* __restrict__ wq,
                    const __nv_bfloat16* __restrict__ wkv,
                    __nv_bfloat16* __restrict__ qn,
                    __nv_bfloat16* __restrict__ kvn,
                    __nv_bfloat16* __restrict__ kpe, float eps) {
  __shared__ float shq[8], shk[8];
  asm volatile("griddepcontrol.launch_dependents;");
  asm volatile("griddepcontrol.wait;" ::: "memory");
  const int m = blockIdx.x, t = threadIdx.x;
  const __nv_bfloat16* row = qkv + size_t(m) * LD;
  // q_c: 8 per thread, kv_c: 2 per thread
  const uint4 qv = reinterpret_cast<const uint4*>(row)[t];
  const __nv_bfloat162 kv2 = reinterpret_cast<const __nv_bfloat162*>(row + QL)[t];
  const __nv_bfloat162* q2 = reinterpret_cast<const __nv_bfloat162*>(&qv);
  float q[8];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    q[2 * i] = __low2float(q2[i]);
    q[2 * i + 1] = __high2float(q2[i]);
  }
  const float k0 = __low2float(kv2), k1 = __high2float(kv2);
  float sq = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) sq += q[i] * q[i];
  const float sk = k0 * k0 + k1 * k1;
  const float rq = rsqrtf(block_sum<8>(sq, shq) / float(QL) + eps);
  const float rk = rsqrtf(block_sum<8>(sk, shk) / float(KVL) + eps);
  const uint4 wqv = reinterpret_cast<const uint4*>(wq)[t];
  const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(&wqv);
  uint4 out;
  __nv_bfloat162* o2 = reinterpret_cast<__nv_bfloat162*>(&out);
#pragma unroll
  for (int i = 0; i < 4; ++i)
    o2[i] = __floats2bfloat162_rn(q[2 * i] * rq * __low2float(w2[i]),
                                  q[2 * i + 1] * rq * __high2float(w2[i]));
  reinterpret_cast<uint4*>(qn + size_t(m) * QL)[t] = out;
  const __nv_bfloat162 wk = reinterpret_cast<const __nv_bfloat162*>(wkv)[t];
  reinterpret_cast<__nv_bfloat162*>(kvn + size_t(m) * KVL)[t] =
      __floats2bfloat162_rn(k0 * rk * __low2float(wk), k1 * rk * __high2float(wk));
  if (t < PE / 2) {
    const __nv_bfloat16* cs = cos_sin + pos[m] * PE;
    const float c = __bfloat162float(cs[t]), s = __bfloat162float(cs[PE / 2 + t]);
    const float x0 = __bfloat162float(row[QL + KVL + 2 * t]);
    const float x1 = __bfloat162float(row[QL + KVL + 2 * t + 1]);
    reinterpret_cast<__nv_bfloat162*>(kpe + size_t(m) * PE)[t] =
        __floats2bfloat162_rn(x0 * c - x1 * s, x1 * c + x0 * s);
  }
}

// glue2 = glue + the fp8_ds_mla KV write (concat_and_cache_ds_mla_kernel's
// entry: 512 e4m3 bytes, 4 fp32 tile scales, 64 bf16 RoPE values) into the
// main cache and, when map2 is given, the skip-KV stage. The entry is built
// once from the normalized, bf16-rounded kv_c and the rotated k_pe.
struct Cache {
  uint8_t* base;
  const int64_t* slots;
  int64_t block_stride, entry_stride;
  int block_size;
};

__device__ __forceinline__ uint8_t* entry(const Cache& c, int m) {
  const int64_t s = c.slots[m];
  if (s < 0) return nullptr;
  return c.base + (s / c.block_size) * c.block_stride +
         (s % c.block_size) * c.entry_stride;
}

__global__ void __launch_bounds__(256)
    qkv_glue2_kernel(const __nv_bfloat16* __restrict__ qkv,
                     const int64_t* __restrict__ pos,
                     const __nv_bfloat16* __restrict__ cos_sin,
                     const __nv_bfloat16* __restrict__ wq,
                     const __nv_bfloat16* __restrict__ wkv,
                     __nv_bfloat16* __restrict__ qn, Cache c1, Cache c2,
                     bool has2, float eps) {
  __shared__ float shq[8], shk[8], tmax[8];
  asm volatile("griddepcontrol.launch_dependents;");
  asm volatile("griddepcontrol.wait;" ::: "memory");
  const int m = blockIdx.x, t = threadIdx.x, lane = t % 32, warp = t / 32;
  const __nv_bfloat16* row = qkv + size_t(m) * LD;
  const uint4 qv = reinterpret_cast<const uint4*>(row)[t];
  const __nv_bfloat162 kv2 = reinterpret_cast<const __nv_bfloat162*>(row + QL)[t];
  const __nv_bfloat162* q2 = reinterpret_cast<const __nv_bfloat162*>(&qv);
  float q[8];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    q[2 * i] = __low2float(q2[i]);
    q[2 * i + 1] = __high2float(q2[i]);
  }
  const float k0 = __low2float(kv2), k1 = __high2float(kv2);
  float sq = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) sq += q[i] * q[i];
  const float rq = rsqrtf(block_sum<8>(sq, shq) / float(QL) + eps);
  const float rk = rsqrtf(block_sum<8>(k0 * k0 + k1 * k1, shk) / float(KVL) + eps);
  const uint4 wqv = reinterpret_cast<const uint4*>(wq)[t];
  const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(&wqv);
  uint4 out;
  __nv_bfloat162* o2 = reinterpret_cast<__nv_bfloat162*>(&out);
#pragma unroll
  for (int i = 0; i < 4; ++i)
    o2[i] = __floats2bfloat162_rn(q[2 * i] * rq * __low2float(w2[i]),
                                  q[2 * i + 1] * rq * __high2float(w2[i]));
  reinterpret_cast<uint4*>(qn + size_t(m) * QL)[t] = out;

  // kv_c: elements 2t, 2t + 1; tile = 128 elements = 64 threads = 2 warps
  const __nv_bfloat162 wk = reinterpret_cast<const __nv_bfloat162*>(wkv)[t];
  const __nv_bfloat162 kn =
      __floats2bfloat162_rn(k0 * rk * __low2float(wk), k1 * rk * __high2float(wk));
  float mx = fmaxf(fabsf(__low2float(kn)), fabsf(__high2float(kn)));
  for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
  if (lane == 0) tmax[warp] = mx;
  __syncthreads();
  const int tile = t / 64;
  const float scale = fmaxf(fmaxf(tmax[2 * tile], tmax[2 * tile + 1]) / 448.f, FLT_MIN);
  const uint16_t fp8 =
      uint16_t(__nv_cvt_float_to_fp8(__low2float(kn) / scale, __NV_SATFINITE, __NV_E4M3)) |
      uint16_t(__nv_cvt_float_to_fp8(__high2float(kn) / scale, __NV_SATFINITE, __NV_E4M3)) << 8;
  uint32_t pe = 0;
  if (t < PE / 2) {
    const __nv_bfloat16* cs = cos_sin + pos[m] * PE;
    const float c = __bfloat162float(cs[t]), s = __bfloat162float(cs[PE / 2 + t]);
    const float x0 = __bfloat162float(row[QL + KVL + 2 * t]);
    const float x1 = __bfloat162float(row[QL + KVL + 2 * t + 1]);
    const __nv_bfloat162 r = __floats2bfloat162_rn(x0 * c - x1 * s, x1 * c + x0 * s);
    pe = *reinterpret_cast<const uint32_t*>(&r);
  }
#pragma unroll
  for (int k = 0; k < 2; ++k) {
    if (k == 1 && !has2) break;
    uint8_t* e = entry(k ? c2 : c1, m);
    if (e == nullptr) continue;
    reinterpret_cast<uint16_t*>(e)[t] = fp8;
    if (t % 64 == 0) reinterpret_cast<float*>(e + KVL)[tile] = scale;
    if (t < PE / 2) reinterpret_cast<uint32_t*>(e + KVL + 16)[t] = pe;
  }
}

Cache cache_of(const torch::Tensor& cache, const torch::Tensor& slots) {
  TORCH_CHECK(cache.dim() == 3 && cache.size(2) == KVL + 16 + 2 * PE &&
              cache.scalar_type() == torch::kUInt8 && cache.stride(2) == 1);
  return {cache.data_ptr<uint8_t>(), slots.data_ptr<int64_t>(), cache.stride(0),
          cache.stride(1), static_cast<int>(cache.size(1))};
}

void glue2(torch::Tensor qkv, torch::Tensor pos, torch::Tensor cos_sin,
           torch::Tensor wq, torch::Tensor wkv, torch::Tensor qn,
           torch::Tensor cache1, torch::Tensor slots1,
           std::optional<torch::Tensor> cache2,
           std::optional<torch::Tensor> slots2, double eps, bool pdl) {
  TORCH_CHECK(qkv.size(1) == LD && qkv.stride(0) == LD && cos_sin.size(1) == PE);
  const Cache c1 = cache_of(cache1, slots1);
  const bool has2 = cache2.has_value();
  const Cache c2 = has2 ? cache_of(*cache2, *slots2) : c1;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(qkv.size(0));
  cfg.blockDim = dim3(256);
  cfg.stream = c10::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr;
  attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr.val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = &attr;
  cfg.numAttrs = pdl ? 1 : 0;
  auto B = [](const torch::Tensor& x) {
    return reinterpret_cast<__nv_bfloat16*>(x.data_ptr());
  };
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, qkv_glue2_kernel, B(qkv),
                                    pos.data_ptr<int64_t>(), B(cos_sin), B(wq),
                                    B(wkv), B(qn), c1, c2, has2, float(eps)));
}

void glue(torch::Tensor qkv, torch::Tensor pos, torch::Tensor cos_sin,
          torch::Tensor wq, torch::Tensor wkv, torch::Tensor qn,
          torch::Tensor kvn, torch::Tensor kpe, double eps, bool pdl) {
  TORCH_CHECK(qkv.size(1) == LD && qkv.stride(0) == LD && cos_sin.size(1) == PE);
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(qkv.size(0));
  cfg.blockDim = dim3(256);
  cfg.stream = c10::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr;
  attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr.val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = &attr;
  cfg.numAttrs = pdl ? 1 : 0;
  auto B = [](const torch::Tensor& x) {
    return reinterpret_cast<__nv_bfloat16*>(x.data_ptr());
  };
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, qkv_glue_kernel, B(qkv),
                                    pos.data_ptr<int64_t>(), B(cos_sin), B(wq),
                                    B(wkv), B(qn), B(kvn), B(kpe), float(eps)));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("glue", &glue);
  m.def("glue2", &glue2);
}

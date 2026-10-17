// Prototype: the glue between fused_qkv_a and q_b in one PDL-aware kernel.
// Replaces inductor's triton_poi_fused_0 (k_pe interleaved RoPE) and
// triton_red_fused_1 (kv_a + q_a RMSNorm) for the GLM-5.3 MLA layout:
// qkv[M, 2624] = [q_c 2048 | kv_c 512 | k_pe 64]. One CTA per token. It lets
// its dependent (q_b's decode_gemm) launch at once, so q_b's first weight
// batch loads while this runs, then waits for qkv_a before reading.
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("glue", &glue); }

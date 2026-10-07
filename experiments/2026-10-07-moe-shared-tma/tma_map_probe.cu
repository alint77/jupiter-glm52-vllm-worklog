// The tiered MoE kernel's weight load, alone: a 3D tensor map over
// [E][rows][cols] uint32 (box 128 x 64 = 32 KiB, no swizzle, L2 promotion
// 256B) into a STAGES-deep ring, consumers only release stages. Variants:
// CTAs per SM, stages, and how many producer warps issue the boxes (each
// producer warp owns every PRODUCERS-th stage).
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <torch/extension.h>

__device__ __forceinline__ uint32_t s32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void bar_init(uint64_t* b, int n) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(s32(b)), "r"(n));
}
__device__ __forceinline__ void bar_expect(uint64_t* b, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(s32(b)), "r"(bytes)
               : "memory");
}
__device__ __forceinline__ void bar_arrive(uint64_t* b) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(s32(b)) : "memory");
}
__device__ __forceinline__ void bar_wait(uint64_t* b, int parity) {
  asm volatile(
      "{\n .reg .pred p;\n W_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      " @!p bra W_%=;\n}\n" ::"r"(s32(b)), "r"(parity)
      : "memory");
}
__device__ __forceinline__ void tma_3d(void* dst, const CUtensorMap* map, int x, int y, int z,
                                       uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3, %4}], [%5];" ::"r"(s32(dst)),
      "l"(reinterpret_cast<uint64_t>(map)), "r"(x), "r"(y), "r"(z), "r"(s32(bar))
      : "memory");
}

constexpr int BOX_BYTES = 128 * 4 * 64;  // 32 KiB

__global__ void tma_map_stream(const __grid_constant__ CUtensorMap map, int E, int rows,
                               int col_boxes, int stages, int producers, uint32_t* sink) {
  extern __shared__ __align__(1024) char ring[];
  __shared__ uint64_t full[16], empty[16];
  const int row_boxes = rows / 64;
  const long long units = (long long)E * row_boxes * col_boxes;
  const int u0 = int(units * blockIdx.x / gridDim.x), u1 = int(units * (blockIdx.x + 1) / gridDim.x);
  const int n = u1 - u0;
  const int consumers = blockDim.x / 32 - producers;
  if (threadIdx.x == 0) {
    for (int s = 0; s < stages; ++s) {
      bar_init(&full[s], 1);
      bar_init(&empty[s], consumers);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  if (warp < producers) {
    if (lane == 0)
      for (int i = warp; i < n; i += producers) {
        const int s = i % stages;
        if (i >= stages) bar_wait(&empty[s], ((i / stages) - 1) & 1);
        const int u = u0 + i;
        const int cb = u % col_boxes, rb = (u / col_boxes) % row_boxes, e = u / (col_boxes * row_boxes);
        bar_expect(&full[s], BOX_BYTES);
        tma_3d(ring + size_t(s) * BOX_BYTES, &map, cb * 128, rb * 64, e, &full[s]);
      }
  } else {
    uint32_t acc = 0;
    for (int i = 0; i < n; ++i) {
      const int s = i % stages;
      bar_wait(&full[s], (i / stages) & 1);
      acc ^= reinterpret_cast<const uint32_t*>(ring + size_t(s) * BOX_BYTES)[threadIdx.x];
      __syncwarp();
      if (lane == 0) bar_arrive(&empty[s]);
    }
    if (acc == 0x12345678u) sink[0] = acc;
  }
}

// w: [E][rows][cols] int32, cols % 128 == 0, rows % 64 == 0
void run(torch::Tensor w, torch::Tensor sink, int64_t grid, int64_t stages, int64_t producers) {
  CUtensorMap map;
  const cuuint64_t dims[3] = {cuuint64_t(w.size(2)), cuuint64_t(w.size(1)), cuuint64_t(w.size(0))};
  const cuuint64_t strides[2] = {cuuint64_t(w.stride(1)) * 4, cuuint64_t(w.stride(0)) * 4};
  const cuuint32_t box[3] = {128, 64, 1}, unit[3] = {1, 1, 1};
  TORCH_CHECK(cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_UINT32, 3, w.data_ptr(), dims,
                                     strides, box, unit, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                     CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                                     CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS);
  const int smem = int(stages * BOX_BYTES);
  cudaFuncSetAttribute(tma_map_stream, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  tma_map_stream<<<grid, (producers + 8) * 32, smem, c10::cuda::getCurrentCUDAStream()>>>(
      map, int(w.size(0)), int(w.size(1)), int(w.size(2) / 128), int(stages), int(producers),
      static_cast<uint32_t*>(sink.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run); }

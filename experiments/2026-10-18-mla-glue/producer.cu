// PDL primary for the race test: lets dependents launch at entry, then waits
// `spin` cycles and only then writes x, like a slow upstream kernel.
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

__global__ void produce_kernel(const uint4* src, uint4* dst, size_t n,
                               long long spin) {
  asm volatile("griddepcontrol.launch_dependents;");
  const long long t0 = clock64();
  while (clock64() - t0 < spin) {
  }
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += size_t(gridDim.x) * blockDim.x)
    dst[i] = src[i];
}

void produce(torch::Tensor src, torch::Tensor dst, int64_t spin) {
  const size_t n = src.numel() * src.element_size() / 16;
  produce_kernel<<<4, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const uint4*>(src.data_ptr()),
      reinterpret_cast<uint4*>(dst.data_ptr()), n, spin);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("produce", &produce); }

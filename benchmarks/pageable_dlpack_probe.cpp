#include <ATen/DLConvertor.h>
#include <torch/extension.h>

#include <cstdint>
#include <vector>

namespace {

struct PageableTensorContext {
  at::Tensor owner;
  std::vector<int64_t> shape;
  std::vector<int64_t> strides;
  DLManagedTensor managed{};
};

at::Tensor pageable_cuda_view(const at::Tensor& owner, int64_t device_index) {
  TORCH_CHECK(owner.device().is_cpu());
  TORCH_CHECK(owner.is_contiguous());

  auto* context = new PageableTensorContext{
      owner,
      owner.sizes().vec(),
      owner.strides().vec(),
  };
  context->managed.manager_ctx = context;
  context->managed.deleter = [](DLManagedTensor* managed) {
    delete static_cast<PageableTensorContext*>(managed->manager_ctx);
  };
  context->managed.dl_tensor.data = owner.data_ptr();
  context->managed.dl_tensor.device = {
      kDLCUDA,
      static_cast<int32_t>(device_index),
  };
  context->managed.dl_tensor.ndim = owner.dim();
  context->managed.dl_tensor.dtype = at::getDLDataType(owner);
  context->managed.dl_tensor.shape = context->shape.data();
  context->managed.dl_tensor.strides = context->strides.data();
  context->managed.dl_tensor.byte_offset = 0;
  return at::fromDLPack(&context->managed);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("view", &pageable_cuda_view);
}

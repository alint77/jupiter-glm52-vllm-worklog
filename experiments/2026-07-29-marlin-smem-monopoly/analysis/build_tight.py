import os, torch
from torch.utils.cpp_extension import load
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "marlin_tight")
mod = load(
    name="marlin_tight",
    sources=[os.path.join(D, f) for f in ("binding.cu", "ops.cu", "sm80_kernel.cu")],
    extra_include_paths=["/e/project1/profound/alint77/vllm/csrc", D],
    extra_cuda_cflags=["-O3", "-DMARLIN_NAMESPACE_NAME=marlin_tight", "-DUSE_CUDA", "-static-global-template-stub=false",
                       "-gencode", "arch=compute_90a,code=sm_90a",
                       "--expt-relaxed-constexpr", "-std=c++17"],
    extra_cflags=["-O2", "-DMARLIN_NAMESPACE_NAME=marlin_tight", "-DUSE_CUDA"],
    build_directory=os.path.join(D, "build"),
    verbose=True,
)
print("OK", mod)

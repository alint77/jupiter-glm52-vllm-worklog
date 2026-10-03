"""NVLink P2P bandwidth seen by SM loads vs SM stores (one process, peer
access enabled): GPU 0 reads from / writes to 1, 2 or 3 peers at once, 16 B per
thread access, grid-stride over 64 MB per peer. Also local copy as reference."""
import torch
from torch.utils.cpp_extension import load_inline

src = r'''
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
struct Ptrs { const uint4* p[4]; uint4* q[4]; };
__global__ void rd(Ptrs s, int n, uint4* out, int64_t per) {
  uint4 acc = make_uint4(0,0,0,0);
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < per; i += (int64_t)gridDim.x * blockDim.x)
    for (int r = 0; r < n; ++r) { uint4 v = s.p[r][i]; acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w; }
  if (acc.x == 0x12345678) out[0] = acc;
}
__global__ void wr(Ptrs s, int n, int64_t per) {
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < per; i += (int64_t)gridDim.x * blockDim.x)
    for (int r = 0; r < n; ++r) s.q[r][i] = make_uint4(i, r, 1, 2);
}
void peer(int j) { cudaError_t e = cudaDeviceEnablePeerAccess(j, 0); if (e != cudaErrorPeerAccessAlreadyEnabled) TORCH_CHECK(e == cudaSuccess, cudaGetErrorString(e)); cudaGetLastError(); }
void run(std::vector<int64_t> ptrs, int64_t per, bool write, int blocks, torch::Tensor out) {
  Ptrs s; for (size_t r = 0; r < ptrs.size(); ++r) { s.p[r] = (const uint4*)ptrs[r]; s.q[r] = (uint4*)ptrs[r]; }
  auto st = at::cuda::getCurrentCUDAStream();
  if (write) wr<<<blocks, 512, 0, st>>>(s, ptrs.size(), per / 16);
  else rd<<<blocks, 512, 0, st>>>(s, ptrs.size(), (uint4*)out.data_ptr(), per / 16);
}
'''
ext = load_inline("p2p_probe2", cpp_sources="void run(std::vector<int64_t>, int64_t, bool, int, torch::Tensor); void peer(int);",
                  cuda_sources=src, functions=["run", "peer"], extra_cuda_cflags=["-O3"])
n = torch.cuda.device_count()
for j in range(1, n):
    assert torch.cuda.can_device_access_peer(0, j)
per = 64 << 20
bufs = [torch.empty(per, dtype=torch.uint8, device=f"cuda:{j}") for j in range(n)]
torch.cuda.set_device(0)
for j in range(1, n):
    ext.peer(j)
out = torch.empty(16, dtype=torch.uint8, device="cuda:0")
ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
for write in (False, True):
    for peers in ([0], [1], [1, 2], [1, 2, 3]):
        for blocks in (132 * 2, 132 * 4):
            ptrs = [bufs[j].data_ptr() for j in peers]
            for _ in range(3):
                ext.run(ptrs, per, write, blocks, out)
            ev[0].record()
            for _ in range(10):
                ext.run(ptrs, per, write, blocks, out)
            ev[1].record(); torch.cuda.synchronize()
            ms = ev[0].elapsed_time(ev[1]) / 10
            print(f"{'write' if write else 'read '} peers {peers} blocks {blocks}: {len(peers) * per / ms / 1e6:6.0f} GB/s", flush=True)

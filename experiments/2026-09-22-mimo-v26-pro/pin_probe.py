"""Check GraceAllocation.allocate_pinned: exact size, pinned, true alias.

pin_memory=True rounds to a power of two; cudaHostRegister should not. The CUDA
view must alias the host pages, not copy them, or loader writes are lost.
"""

import torch

from vllm.model_executor.offloader.grace import GraceAllocation

SIZE = 54 * 20_054_016  # one MiMo V2.6 cold tier, just past 2**30


def node_free_kib(node: int) -> int:
    with open(f"/sys/devices/system/node/node{node}/meminfo") as f:
        for line in f:
            if "MemFree" in line:
                return int(line.split()[3])
    raise RuntimeError("no MemFree")


def main() -> None:
    torch.cuda.set_device(0)
    torch.cuda.init()
    node = 0
    before = node_free_kib(node)
    alloc = GraceAllocation.allocate_pinned((SIZE,), torch.uint8, 0, node)
    after = node_free_kib(node)
    print("requested       %.3f GiB" % (SIZE / 2**30))
    print("node0 consumed  %.3f GiB (registered path)" % ((before - after) / 2**20))
    print("is_pinned       ", alloc.cpu_tensor.is_pinned())

    alloc.cpu_tensor[:1024] = 7
    alloc.cpu_tensor[-1024:] = 9
    torch.cuda.synchronize()
    ok_h2d = bool((alloc.cuda_alias[:1024] == 7).all() and (alloc.cuda_alias[-1024:] == 9).all())
    alloc.cuda_alias[4096:8192].fill_(3)
    torch.cuda.synchronize()
    ok_d2h = bool((alloc.cpu_tensor[4096:8192] == 3).all())
    print("alias cpu->gpu  ", ok_h2d)
    print("alias gpu->cpu  ", ok_d2h)
    print("same address    ", alloc.cpu_tensor.data_ptr() == alloc.cuda_alias.data_ptr())

    b2 = node_free_kib(node)
    ref = torch.empty(SIZE, dtype=torch.uint8, pin_memory=True)
    a2 = node_free_kib(node)
    print("pin_memory=True %.3f GiB for the same request (reference)" % ((b2 - a2) / 2**20))
    del ref
    print("PIN PROBE OK" if (alloc.cpu_tensor.is_pinned() and ok_h2d and ok_d2h) else "PIN PROBE FAIL")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import argparse
import ctypes
import json
import os
from collections import Counter

import torch

from vllm.utils.torch_utils import (
    get_accelerator_view_from_cpu_tensor,
    get_pageable_accelerator_view_from_cpu_tensor,
    is_pageable_accelerator_view_supported,
)


def measure_copy(
    source: torch.Tensor,
    destination: torch.Tensor,
    warmups: int,
    iterations: int,
) -> dict[str, float]:
    for _ in range(warmups):
        torch.add(source, 1, out=destination)
    torch.accelerator.synchronize()

    start = torch.Event(enable_timing=True)
    end = torch.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        torch.add(source, 1, out=destination)
    end.record()
    end.synchronize()

    milliseconds = start.elapsed_time(end) / iterations
    if destination[7].item() != source[7].item() + 1:
        raise RuntimeError("CUDA kernel did not copy the expected value")
    num_bytes = source.numel() * source.element_size()
    return {
        "milliseconds": milliseconds,
        "read_gbps": num_bytes / (milliseconds / 1000) / 1e9,
    }


def sample_numa_nodes(tensor: torch.Tensor, samples: int = 512) -> dict[str, int]:
    page_size = os.sysconf("SC_PAGE_SIZE")
    num_bytes = tensor.numel() * tensor.element_size()
    step = max(page_size, num_bytes // samples)
    step -= step % page_size
    offsets = list(range(0, num_bytes, step))[:samples]
    pages = (ctypes.c_void_p * len(offsets))(
        *(tensor.data_ptr() + offset for offset in offsets)
    )
    status = (ctypes.c_int * len(offsets))()
    libnuma = ctypes.CDLL("libnuma.so.1", use_errno=True)
    result = libnuma.move_pages(0, len(offsets), pages, None, status, 0)
    if result < 0:
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno))
    return dict(sorted(Counter(str(node) for node in status).items()))


def lock_pages(tensor: torch.Tensor) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.mlock(
        ctypes.c_void_p(tensor.data_ptr()), tensor.numel() * tensor.element_size()
    )
    if result < 0:
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size-gib", type=float, default=1.0)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--mlock", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device_index = torch.accelerator.current_device_index()
    if not is_pageable_accelerator_view_supported(device_index):
        raise RuntimeError("Pageable Grace memory is not directly CUDA-accessible")

    num_bytes = int(args.size_gib * 1024**3)
    num_elements = num_bytes // torch.empty((), dtype=torch.int64).element_size()
    destination = torch.empty(num_elements, dtype=torch.int64, device="cuda")

    hbm = torch.arange(num_elements, dtype=torch.int64, device="cuda")
    pinned_owner = torch.arange(num_elements, dtype=torch.int64).pin_memory()
    pinned_uva = get_accelerator_view_from_cpu_tensor(pinned_owner)
    pageable_owner = torch.arange(num_elements, dtype=torch.int64)
    if args.mlock:
        lock_pages(pageable_owner)
    pageable_uva = get_pageable_accelerator_view_from_cpu_tensor(
        pageable_owner, device_index
    )
    numa_nodes_before = sample_numa_nodes(pageable_owner)

    results = {
        "device": torch.cuda.get_device_name(device_index),
        "device_index": device_index,
        "size_bytes": num_elements * destination.element_size(),
        "warmups": args.warmups,
        "iterations": args.iterations,
        "mlock": args.mlock,
        "pageable_pointer_identity": pageable_owner.data_ptr()
        == pageable_uva.data_ptr(),
        "pageable_numa_nodes_before": numa_nodes_before,
        "measurements": {
            "hbm": measure_copy(hbm, destination, args.warmups, args.iterations),
            "pinned_uva": measure_copy(
                pinned_uva, destination, args.warmups, args.iterations
            ),
            "pageable_uva": measure_copy(
                pageable_uva, destination, args.warmups, args.iterations
            ),
        },
        "pageable_numa_nodes_after": sample_numa_nodes(pageable_owner),
    }
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

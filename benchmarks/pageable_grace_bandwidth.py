#!/usr/bin/env python3

import argparse
import json

import torch
import triton
import triton.language as tl

from vllm.utils.torch_utils import (
    get_accelerator_view_from_cpu_tensor,
    get_pageable_accelerator_view_from_cpu_tensor,
    is_pageable_accelerator_view_supported,
)


@triton.jit
def copy_kernel(src, dst, num_elements: tl.constexpr, block_size: tl.constexpr):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < num_elements
    values = tl.load(src + offsets, mask=mask)
    tl.store(dst + offsets, values, mask=mask)


def measure_copy(
    source: torch.Tensor,
    destination: torch.Tensor,
    warmups: int,
    iterations: int,
) -> dict[str, float]:
    block_size = 1024
    grid = (triton.cdiv(source.numel(), block_size),)

    for _ in range(warmups):
        copy_kernel[grid](
            source,
            destination,
            num_elements=source.numel(),
            block_size=block_size,
            num_warps=8,
        )
    torch.accelerator.synchronize()

    start = torch.Event(enable_timing=True)
    end = torch.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        copy_kernel[grid](
            source,
            destination,
            num_elements=source.numel(),
            block_size=block_size,
            num_warps=8,
        )
    end.record()
    end.synchronize()

    milliseconds = start.elapsed_time(end) / iterations
    num_bytes = source.numel() * source.element_size()
    return {
        "milliseconds": milliseconds,
        "read_gbps": num_bytes / (milliseconds / 1000) / 1e9,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size-gib", type=float, default=1.0)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device_index = torch.accelerator.current_device_index()
    if not is_pageable_accelerator_view_supported(device_index):
        raise RuntimeError("Pageable Grace memory is not directly CUDA-accessible")

    num_bytes = int(args.size_gib * 1024**3)
    num_elements = num_bytes // torch.empty((), dtype=torch.int64).element_size()
    destination = torch.empty(num_elements, dtype=torch.int64, device="cuda")

    hbm = torch.ones_like(destination)
    pinned_owner = torch.ones(num_elements, dtype=torch.int64, pin_memory=True)
    pinned_uva = get_accelerator_view_from_cpu_tensor(pinned_owner)
    pageable_owner = torch.ones(num_elements, dtype=torch.int64)
    pageable_uva = get_pageable_accelerator_view_from_cpu_tensor(
        pageable_owner, device_index
    )

    results = {
        "device": torch.cuda.get_device_name(device_index),
        "device_index": device_index,
        "size_bytes": num_elements * destination.element_size(),
        "warmups": args.warmups,
        "iterations": args.iterations,
        "pageable_pointer_identity": pageable_owner.data_ptr()
        == pageable_uva.data_ptr(),
        "measurements": {
            "hbm": measure_copy(hbm, destination, args.warmups, args.iterations),
            "pinned_uva": measure_copy(
                pinned_uva, destination, args.warmups, args.iterations
            ),
            "pageable_uva": measure_copy(
                pageable_uva, destination, args.warmups, args.iterations
            ),
        },
    }
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

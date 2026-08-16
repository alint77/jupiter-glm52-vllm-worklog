#!/usr/bin/env python3
"""What read bandwidth can a GPU kernel actually pull from Grace over C2C?

`dense_attn_kv_tier.py` found FA3 GQA pinned at ~52 GB/s from Grace, flat across
batch and context, while the FlashMLA sparse kernel reached ~157 GB/s from the
same kind of buffer and Phase 21 measured 421 GB/s achievable on the link. Those
cannot all be the link. This isolates the ceiling with a trivial streaming-read
kernel whose only variable is how much parallelism it is given.

If a plain read reaches several hundred GB/s, 52 is FA3's problem. If it also
sits near 52, the mapping or the link is.
"""

import argparse
import statistics

import torch
import triton
import triton.language as tl

from vllm.model_executor.offloader.grace import GraceAllocation


@triton.jit
def _stream_read(src_ptr, out_ptr, n_elem, BLOCK: tl.constexpr, PER_CTA: tl.constexpr):
    """Each CTA reads PER_CTA blocks strided by the grid, accumulating so the
    loads cannot be optimized away."""
    pid = tl.program_id(0)
    grid = tl.num_programs(0)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for i in range(PER_CTA):
        offs = (pid + i * grid) * BLOCK + tl.arange(0, BLOCK)
        acc += tl.load(src_ptr + offs, mask=offs < n_elem, other=0.0).to(tl.float32)
    tl.store(out_ptr + pid, tl.sum(acc))


@triton.jit
def _gather_rows(
    src_ptr, idx_ptr, out_ptr, n_rows, ROW: tl.constexpr, PER_CTA: tl.constexpr
):
    """Gather PER_CTA scattered rows of ROW elements each -- the DSA pattern.

    Contiguous streaming reaches the C2C roof; this asks whether a scattered
    row gather can, which is what decides if rewriting the attention kernel
    could recover the bandwidth FlashMLA sparse currently leaves unused.
    """
    pid = tl.program_id(0)
    grid = tl.num_programs(0)
    acc = tl.zeros((ROW,), dtype=tl.float32)
    for i in range(PER_CTA):
        slot = pid + i * grid
        row = tl.load(idx_ptr + slot, mask=slot < n_rows, other=0)
        offs = row * ROW + tl.arange(0, ROW)
        acc += tl.load(src_ptr + offs).to(tl.float32)
    tl.store(out_ptr + pid, tl.sum(acc))


def timed(fn, warmups: int, iters: int) -> float:
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


@triton.jit
def _tma_read(desc, out_ptr, n_tiles, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    grid = tl.num_programs(0)
    acc = tl.zeros((1, BLOCK), dtype=tl.float32)
    for i in range(n_tiles):
        row = pid + i * grid
        acc += desc.load([row, 0]).to(tl.float32)
    tl.store(out_ptr + pid, tl.sum(acc))


def tma_probe(hbm, grace, device, args) -> None:
    """Can a TMA descriptor target host-mapped (UVA) memory, and how fast?

    FA3's Hopper mainloop loads K and V through TMA, so if TMA degrades or
    falls back on host memory that would explain its 52 GB/s from Grace.
    """
    from triton.tools.tensor_descriptor import TensorDescriptor

    BLOCK = 512
    print(f"\n{'tier':8} {'TMA GB/s':>10}   note")
    for name, buf in (("hbm", hbm), ("grace", grace)):
        rows = buf.numel() // BLOCK
        view = buf[: rows * BLOCK].view(rows, BLOCK)
        try:
            desc = TensorDescriptor.from_tensor(view, [1, BLOCK])
        except Exception as exc:  # noqa: BLE001
            print(
                f"{name:8} {'--':>10}   descriptor failed: {type(exc).__name__}: {exc}"[
                    :110
                ]
            )
            continue
        ctas = 1056
        n_tiles = max(1, rows // ctas)
        out = torch.zeros(ctas, dtype=torch.float32, device=device)
        try:
            triton.set_allocator(
                lambda size, align, stream: torch.empty(
                    size, dtype=torch.int8, device=device
                )
            )
            fn = lambda: _tma_read[(ctas,)](desc, out, n_tiles, BLOCK=BLOCK)  # noqa: E731
            ms = timed(fn, args.warmups, args.iterations)
            covered = ctas * n_tiles * BLOCK * 2
            print(f"{name:8} {covered / (ms / 1000) / 1e9:>10.1f}   ok")
        except Exception as exc:  # noqa: BLE001
            print(
                f"{name:8} {'--':>10}   kernel failed: {type(exc).__name__}: {exc}"[
                    :110
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--numa-node", type=int, default=-1)
    parser.add_argument("--gib", type=float, default=4.0)
    parser.add_argument(
        "--ctas", type=int, nargs="+", default=[132, 264, 528, 1056, 2112]
    )
    parser.add_argument("--block", type=int, default=1024)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cuda:0")
    numa_node = args.numa_node
    if numa_node < 0:
        from vllm.platforms import current_platform

        numa_node = current_platform.get_device_numa_node(device.index or 0)
    print(f"Grace NUMA node {numa_node}, {torch.cuda.get_device_name(device)}")

    n_elem = int(args.gib * 2**30) // 2  # bf16
    nbytes = n_elem * 2

    hbm = torch.ones(n_elem, dtype=torch.bfloat16, device=device)
    alloc = GraceAllocation.allocate_pinned(
        (n_elem,), torch.bfloat16, device.index or 0, numa_node
    )
    alloc.cpu_tensor.fill_(1.0)
    placement = alloc.audit_numa(samples=64)
    print(f"Grace buffer {args.gib} GiB, {placement.local_fraction:.0%} local")
    if placement.local_fraction < 0.95:
        raise SystemExit("not NUMA-local; bind with numactl --membind")
    grace = alloc.cuda_alias

    print(f"\n{'CTAs':>6} {'HBM GB/s':>10} {'Grace GB/s':>12} {'ratio':>7}")
    for ctas in args.ctas:
        per_cta = max(1, n_elem // (ctas * args.block))
        out = torch.zeros(ctas, dtype=torch.float32, device=device)

        def run(src):
            return lambda: _stream_read[(ctas,)](
                src, out, n_elem, BLOCK=args.block, PER_CTA=per_cta
            )

        covered = ctas * per_cta * args.block * 2
        h = timed(run(hbm), args.warmups, args.iterations)
        g = timed(run(grace), args.warmups, args.iterations)
        hg = covered / (h / 1000) / 1e9
        gg = covered / (g / 1000) / 1e9
        print(f"{ctas:>6} {hg:>10.1f} {gg:>12.1f} {hg / gg:>6.2f}x")

    # Scattered row gather: the access pattern the sparse attention kernel has.
    # Triton needs power-of-2 tiles; sweep granularity around the
    # 656-byte MLA row (328 bf16 elements) instead of matching it.
    row_elems = 256
    n_rows = n_elem // row_elems
    gen = torch.Generator(device=device).manual_seed(17)
    print(
        f"\n{'gran':>7} {'CTAs':>6} {'HBM GB/s':>10} {'Grace GB/s':>12} "
        f"{'ratio':>7}   (scattered gather)"
    )
    for row_elems in (64, 128, 256, 512):
        n_rows = n_elem // row_elems
        for gather_rows in (262144,):
            idx = torch.randint(
                0,
                n_rows,
                (gather_rows,),
                generator=gen,
                device=device,
                dtype=torch.int32,
            )
            for ctas in (132, 264, 528, 1056, 2112):
                per_cta = max(1, gather_rows // ctas)
                out_g = torch.zeros(ctas, dtype=torch.float32, device=device)

                def run(src, per_cta=per_cta, ctas=ctas):
                    return lambda: _gather_rows[(ctas,)](
                        src, idx, out_g, gather_rows, ROW=row_elems, PER_CTA=per_cta
                    )

                covered = ctas * per_cta * row_elems * 2
                h = timed(run(hbm), args.warmups, args.iterations)
                g = timed(run(grace), args.warmups, args.iterations)
                hg = covered / (h / 1000) / 1e9
                gg = covered / (g / 1000) / 1e9
                print(
                    f"{gather_rows:>9} {ctas:>6} {hg:>10.1f} {gg:>12.1f} {hg / gg:>6.2f}x"
                )

    tma_probe(hbm, grace, device, args)

    # Copy-engine control: cudaMemcpyAsync does not go through SMs at all.
    dst = torch.empty(n_elem, dtype=torch.bfloat16, device=device)
    ms = timed(lambda: dst.copy_(alloc.cpu_tensor, non_blocking=True), 3, 10)
    print(f"\ncopy engine H2D: {nbytes / (ms / 1000) / 1e9:.1f} GB/s")


if __name__ == "__main__":
    main()

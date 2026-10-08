"""FlashMLA fp8 sparse decode at the prod DCP4 verify shape, 78 layers per
CUDA graph (as in the verify graph), one schedule metadata per step shared by
all layers (as FlashMLASparseMetadataBuilder does).

Per rank: q (1, 8, 64, 576) bf16, fp8 KV (656 B/token), each query attends
its own ~512 of the rank's 1280 tokens (a 5K context over 4 DCP ranks), the 8
queries' sets overlapping. Prod passes 2048-wide indices with this rank's
entries compacted to the front and -1 after, and no topk_length.

Variants: index width W (2048 = prod, or narrower), topk_length passed or not.
Reports graph replay time per layer, main/combine kernel split from the torch
profiler, and max |out - prod out|.

    bench.py [--layers 78] [--valid 512] [--iters 50]
"""
import argparse

import torch

from vllm.third_party.flashmla.flash_mla_interface import (
    flash_mla_with_kvcache,
    get_mla_metadata,
)

BS, D, DV, HQ, T, ROW = 64, 576, 512, 64, 8, 656


def make_layer(owned: int, valid: int, gen: torch.Generator):
    nblk = (owned + BS - 1) // BS + 1
    kv = torch.empty(nblk, BS, 1, ROW, dtype=torch.uint8, device="cuda")
    nope = (torch.randn(nblk, BS, 512, device="cuda") * 0.5).to(torch.float8_e4m3fn)
    kv[..., 0, :512] = nope.view(torch.uint8)
    kv[..., 0, 512:528] = torch.full((nblk, BS, 4), 0.02, device="cuda").view(torch.uint8)
    kv[..., 0, 528:] = torch.randn(nblk, BS, 64, device="cuda").to(torch.bfloat16).view(torch.uint8)
    q = torch.randn(1, T, HQ, D, device="cuda", dtype=torch.bfloat16)
    base = torch.randperm(owned, generator=gen)[:valid]
    rows, counts = [], []
    for _ in range(T):
        # each query: the shared base with ~10% swapped and +-5% size jitter
        n = int(valid * (0.95 + 0.1 * torch.rand(1, generator=gen).item()))
        swap = torch.randperm(owned, generator=gen)[: n // 10]
        s = torch.unique(torch.cat([base[: n - n // 10], swap]))[:n]
        rows.append(s)
        counts.append(len(s))
    return kv, q, rows, counts


def indices_for(rows, width: int) -> torch.Tensor:
    idx = torch.full((1, T, width), -1, dtype=torch.int32)
    for t, s in enumerate(rows):
        idx[0, t, : len(s)] = s[:width].to(torch.int32)
    return idx.cuda()


def run_variant(layers, width, with_len, iters):
    idxs = [indices_for(rows, width) for _, _, rows, _ in layers]
    lens = [torch.tensor([max(c)], dtype=torch.int32, device="cuda") for *_, c in layers]
    outs = [torch.empty(1, T, HQ, DV, dtype=torch.bfloat16, device="cuda") for _ in layers]
    meta_box = {}

    def step():
        meta, _ = get_mla_metadata()
        meta_box["m"] = meta
        for (kv, q, _, _), idx, ln, out in zip(layers, idxs, lens, outs):
            flash_mla_with_kvcache(
                q=q, k_cache=kv, block_table=None, cache_seqlens=None, head_dim_v=DV,
                tile_scheduler_metadata=meta, is_fp8_kvcache=True, indices=idx,
                topk_length=ln if with_len else None, softmax_scale=D**-0.5, out=out,
            )

    step()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        step()
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        g.replay()
    b.record()
    torch.cuda.synchronize()
    per_layer_us = a.elapsed_time(b) / iters / len(layers) * 1e3

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as p:
        g.replay()
        torch.cuda.synchronize()
    split = {}
    grids = {}
    for e in p.events():
        if e.device_type != torch.autograd.DeviceType.CUDA:
            continue
        k = "combine" if "combine" in e.name else ("main" if "mla" in e.name else "other")
        split[k] = split.get(k, 0.0) + e.device_time_total / len(layers)
    nsp = meta_box["m"].tile_scheduler_metadata.shape[0] if meta_box["m"].tile_scheduler_metadata is not None else None
    return per_layer_us, split, nsp, [o.clone() for o in outs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=78)
    ap.add_argument("--valid", type=int, default=512)
    ap.add_argument("--owned", type=int, default=1280)
    ap.add_argument("--iters", type=int, default=50)
    a = ap.parse_args()
    gen = torch.Generator().manual_seed(0)
    layers = [make_layer(a.owned, a.valid, gen) for _ in range(a.layers)]
    maxc = max(max(c) for *_, c in layers)
    print(f"{a.layers} layers, valid per query {min(min(c) for *_, c in layers)}..{maxc}, "
          f"owned {a.owned}, {torch.cuda.get_device_name()}")
    ref = None
    # topk_length: the sm90 sparse fp8 kernel asserts it is null ("V3.2 does not
    # support dynamic topk length")
    variants = [(2048, False), (1024, False), (768, False), (640, False)]
    if maxc <= 512:
        variants.append((512, False))
    print(f"{'width':>6s} {'topk_len':>8s} {'us/layer':>9s} {'main':>7s} {'combine':>8s} "
          f"{'other':>6s} {'sm_parts':>8s} {'max|d|':>9s}")
    for width, with_len in variants:
        if width < maxc:
            continue
        us, split, nsp, outs = run_variant(layers, width, with_len, a.iters)
        if ref is None:
            ref = outs
        d = max((o.float() - r.float()).abs().max().item() for o, r in zip(outs, ref))
        print(f"{width:6d} {str(with_len):>8s} {us:9.2f} {split.get('main', 0):7.2f} "
              f"{split.get('combine', 0):8.2f} {split.get('other', 0):6.2f} {str(nsp):>8s} {d:9.2e}")


if __name__ == "__main__":
    main()

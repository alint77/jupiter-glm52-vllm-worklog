#!/usr/bin/env python3
"""Integrated Phase 1 decode attention vs prod, graph replay of one step.

    PYTHONPATH=<worktree> python bench_integrated.py --tokens 8,16,32,64

One step = 78 layers; every `--indexer-every`-th layer has its own indexer and
builds the schedule (dcp_decode_sched_meta), the others reuse it, as in the
served model. prod: FlashMLA's planner once per step (shared sched_meta) and
flash_mla_with_kvcache per layer. new: the vLLM schedule + direct
sparse_decode_fwd. The FlashMLA library is whatever the worktree holds.
Index rows: W wide, 470-540 live keys as a prefix, one KV region per layer.
"""

import argparse

import torch

from vllm.v1.attention.backends.mla.flashmla_sparse import FlashMLASparseImpl
from vllm.v1.attention.ops.flashmla import flash_mla_with_kvcache, get_mla_metadata

H, D_QK, REGIONS, BLOCKS = 64, 576, 8, 1600


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default="8,16,32,64")
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--layers", type=int, default=78)
    ap.add_argument("--indexer-layers", type=int, default=21)
    ap.add_argument("--reps", type=int, default=30)
    args = ap.parse_args()
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    gen = torch.Generator(device=dev).manual_seed(1234)
    n = BLOCKS * 64
    impl = object.__new__(FlashMLASparseImpl)
    impl.fp8_decode_padded_heads = H
    scale = D_QK**-0.5
    indexer = {round(i * args.layers / args.indexer_layers)
               for i in range(args.indexer_layers)}

    def region():
        nope = (torch.rand((n, 512), generator=gen, device=dev) * 4 - 2).to(
            torch.float8_e4m3fn)
        sc = torch.rand((n, 4), generator=gen, device=dev) * 0.5 + 0.05
        rope = torch.randn((n, 64), generator=gen, device=dev).to(torch.bfloat16)
        return torch.cat([nope.view(torch.uint8), sc.view(torch.uint8),
                          rope.view(torch.uint8)], 1).view(BLOCKS, 64, 1, 656)

    kvs = [region() for _ in range(REGIONS)]

    def time_graph(step):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            step()
        torch.cuda.current_stream().wait_stream(stream)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = step()
        g.replay()
        torch.cuda.synchronize()
        best = []
        for _ in range(3):
            s, e = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            s.record()
            for _ in range(args.reps):
                g.replay()
            e.record()
            torch.cuda.synchronize()
            best.append(s.elapsed_time(e) / args.reps * 1000)
        return min(best), out

    for t in [int(x) for x in args.tokens.split(",")]:
        q = (torch.randn((t, H, D_QK), generator=gen, device=dev) / 8).bfloat16()
        idx = torch.randint(0, n, (t, args.width), generator=gen, device=dev).int()
        live = torch.randint(470, 541, (t,), generator=gen, device=dev).int()
        idx = idx.masked_fill(torch.arange(args.width, device=dev) >= live[:, None], -1)
        cache_lens = torch.full((1,), n, dtype=torch.int32, device=dev)
        dummy_bt = torch.zeros((1, 1), dtype=torch.int32, device=dev)

        def prod():
            meta, _ = get_mla_metadata()
            out = None
            for layer in range(args.layers):
                out, _ = flash_mla_with_kvcache(
                    q=q.unsqueeze(0), k_cache=kvs[layer % REGIONS],
                    block_table=dummy_bt, head_dim_v=512, cache_seqlens=cache_lens,
                    tile_scheduler_metadata=meta, is_fp8_kvcache=True,
                    indices=idx.unsqueeze(0), softmax_scale=scale)
            return out

        def new():
            sched, out = None, None
            for layer in range(args.layers):
                if layer in indexer or sched is None:
                    sched = impl._decode_sched_meta(idx, live)
                out = torch.ops._flashmla_C.sparse_decode_fwd(
                    q.unsqueeze(0), kvs[layer % REGIONS], idx.unsqueeze(0), None,
                    None, *sched, None, None, None, 512, scale, None)[0]
            return out

        tp, op = time_graph(prod)
        tn, on = time_graph(new)
        err = (op.float() - on.float()).abs().max().item()
        print(f"T={t:3d} W={args.width} prod {tp:8.1f} us/step  new {tn:8.1f}"
              f" us/step  delta {tn - tp:+7.1f} us  maxdiff {err:.1e}", flush=True)
    print("BENCH DONE", flush=True)


if __name__ == "__main__":
    main()

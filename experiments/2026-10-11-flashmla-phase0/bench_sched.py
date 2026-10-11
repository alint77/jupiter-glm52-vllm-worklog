#!/usr/bin/env python3
"""FlashMLA sparse fp8 decode (V3.2, h64): split schedule and width sweep.

    python bench_sched.py --lib <_flashmla_C.so> --tokens 8,32,64

Production DCP4 decode shape: q [1, T, 64, 576], fp8_ds_mla KV, index rows
WIDTH wide with 470-540 live keys compacted to the front (-1 tail), one KV
region per layer so the gather misses L2. For each T:
  planner   FlashMLA's own schedule (fixed_overhead_num_blocks = 5)
  bal-S     hand-built DecodingSchedMeta: the row's blocks split evenly into
            S splits (S <= num_sm_parts = 132 // T)
at widths 768 (prod) and narrower (early stop at the max live count). Timing
is graph replay of --layers layers, main+combine and main only.
"""

import argparse

import torch

H, D_QK, REGIONS, BLOCKS = 64, 576, 8, 1600


def balanced(nb: int, splits: int, parts: int, dev) -> tuple:
    meta = torch.zeros((parts, 8), dtype=torch.int32)
    meta[:, 0] = 1  # begin_req_idx = b: idle part
    bounds = [nb * i // splits for i in range(splits + 1)]
    for i in range(splits):
        meta[i] = torch.tensor([0, 0, bounds[i], bounds[i + 1], i,
                                int(splits > 1), int(splits > 1), 0])
    nsp = torch.tensor([0, splits], dtype=torch.int32)
    return meta.to(dev), nsp.to(dev)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True)
    ap.add_argument("--tokens", default="8,32,64")
    ap.add_argument("--widths", default="768,640,576")
    ap.add_argument("--layers", type=int, default=78)
    ap.add_argument("--reps", type=int, default=30)
    args = ap.parse_args()
    torch.ops.load_library(args.lib)
    fmla = torch.ops._flashmla_C
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    gen = torch.Generator(device=dev).manual_seed(1234)
    n = BLOCKS * 64

    def region():
        nope = (torch.rand((n, 512), generator=gen, device=dev) * 4 - 2).to(
            torch.float8_e4m3fn)
        scale = torch.rand((n, 4), generator=gen, device=dev) * 0.5 + 0.05
        rope = torch.randn((n, 64), generator=gen, device=dev).to(torch.bfloat16)
        return torch.cat([nope.view(torch.uint8), scale.view(torch.uint8),
                          rope.view(torch.uint8)], 1).view(BLOCKS, 64, 1, 656)

    kvs = [region() for _ in range(REGIONS)]
    has_partial = hasattr(fmla, "sparse_decode_fwd_partial")

    def full(q, kv, idx, sched):
        return fmla.sparse_decode_fwd(q, kv, idx, None, None, *sched, None, None,
                                      None, 512, 0.07, None)[0]

    def main_only(q, kv, idx, sched):
        fmla.sparse_decode_fwd_partial(q, kv, idx, None, None, *sched, None,
                                       None, None, 512, 0.07, None)

    def time_graph(fn, q, idxs, sched):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            fn(q, kvs[0], idxs, sched)
        torch.cuda.current_stream().wait_stream(stream)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for layer in range(args.layers):
                fn(q, kvs[layer % REGIONS], idxs, sched)
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
            best.append(s.elapsed_time(e) / args.reps / args.layers * 1000)
        return min(best)

    for T in [int(t) for t in args.tokens.split(",")]:
        parts = max(132 // T, 1)
        q = (torch.randn((1, T, H, D_QK), generator=gen, device=dev) / 8).bfloat16()
        idx = torch.randint(0, n, (1, T, 768), generator=gen, device=dev).int()
        live = torch.randint(470, 540, (T, 1), generator=gen, device=dev)
        idx = idx.masked_fill(torch.arange(768, device=dev) >= live, -1)
        ref = None
        print(f"T={T} parts={parts} max_live={int(live.max())}", flush=True)
        for W in [int(w) for w in args.widths.split(",")]:
            if int(live.max()) > W:
                continue
            idxw = idx[..., :W].contiguous()
            nb = W // 64
            _, _, sm, sp = fmla.sparse_decode_fwd(
                q, kvs[0], idxw, None, None, None, None, None, None, None, 512,
                0.07, None)
            pm = sm.view(-1, 8)[: max(parts, 1)].cpu()
            spl = int((sp[1] - sp[0]).item())
            lens = [int(r[3] - r[2]) for r in pm if r[0] == 0]
            scheds = {f"planner({spl}: {'+'.join(map(str, lens))})": (sm, sp)}
            for S in range(1, min(parts, nb) + 1):
                scheds[f"bal-{S} ({-(-nb // S)} blk)"] = balanced(nb, S, parts, dev)
            for name, sched in scheds.items():
                out = full(q, kvs[0], idxw, sched).float()
                if ref is None:
                    ref = out
                err = (out - ref).abs().max().item()
                tf = time_graph(full, q, idxw, sched)
                tm = time_graph(main_only, q, idxw, sched) if has_partial else 0.0
                print(f"  W={W:4d} {name:28s} main+combine {tf:6.2f}  main {tm:6.2f}"
                      f" us/layer  maxdiff {err:.1e}", flush=True)
    print("SCHED BENCH DONE", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""FlashMLA sparse decode probes: timing, output checksum, per-CTA timeline.

    python bench_probe.py --lib <variant .so> [--trace] \
        --configs 1:768:bal-1,8:768:planner,64:768:bal-2

Same setup as bench_sched.py. Per config (T:width:schedule):
  - graph replay of --layers layers: main+combine and main only (us/layer);
  - checksum of the bf16 output bits (variants that keep the math must match);
  - with --trace (a -DFMLA_TRACE build): the last layer's per-CTA stamps
    (globaltimer ns), summarized as median / max over the active CTAs.
"""

import argparse
import math
import os

import torch

H, D_QK, REGIONS, BLOCKS = 64, 576, 8, 1600
NCTA, SLOTS = 132, 128


def balanced(nb, splits, parts, dev):
    meta = torch.zeros((parts, 8), dtype=torch.int32)
    meta[:, 0] = 1
    bounds = [nb * i // splits for i in range(splits + 1)]
    for i in range(splits):
        meta[i] = torch.tensor([0, 0, bounds[i], bounds[i + 1], i,
                                int(splits > 1), int(splits > 1), 0])
    return meta.to(dev), torch.tensor([0, splits], dtype=torch.int32, device=dev)


def summarize(buf: torch.Tensor, nblk: int) -> None:
    gt = buf.view(NCTA, SLOTS)[:, :64].cpu().long()
    entered = gt[:, 0] > 0
    t0 = int(gt[entered, 0].min())
    active = entered & (gt[:, 2] > 0)
    a = gt[active] - t0
    a[gt[active] == 0] = -1

    def stat(name, x):
        x = x[x >= 0].double() / 1000
        if len(x):
            print(f"    {name:34s} med {x.median():6.2f}  max {x.max():6.2f} us",
                  flush=True)

    print(f"    CTAs entered {int(entered.sum())}, active {int(active.sum())}; "
          f"entry spread (all) {(gt[entered, 0].max() - t0) / 1000:.2f} us; "
          f"globaltimer step gcd {math.gcd(*(gt[entered] - t0)[gt[entered] > 0].tolist())} ns",
          flush=True)
    stat("entry", a[:, 0])
    stat("entry -> barriers init (1)", a[:, 1] - a[:, 0])
    stat("entry -> sched meta (44)", a[:, 44] - a[:, 0])
    stat("entry -> Q ready (2)", a[:, 2] - a[:, 0])
    stat("entry -> producer start (3)", a[:, 3] - a[:, 0])
    for j in range(nblk):
        stat(f"blk{j}: producer done (from entry)", a[:, 28 + j] - a[:, 0])
        stat(f"blk{j}: K ready at WG0 (from entry)", a[:, 4 + j] - a[:, 0])
        stat(f"blk{j}: PV done WG0 (from entry)", a[:, 16 + j] - a[:, 0])
    stat("loop end (40, from entry)", a[:, 40] - a[:, 0])
    stat("epilogue WG0 40 -> 41", a[:, 41] - a[:, 40])
    stat("WG1 epilogue end (42, from entry)", a[:, 42] - a[:, 0])
    stat("CTA end (41)", a[:, 41])
    ends = a[:, 41][a[:, 41] >= 0]
    print(f"    kernel span first entry -> last WG0 end {ends.max() / 1000:.2f} us",
          flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True)
    ap.add_argument("--configs", required=True)
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--layers", type=int, default=78)
    ap.add_argument("--reps", type=int, default=30)
    args = ap.parse_args()
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    tbuf = torch.zeros(NCTA * SLOTS, dtype=torch.int64, device=dev)
    if args.trace:
        os.environ["FMLA_TRACE_PTR"] = str(tbuf.data_ptr())
    torch.ops.load_library(args.lib)
    fmla = torch.ops._flashmla_C
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

    def full(q, kv, idx, sched):
        return fmla.sparse_decode_fwd(q, kv, idx, None, None, *sched, None, None,
                                      None, 512, 0.07, None)[0]

    def main_only(q, kv, idx, sched):
        fmla.sparse_decode_fwd_partial(q, kv, idx, None, None, *sched, None,
                                       None, None, 512, 0.07, None)

    def graph(fn, q, idx, sched):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            fn(q, kvs[0], idx, sched)
        torch.cuda.current_stream().wait_stream(stream)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for layer in range(args.layers):
                fn(q, kvs[layer % REGIONS], idx, sched)
        return g

    def time_graph(g):
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

    data = {}
    for cfg in args.configs.split(","):
        T, W, name = cfg.split(":")
        T, W = int(T), int(W)
        if T not in data:
            q = (torch.randn((1, T, H, D_QK), generator=gen, device=dev) / 8).bfloat16()
            idx = torch.randint(0, n, (1, T, 768), generator=gen, device=dev).int()
            live = torch.randint(470, 540, (T, 1), generator=gen, device=dev)
            idx = idx.masked_fill(torch.arange(768, device=dev) >= live, -1)
            data[T] = (q, idx)
        q, idx = data[T]
        idxw = idx[..., :W].contiguous()
        nb, parts = W // 64, max(132 // T, 1)
        _, _, sm, sp = fmla.sparse_decode_fwd(q, kvs[0], idxw, None, None, None,
                                              None, None, None, None, 512, 0.07,
                                              None)
        if name == "planner":
            sched = (sm, sp)
            pm = sm.view(-1, 8)[:parts].cpu()
            lens = [int(r[3] - r[2]) for r in pm if r[0] == 0]
        else:
            S = int(name.split("-")[1])
            sched = balanced(nb, S, parts, dev)
            lens = [nb * (i + 1) // S - nb * i // S for i in range(S)]
        out = full(q, kvs[0], idxw, sched)
        torch.cuda.synchronize()
        csum = int(out.view(torch.int16).long().sum())
        gf, gm = graph(full, q, idxw, sched), graph(main_only, q, idxw, sched)
        tf, tm = time_graph(gf), time_graph(gm)
        print(f"T={T:2d} W={W} {name:8s} ({'+'.join(map(str, lens))})  main+combine"
              f" {tf:6.2f}  main {tm:6.2f} us/layer  checksum {csum}", flush=True)
        if args.trace:
            for label, g in (("main only", gm),):
                tbuf.zero_()
                g.replay()
                torch.cuda.synchronize()
                print(f"  trace ({label}, last layer):", flush=True)
                summarize(tbuf, max(lens))
    print("PROBE DONE", flush=True)


if __name__ == "__main__":
    main()

"""Does an L2 prefetch issued at the start of a latency-bound window make the
decode GEMMs after it faster, and what does it cost the work in between?

Window A (o_proj): FlashMLA-like phase (an 18 us idle kernel plus a 17 MB
split-accumulator write and read, as the split combine does), then 7 us (DCP
reduce-scatter), then o_proj. Window B (qkv_a + q_b of the next layer): 15 us
(router / top-k / prep), then a MoE-like phase: either HBM-bound (stream 160
MB of weights) or C2C-bound (idle 60 us), then qkv_a and q_b. The prefetch
runs on a side stream forked at the window start, inside a CUDA graph, like it
would in the model. L2 is flushed before each replay. Kernel times come from
the profiler's device timestamps (per replay, by position).

    window_bench.py [--reps 100]
"""
import argparse
import statistics as st

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

from vllm.model_executor.layers.l2_prefetch import l2_prefetch

dev = torch.device("cuda")
CLK = torch.cuda.get_device_properties(0).clock_rate * 1e3  # Hz
flush = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
accum = torch.empty(17 << 20, dtype=torch.uint8, device=dev)
moe_w = torch.empty(160 << 20, dtype=torch.uint8, device=dev)
red = torch.empty(1, dtype=torch.float32, device=dev)


def idle(us):
    torch.cuda._sleep(int(us * 1e-6 * CLK))


W = {"o_proj": torch.randn(6144, 4096, dtype=torch.bfloat16, device=dev),
     "qkv_a": torch.randn(2624, 6144, dtype=torch.bfloat16, device=dev),
     "q_b": torch.randn(4096, 2048, dtype=torch.bfloat16, device=dev)}
X = {k: torch.randn(8, w.shape[1], dtype=torch.bfloat16, device=dev) for k, w in W.items()}


def window_a(pf, post=False, late_join=False):
    side = torch.cuda.Stream()
    main = torch.cuda.current_stream()
    flush.fill_(1)

    def fork():
        side.wait_stream(main)
        with torch.cuda.stream(side):
            if pf:
                pf(W["o_proj"])

    if not post:
        fork()
    idle(18)
    accum.fill_(3)
    torch.sum(accum.view(torch.int32), dtype=torch.int64)
    if post:
        fork()
    idle(11)  # DCP reduce-scatter + W_UV
    if not late_join:
        main.wait_stream(side)
    F.linear(X["o_proj"], W["o_proj"])
    if late_join:
        main.wait_stream(side)


def window_b(pf, moe, which=("qkv_a", "q_b"), frac=None):
    side = torch.cuda.Stream()
    main = torch.cuda.current_stream()
    flush.fill_(1)
    side.wait_stream(main)
    with torch.cuda.stream(side):
        if pf:
            for k in which:
                pf(W[k]) if frac is None or k == "q_b" else l2_prefetch(W[k], frac, evict_last=True)
    idle(15)
    if moe == "hbm":
        torch.sum(moe_w.view(torch.int32), dtype=torch.int64)
    else:
        idle(60)
    main.wait_stream(side)
    F.linear(X["qkv_a"], W["qkv_a"])
    red.zero_()  # separates the two GEMMs' kernels in the trace
    F.linear(X["q_b"], W["q_b"])


def run(fn, reps):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(reps):
            g.replay()
        torch.cuda.synchronize()
    ev = sorted(((e.start_ns(), e.duration_ns(), e.name()) for e in p.profiler.kineto_results.events()
                 if e.device_type().name == "CUDA"), key=lambda x: x[0])
    return ev


def is_gemm(name):
    return any(k in name for k in ("nvjet", "gemm", "splitKreduce"))


def is_sum(name):
    return "reduce_kernel" in name


def split_reps(ev):
    # each replay starts with the flush fill (largest fill); cut there
    reps, cur = [], []
    for s, d, n in ev:
        if "FillFunctor" in n and d > 30_000 and cur:  # 256 MB fill
            reps.append(cur)
            cur = []
        cur.append((s, d, n))
    reps.append(cur)
    return reps[1:-1]


def span(rep):
    """First kernel after the flush fill to the last GEMM kernel's end, us."""
    body = rep[1:]
    end = max(s + d for s, d, n in body if is_gemm(n))
    return (end - body[0][0]) / 1e3


def gemm_groups(rep):
    # contiguous runs of GEMM kernels at the end of the replay, in order
    groups, cur = [], []
    for s, d, n in rep:
        if is_gemm(n):
            cur.append((s, d))
        elif cur:
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    return [(g[-1][0] + g[-1][1] - g[0][0]) / 1e3 for g in groups]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100)
    a = ap.parse_args()
    el = lambda w: l2_prefetch(w, evict_last=True)  # noqa: E731
    el50 = lambda w: l2_prefetch(w, 0.5, evict_last=True)  # noqa: E731
    bn = lambda w: l2_prefetch(w)  # noqa: E731
    print("window A: o_proj (us, median); normal L2 policy; join after o_proj")
    cases = [("none", None, False)]
    for grid in (8, 32, 132):
        for post in (False, True):
            cases.append((f"{'post' if post else 'pre '} grid {grid}",
                          (lambda g: lambda w: l2_prefetch(w, grid=g, threads=32))(grid), post))
    for name, pf, post in cases:
        reps = split_reps(run(lambda: window_a(pf, post, True), a.reps))
        g = [gemm_groups(r)[-1] for r in reps]
        sums = [d / 1e3 for r in reps for s, d, n in r if is_sum(n)]
        pk = [d / 1e3 for r in reps for s, d, n in r if "prefetch" in n]
        sp = [span(r) for r in reps]
        print(f"  {name:16s} o_proj {st.median(g):6.2f}   accum read {st.median(sums):5.2f}"
              f"   window to o_proj end {st.median(sp):6.2f}"
              + (f"   prefetch kernel {st.median(pk):5.2f}" if pk else ""))
    return
    for moe in ("hbm", "c2c"):
        print(f"\nwindow B ({moe}-bound MoE): qkv_a, q_b (us, median)")
        for name, pf, which, frac in (("none", None, (), None),
                                      ("q_b only, evict_last", el, ("q_b",), None),
                                      ("q_b + 50% qkv_a", el, ("q_b", "qkv_a"), 0.5),
                                      ("q_b + 25% qkv_a", el, ("q_b", "qkv_a"), 0.25),
                                      ("both, evict_last", el, ("qkv_a", "q_b"), None)):
            reps = split_reps(run(lambda: window_b(pf, moe, which, frac), a.reps))
            gg = [gemm_groups(r) for r in reps]
            qkv = [g[-2] if len(g) > 1 else float("nan") for g in gg]
            qb = [g[-1] for g in gg]
            sums = [d / 1e3 for r in reps for s, d, n in r if is_sum(n)]
            extra = f"   MoE stream {st.median(sums):6.2f}" if sums else ""
            sp = [span(r) for r in reps]
            print(f"  {name:22s} qkv_a {st.median(qkv):6.2f}  q_b {st.median(qb):6.2f}{extra}"
                  f"   window {st.median(sp):6.2f}")


main()

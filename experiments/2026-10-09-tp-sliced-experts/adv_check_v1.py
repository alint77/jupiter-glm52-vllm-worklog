"""Adversarial correctness check of a layer-kernel variant: the four 512
slices of 52 checkpoint experts (40 hot, 12 cold) against fp32, over routings
the round-1 check never produces: sparse top-8 over many experts, every token
on the same experts (entries split at 8 tokens), one heavy-hitter expert,
cold-only and hot-only, -1 padded routes (including fully masked tokens),
shared-only, large / tiny / zero activations, odd token counts, workspace
reused across M and routing changes, and a CUDA graph replaying several
different calls back to back (PDL chain).
  adv_check.py --v td_v53[:FLAGS] [--reps N] [--seed S]"""
import argparse
import json
import sys

import torch

import kdev
from kdev import HIDDEN, TOPK

N_HOT, N_COLD = 40, 12
E = N_HOT + N_COLD
SSCALE = 0.4


def fixture(dev):
    import bench_slice as BS

    T = BS.T
    gen = torch.Generator().manual_seed(1)
    ck = T._int4_experts(E, gen)
    slices = [T._int4_marlin_tier(*BS.slice_ckpt(*ck, r), dev) for r in range(4)]

    def split(tier):
        hot = {k: v[:N_HOT].contiguous() for k, v in tier.items()}
        cold = T._to_grace({k: v[N_HOT:].contiguous() for k, v in tier.items()}, dev)
        return hot, cold

    sg = torch.Generator().manual_seed(7)
    sw13 = (torch.randn((4096, HIDDEN), generator=sg) * 0.02).to(torch.bfloat16)
    sw2 = (torch.randn((HIDDEN, 2048), generator=sg) * 0.02).to(torch.bfloat16)
    rows = lambda r: torch.cat([torch.arange(512 * r, 512 * r + 512),  # noqa: E731
                                2048 + torch.arange(512 * r, 512 * r + 512)])
    return {"tiers": [split(sl) for sl in slices],
            "w13": T._int4_dequant(ck[0], ck[2]).to(dev),
            "w2": T._int4_dequant(ck[1], ck[3]).to(dev),
            "sw13": sw13.to(dev), "sw2": sw2.to(dev),
            "sslices": [(sw13[rows(r)].contiguous().to(dev),
                         sw2[:, 512 * r:512 * r + 512].contiguous().to(dev))
                        for r in range(4)]}


def routing(kind, m, g):
    def pick(pool, k):
        return torch.stack([pool[torch.randperm(len(pool), generator=g)[:k]]
                            for _ in range(m)])

    allx = torch.arange(E)
    if kind == "sparse":
        ids = pick(allx, TOPK)
    elif kind == "same":  # every token on the same 8 experts
        ids = pick(allx, TOPK)[:1].expand(m, TOPK).clone()
    elif kind == "heavy":  # expert 3 in every token + random others
        rest = torch.cat([allx[:3], allx[4:]])
        ids = torch.cat([torch.full((m, 1), 3), pick(rest, TOPK - 1)], 1)
    elif kind == "cold":
        ids = pick(allx[N_HOT:], TOPK)
    elif kind == "hot":
        ids = pick(allx[:N_HOT], TOPK)
    elif kind == "masked":  # random -1 holes, token 0 fully masked
        ids = pick(allx, TOPK)
        ids[torch.rand((m, TOPK), generator=g) < 0.4] = -1
        ids[0] = -1
    elif kind == "shared_only":
        ids = torch.full((m, TOPK), -1)
    elif kind == "one_cold":  # a single cold route in the whole call
        ids = torch.full((m, TOPK), -1)
        ids[m - 1, 0] = N_HOT + 5
    elif kind == "one_hot":
        ids = torch.full((m, TOPK), -1)
        ids[0, TOPK - 1] = 17
    else:
        raise ValueError(kind)
    return ids.to(torch.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v", required=True)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = torch.device("cuda:0")
    f = fixture(dev)
    mod, _ = kdev.build(a.v)
    ver = kdev.version(a.v)
    rscale = 2.5 if ver >= 28 else 1.0
    ws = torch.zeros(mod.workspace_bytes(), dtype=torch.uint8, device=dev)
    e = torch.empty(0, device=dev)
    hm = torch.full((E,), -1, dtype=torch.int32, device=dev)
    cm = torch.full((E,), -1, dtype=torch.int32, device=dev)
    hm[:N_HOT] = torch.arange(N_HOT, dtype=torch.int32, device=dev)
    cm[N_HOT:] = torch.arange(N_COLD, dtype=torch.int32, device=dev)
    kf = {"w13": f["w13"], "w2": f["w2"], "sw13": f["sw13"], "sw2": f["sw2"]}

    def parts(t):
        return (t["w13_weight_packed"], t["w13_weight_scale"], t["w2_weight_packed"],
                t["w2_weight_scale"])

    def call(out, x, ids, wt, r, shared=True):
        hot, cold = f["tiers"][r]
        extra = (*f["sslices"][r], SSCALE) if shared else (None, None, 1.0)
        mod.forward(out, x, ids, wt, hm, cm, e, e, e, 0, False, *parts(hot),
                    *parts(cold), ws, True, e, *extra, rscale)

    def err_of(out, ref):
        return float((out.float() - ref).abs().max() / ref.abs().max().clamp_min(1e-30))

    g = torch.Generator().manual_seed(a.seed)
    kinds = ["sparse", "same", "heavy", "cold", "hot", "masked", "shared_only",
             "one_cold", "one_hot"]
    ms = [1, 2, 3, 5, 7, 8, 9, 13, 16, 17, 24, 31, 32]
    scales = {"normal": 0.3, "large": 30.0, "tiny": 3e-4}
    worst, fails, n = 0.0, 0, 0
    cases = []
    for m in ms:
        for kind in kinds:
            cases.append((m, kind, "normal", True))
        for sc in ("large", "tiny"):
            cases.append((m, "sparse", sc, True))
        cases.append((m, "sparse", "zero_rows", True))
        cases.append((m, "sparse", "normal", False))  # no shared expert
    # interleave M and routing so the workspace sees every transition
    order = torch.randperm(len(cases), generator=g).tolist()
    for rep in range(a.reps):
        for ci in order:
            m, kind, sc, shared = cases[ci]
            ids = routing(kind, m, g)
            x = torch.randn((m, HIDDEN), generator=g) * scales.get(sc, 0.3)
            if sc == "zero_rows":
                x[::2] = 0
            x = x.to(torch.bfloat16).to(dev)
            wt = torch.rand((m, TOPK), generator=g).to(dev)
            ids = ids.to(dev)
            for r in range(4):
                ref = kdev.reference(kf, x, ids, wt, shared, SSCALE, r, rscale)
                out = torch.full((m, HIDDEN), float("nan"), dtype=torch.bfloat16, device=dev)
                call(out, x, ids, wt, r, shared)
                if kind == "shared_only" and not shared:
                    continue
                err = err_of(out, ref)
                bad = not (err <= 5e-3) or bool(torch.isnan(out).any())
                if ref.abs().max() == 0:  # all routes masked, no shared: exact zero
                    bad = bool(out.float().abs().max() != 0)
                    err = float(out.float().abs().max())
                n += 1
                worst = max(worst, err if err == err else 1e9)
                if bad:
                    fails += 1
                    print(json.dumps({"rep": rep, "m": m, "kind": kind, "x": sc,
                                      "shared": shared, "slice": r, "err": err}), flush=True)
    # CUDA graph: several different calls replayed back to back (PDL chain)
    gcases = [(32, "sparse"), (1, "one_cold"), (8, "same"), (17, "masked"),
              (32, "heavy"), (5, "cold")]
    ins, outs, refs = [], [], []
    for m, kind in gcases:
        ids = routing(kind, m, g).to(dev)
        x = (torch.randn((m, HIDDEN), generator=g) * 0.3).to(torch.bfloat16).to(dev)
        wt = torch.rand((m, TOPK), generator=g).to(dev)
        ins.append((x, ids, wt))
        outs.append(torch.empty((m, HIDDEN), dtype=torch.bfloat16, device=dev))
        refs.append(kdev.reference(kf, x, ids, wt, True, SSCALE, 0, rscale))
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for (x, ids, wt), o in zip(ins, outs):
            call(o, x, ids, wt, 0)
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for (x, ids, wt), o in zip(ins, outs):
            call(o, x, ids, wt, 0)
    gfails = 0
    for it in range(50 * a.reps):
        for o in outs:
            o.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        for (m, kind), o, ref in zip(gcases, outs, refs):
            err = err_of(o, ref)
            n += 1
            worst = max(worst, err if err == err else 1e9)
            if not (err <= 5e-3):
                gfails += 1
                print(json.dumps({"graph_iter": it, "m": m, "kind": kind, "err": err}), flush=True)
    fails += gfails
    print(json.dumps({"v": a.v, "calls_checked": n, "worst_rel_err": round(worst, 5),
                      "fails": fails, "graph_fails": gfails, "pass": fails == 0}))
    sys.exit(0 if fails == 0 else 1)


if __name__ == "__main__":
    main()

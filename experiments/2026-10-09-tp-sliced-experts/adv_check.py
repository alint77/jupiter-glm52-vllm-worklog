"""Adversarial correctness check of a layer-kernel variant: the four 512
slices of 52 checkpoint experts (40 hot, 12 cold) against fp32, over routings
the round-1 check never produces: sparse top-8 over many experts, every token
on the same experts (entries split at 8 tokens), one heavy-hitter expert,
cold-only and hot-only, -1 padded routes (including fully masked tokens),
shared-only, large / tiny / zero activations, odd token counts, workspace
reused across M and routing changes, and a CUDA graph replaying several
different calls back to back (PDL chain).
Round 2 (Astra review 13): slot maps that permute the tiers, experts in
neither tier, an empty hot or cold tier, the padding argument, all routes
masked without the shared expert, concentrated cold routing, isolated routes
(one-hot weights, no shared), per-row checks (exact zero where the reference
row is zero), graph replays with new inputs every replay, and the ready epoch
wrapping past 2^32 with stale flags that alias the next epoch.
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
MAX_M = 32  # the variant's TD_MAX_TOKENS, set in main


def fixture(dev):
    import bench_slice as BS

    T = BS.T
    gen = torch.Generator().manual_seed(1)
    ck = T._int4_experts(E, gen)
    slices = [T._int4_marlin_tier(*BS.slice_ckpt(*ck, r), dev) for r in range(4)]

    def split(tier, ph=None, pc=None):
        ph = torch.arange(N_HOT) if ph is None else ph
        pc = torch.arange(N_COLD) if pc is None else pc
        hot = {k: v[:N_HOT][ph.to(v.device)].contiguous() for k, v in tier.items()}
        cold = T._to_grace({k: v[N_HOT:][pc.to(v.device)].contiguous()
                            for k, v in tier.items()}, dev)
        return hot, cold

    sg = torch.Generator().manual_seed(7)
    sw13 = (torch.randn((4096, HIDDEN), generator=sg) * 0.02).to(torch.bfloat16)
    sw2 = (torch.randn((HIDDEN, 2048), generator=sg) * 0.02).to(torch.bfloat16)
    rows = lambda r: torch.cat([torch.arange(512 * r, 512 * r + 512),  # noqa: E731
                                2048 + torch.arange(512 * r, 512 * r + 512)])
    pg_ = torch.Generator().manual_seed(3)
    ph, pc = torch.randperm(N_HOT, generator=pg_), torch.randperm(N_COLD, generator=pg_)
    return {"tiers": [split(sl) for sl in slices],
            "ptiers": [split(sl, ph, pc) for sl in slices], "ph": ph, "pc": pc,
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
    elif kind == "same_cold":  # every token on the same 8 cold experts
        ids = pick(allx[N_HOT:], TOPK)[:1].expand(m, TOPK).clone()
    elif kind == "isolate":  # sparse ids; the weights pick one route per token
        ids = pick(allx, TOPK)
    elif kind == "dup_cold":  # every route on one cold expert (not top-k)
        ids = torch.full((m, TOPK), N_HOT + 2)
    else:
        raise ValueError(kind)
    return ids.to(torch.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v", required=True)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stress", help="m,kind,scale,cfg,pad(0/1),shared(0/1),calls: one case with fresh inputs every call (random pad mask when pad=1)")
    a = ap.parse_args()
    dev = torch.device("cuda:0")
    f = fixture(dev)
    mod, _ = kdev.build(a.v)
    global MAX_M
    MAX_M = kdev.max_tokens(a.v)
    ver = kdev.version(a.v)
    rscale = 2.5 if ver >= 28 else 1.0
    ws = torch.zeros(mod.workspace_bytes(), dtype=torch.uint8, device=dev)
    e = torch.empty(0, device=dev)
    kf = {"w13": f["w13"], "w2": f["w2"], "sw13": f["sw13"], "sw2": f["sw2"]}

    def maps(hot_ids, cold_ids):
        hm = torch.full((E,), -1, dtype=torch.int32)
        cm = torch.full((E,), -1, dtype=torch.int32)
        hm[hot_ids] = torch.arange(len(hot_ids), dtype=torch.int32)
        cm[cold_ids] = torch.arange(len(cold_ids), dtype=torch.int32)
        return hm.to(dev), cm.to(dev)

    hot_all, cold_all = torch.arange(N_HOT), torch.arange(N_HOT, E)
    absent = torch.tensor([3, 17, 45])
    # name -> (tiers per slice, hot map, cold map, experts present here)
    configs = {
        "ident": (f["tiers"], *maps(hot_all, cold_all), torch.ones(E, dtype=torch.bool)),
        "perm": (f["ptiers"], *maps(f["ph"], N_HOT + f["pc"]),
                 torch.ones(E, dtype=torch.bool)),
    }
    hm, cm = maps(hot_all, cold_all)
    hm[absent[absent < N_HOT]] = -1
    cm[absent[absent >= N_HOT]] = -1
    pres = torch.ones(E, dtype=torch.bool)
    pres[absent] = False
    configs["absent"] = (f["tiers"], hm, cm, pres)
    empty = {k: e for k in f["tiers"][0][0]}
    hm, cm = maps(hot_all, cold_all)
    configs["nohot"] = ([(empty, c) for _, c in f["tiers"]], torch.full_like(hm, -1), cm,
                        torch.arange(E) >= N_HOT)
    configs["nocold"] = ([(h, empty) for h, _ in f["tiers"]], hm, torch.full_like(cm, -1),
                         torch.arange(E) < N_HOT)

    def parts(t):
        return (t["w13_weight_packed"], t["w13_weight_scale"], t["w2_weight_packed"],
                t["w2_weight_scale"])

    def call(out, x, ids, wt, r, shared=True, cfg="ident", pad=None):
        tiers, hm, cm, _ = configs[cfg]
        hot, cold = tiers[r]
        extra = (*f["sslices"][r], SSCALE) if shared else (None, None, 1.0)
        mod.forward(out, x, ids, wt, hm, cm, e, e, e, 0, False, *parts(hot),
                    *parts(cold), ws, True, e if pad is None else pad, *extra, rscale)

    def ref_of(x, ids, wt, r, shared, cfg="ident", pad=None):
        """The reference drops what the kernel must drop: experts absent here
        and padded tokens' routes."""
        keep = configs[cfg][3].to(dev)[ids.clamp_min(0).long()] & (ids >= 0)
        if pad is not None:
            keep &= ~pad[:, None]
        return kdev.reference(kf, x, torch.where(keep, ids, -1), wt, shared, SSCALE, r,
                              rscale)

    def err_of(out, ref):
        """Worst per-row error relative to the row's max |ref|; a zero
        reference row must come out exactly zero."""
        o = out.float()
        if bool(torch.isnan(o).any()):
            return float("nan")
        rmax = ref.abs().amax(1)
        d = (o - ref).abs().amax(1)
        zero = rmax == 0
        if bool((d[zero] != 0).any()):
            return float("inf")
        return float((d[~zero] / rmax[~zero]).max()) if bool((~zero).any()) else 0.0

    g = torch.Generator().manual_seed(a.seed)
    kinds = ["sparse", "same", "heavy", "cold", "hot", "masked", "shared_only",
             "one_cold", "one_hot", "same_cold"]
    ms = [1, 2, 3, 5, 7, 8, 9, 13, 16, 17, 24, 31, 32]
    if MAX_M >= 64:
        ms += [33, 40, 47, 48, 57, 63, 64]
    scales = {"normal": 0.3, "large": 30.0, "tiny": 3e-4}
    worst, fails, n = 0.0, 0, 0
    cases = []
    for m in ms:
        for kind in kinds:
            cases.append((m, kind, "normal", True))
        for sc in ("large", "tiny"):
            cases.append((m, "sparse", sc, True))
        cases.append((m, "sparse", "zero_rows", True))
        for kind in ("sparse", "shared_only", "isolate", "one_cold", "one_hot", "masked"):
            cases.append((m, kind, "normal", False))  # no shared expert
    cfg_names = ["ident", "ident", "perm", "absent", "nohot", "nocold"]
    by_cfg = {c: 0 for c in configs}

    def inputs(m, kind, sc="normal"):
        ids = routing(kind, m, g)
        x = torch.randn((m, HIDDEN), generator=g) * scales.get(sc, 0.3)
        if sc == "zero_rows":
            x[::2] = 0
        wt = torch.rand((m, TOPK), generator=g)
        if kind == "isolate":
            wt = torch.nn.functional.one_hot(torch.randint(0, TOPK, (m,), generator=g),
                                             TOPK).float()
        return x.to(torch.bfloat16).to(dev), ids.to(dev), wt.to(dev)

    if a.stress:
        m, kind, sc, cfg, pd, sh, calls = a.stress.split(",")
        m, pd, sh, calls = int(m), pd == "1", sh == "1", int(calls)
        bad = 0
        lib = getattr(mod, "lib", None)
        zero_bufs = []
        if lib is not None and hasattr(lib, "td_workspace_offsets"):
            import ctypes
            o = (ctypes.c_longlong * 7)()
            lib.td_workspace_offsets(o)
            zero_bufs = [(nm, o[2 * k], o[2 * k + 1]) for k, nm in enumerate(("y", "y13", "y13s"))]
            dbg_off = o[6]
        else:
            dbg_off = -1

        def dirty():
            torch.cuda.synchronize()
            return {nm: int((ws[off:off + n] != 0).sum()) for nm, off, n in zero_bufs
                    if bool((ws[off:off + n] != 0).any())}

        for i in range(calls):
            x, ids, wt = inputs(m, kind, sc)
            pad = (torch.rand(m, generator=g) < 0.3).to(dev) if pd else None
            r = i % 4
            out = torch.full((m, HIDDEN), float("nan"), dtype=torch.bfloat16, device=dev)
            before = dirty() if zero_bufs else {}
            call(out, x, ids, wt, r, sh, cfg, pad)
            after = dirty() if zero_bufs else {}
            ref = ref_of(x, ids, wt, r, sh, cfg, pad)
            err = err_of(out, ref)
            if dbg_off >= 0:
                dv = ws[dbg_off:dbg_off + 4 * (1 + 16 * 12)].view(torch.int32)
                nd = int(dv[0])
                if nd:
                    recs = dv[1:1 + 12 * min(nd, 16)].view(-1, 12).tolist()
                    print(json.dumps({"call": i, "dbg_n": nd, "err": err, "recs": recs,
                                      "keys": "cta warp kind s ph it tile chunk bad_row bad_ck bad_tok*8+ck lanes"}), flush=True)
                    ws[dbg_off:dbg_off + 4 * (1 + 16 * 12)] = 0
            if before or after or not (err <= 5e-3):
                bad += not (err <= 5e-3)
                o = out.float()
                d = (o - ref).abs()
                badm = (d > 5e-3 * ref.abs().amax(1, keepdim=True).clamp_min(1e-30)) | o.isnan()
                rows = badm.any(1).nonzero().flatten()
                cols = badm.any(0).nonzero().flatten()
                live = (ids >= 0) & configs[cfg][3].to(dev)[ids.clamp_min(0).long()]
                if pad is not None:
                    live &= ~pad[:, None]
                print(json.dumps({
                    "call": i, "err": err, "slice": r, "dirty_before": before, "dirty_after": after,
                    "bad_rows": rows.tolist(),
                    "bad_rows_padded": [] if pad is None else pad[rows].nonzero().flatten().tolist(),
                    "n_bad_cols": len(cols), "bad_cols_head": cols.tolist()[:12],
                    "bad_cols_tiles256": sorted(set((cols // 256).tolist()))[:24],
                    "bad_cols_mod256": sorted(set((cols % 256).tolist()))[:40],
                    "n_bad": int(badm.sum()), "max_abs_out": float(o[badm].abs().max()) if bool(badm.any()) else 0,
                    "live_routes": int(live.sum()), "pads": 0 if pad is None else int(pad.sum())}), flush=True)
                torch.save({"x": x.cpu(), "ids": ids.cpu(), "wt": wt.cpu(), "pad": None if pad is None else pad.cpu(),
                            "out": out.cpu(), "ref": ref.cpu(), "slice": r, "cfg": cfg, "shared": sh},
                           f"logs/fail-{a.seed}-{i}.pt")
        print(json.dumps({"v": a.v, "stress": a.stress, "fails": bad}), flush=True)
        return

    # interleave M, routing and map configs so the workspace sees every transition
    order = torch.randperm(len(cases), generator=g).tolist()
    for rep in range(a.reps):
        for ci in order:
            m, kind, sc, shared = cases[ci]
            cfg = cfg_names[int(torch.randint(0, len(cfg_names), (1,), generator=g))]
            x, ids, wt = inputs(m, kind, sc)
            pad = None
            if float(torch.rand(1, generator=g)) < 0.25:
                pad = (torch.rand(m, generator=g) < 0.3).to(dev)
            by_cfg[cfg] += 1
            for r in range(4):
                ref = ref_of(x, ids, wt, r, shared, cfg, pad)
                out = torch.full((m, HIDDEN), float("nan"), dtype=torch.bfloat16, device=dev)
                call(out, x, ids, wt, r, shared, cfg, pad)
                err = err_of(out, ref)
                n += 1
                worst = max(worst, err if err == err and err != float("inf") else 1e9)
                if not (err <= 5e-3):
                    fails += 1
                    print(json.dumps({"rep": rep, "m": m, "kind": kind, "x": sc, "cfg": cfg,
                                      "pad": pad is not None, "shared": shared, "slice": r,
                                      "err": err}), flush=True)
    # every route of every token on one cold expert: outside top-k's contract
    # (no repeated ids per token), reported but not part of the verdict
    dup = []
    for m in (1, 8, 32, MAX_M):
        x, ids, wt = inputs(m, "dup_cold")
        out = torch.empty((m, HIDDEN), dtype=torch.bfloat16, device=dev)
        call(out, x, ids, wt, 0)
        dup.append(err_of(out, ref_of(x, ids, wt, 0, True)))

    # CUDA graph: different calls replayed back to back (PDL chain), with new
    # inputs before every replay; shared on/off and an empty call in the chain
    gcases = [(32, "sparse", True), (1, "one_cold", True), (8, "same", True),
              (17, "masked", False), (4, "shared_only", False), (32, "heavy", True),
              (5, "cold", False), (9, "isolate", False)]
    if MAX_M >= 64:
        gcases += [(64, "heavy", True), (41, "masked", False), (64, "sparse", False)]
    bufs, outs = [], []
    for m, kind, sh in gcases:
        bufs.append(inputs(m, kind))
        outs.append(torch.empty((m, HIDDEN), dtype=torch.bfloat16, device=dev))
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for (x, ids, wt), o, (_, _, sh) in zip(bufs, outs, gcases):
            call(o, x, ids, wt, 0, sh)
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for (x, ids, wt), o, (_, _, sh) in zip(bufs, outs, gcases):
            call(o, x, ids, wt, 0, sh)
    gfails = 0
    for it in range(20 * a.reps):
        refs = []
        for (x, ids, wt), o, (m, kind, sh) in zip(bufs, outs, gcases):
            nx, nids, nwt = inputs(m, kind)
            x.copy_(nx)
            ids.copy_(nids)
            wt.copy_(nwt)
            o.fill_(float("nan"))
            refs.append(ref_of(x, ids, wt, 0, sh))
        graph.replay()
        torch.cuda.synchronize()
        for (m, kind, sh), o, ref in zip(gcases, outs, refs):
            err = err_of(o, ref)
            n += 1
            worst = max(worst, err if err == err and err != float("inf") else 1e9)
            if not (err <= 5e-3):
                gfails += 1
                print(json.dumps({"graph_iter": it, "m": m, "kind": kind, "err": err}),
                      flush=True)
    fails += gfails

    # epoch wrap: find the epoch word (the only one an empty call bumps) and
    # the ready flags (bumped by a repeated busy call), then set the epoch to
    # 2^32 - 1 and every flag to 0 (what a plain increment wraps to) or 1 (what
    # skipping 0 without clearing would alias)
    w32 = ws.view(torch.int32)
    xa, ia, wa = inputs(32, "sparse")
    out = torch.empty((32, HIDDEN), dtype=torch.bfloat16, device=dev)
    call(out, xa, ia, wa, 0)
    w0 = w32.clone()
    call(out, xa, ia, wa, 0)
    w1 = w32.clone()
    xe, ie, we = inputs(4, "shared_only")
    call(torch.empty((4, HIDDEN), dtype=torch.bfloat16, device=dev), xe, ie, we, 0, False)
    w2 = w32.clone()
    ep = (w2 == w1 + 1).nonzero().flatten()
    assert len(ep) == 1 and int(w1[ep[0]]) == int(w0[ep[0]]) + 1, len(ep)
    # ready flags: bumped by the repeated call and equal to its epoch
    flags = ((w1 == w0 + 1) & (w1 == w1[ep[0]])).nonzero().flatten()
    wfails = 0
    for trial in range(10):
        w32[flags] = trial % 2
        w32[ep] = -1
        for i in range(3):  # the wrapping call, then two ordinary ones
            x, ids, wt = inputs(32, "sparse")
            out = torch.full((32, HIDDEN), float("nan"), dtype=torch.bfloat16, device=dev)
            call(out, x, ids, wt, 0)
            err = err_of(out, ref_of(x, ids, wt, 0, True))
            n += 1
            worst = max(worst, err if err == err and err != float("inf") else 1e9)
            if not (err <= 5e-3):
                wfails += 1
                print(json.dumps({"wrap_trial": trial, "call": i, "err": err,
                                  "epoch": int(w32[ep[0]])}), flush=True)
        # the wrapping call must skip epoch 0
        if ver >= 55 and int(w32[ep[0]]) != 3:
            wfails += 1
            print(json.dumps({"wrap_trial": trial, "epoch_after": int(w32[ep[0]])}))
    fails += wfails
    # ids past the maps (num_experts .. 511, and past MAX_EXPERTS) are dropped
    ofails = 0
    if ver >= 55:
        for m in (1, 8, 32):
            x, ids, wt = inputs(m, "sparse")
            bad = torch.rand(ids.shape, generator=g).to(dev) < 0.3
            junk = torch.where(torch.rand(ids.shape, generator=g).to(dev) < 0.5, E + 3, 700)
            ids_bad = torch.where(bad, junk.to(ids.dtype), ids)
            out = torch.empty((m, HIDDEN), dtype=torch.bfloat16, device=dev)
            call(out, x, ids_bad, wt, 0)
            err = err_of(out, ref_of(x, torch.where(bad, -1, ids), wt, 0, True))
            n += 1
            worst = max(worst, err if err == err and err != float("inf") else 1e9)
            if not (err <= 5e-3):
                ofails += 1
                print(json.dumps({"oob_m": m, "err": err}), flush=True)
    fails += ofails
    print(json.dumps({"v": a.v, "oob_fails": ofails, "calls_checked": n, "worst_rel_err": round(worst, 5),
                      "fails": fails, "graph_fails": gfails, "wrap_fails": wfails,
                      "cases_by_cfg": by_cfg,
                      "dup_cold_err": [round(d, 5) for d in dup], "pass": fails == 0}))
    sys.exit(0 if fails == 0 else 1)


if __name__ == "__main__":
    main()

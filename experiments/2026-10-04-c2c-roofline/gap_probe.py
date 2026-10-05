#!/usr/bin/env python3
"""Where the tiered decode MoE call loses link time: per-call timeline of the
five kernels from the TD_CTA_TRACE probe build, cold-only and mixed cells.

Records (td_record, %globaltimer ns), word0 phase:
  0/1 w13/w2 CTA {entry, ready, exit}; 10/11 empty CTA; 50 route_prep,
  53/63 act (live/dead route), 54 finalize {entry, past PDL wait, exit};
  60/61 producer {first issue, past PDL wait (w2), last issue};
  70/71 consumer warp 0 {first stage full, last stage full, exit}.
Calls are replayed back to back in one CUDA graph (20 calls), as in bench_fmt.
Run with VLLM_TIERED_DECODE_DEFINES=TD_CTA_TRACE, NUMA-bound:
    gap_probe.py --grid 0,1 0,2 9,2 --out dir
"""
import argparse
import importlib.util
import json
import os
import statistics as st
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "bench_fmt", HERE.parent / "2026-09-27-glm53-mtp7-profile/bench_fmt.py")
BF = importlib.util.module_from_spec(spec)
spec.loader.exec_module(BF)
B = BF.B
EXPERT = 21_233_672
W13 = 4096 * 6144 // 2 + 6144 // 32 * 4096 * 2
W2 = EXPERT - W13
LINK = 421e9


def capture(x, calls, hot, cold):
    from vllm.model_executor.layers.fused_moe.tiered_decode import tiered_decode_moe
    for c in calls[:2]:
        tiered_decode_moe(x, *c, hot, cold)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(20):
            tiered_decode_moe(x, *calls[i % len(calls)], hot, cold)
    return g


def split_calls(recs):
    rows = [dict(ph=w & 0xFF, blk=(w >> 8) & 0xFF, nh=(w >> 32) & 0xFFFF,
                 nc=(w >> 48) & 0xFFFF, t0=a, t1=b, t2=c) for w, a, b, c in recs.tolist()]
    rp = sorted(r["t1"] for r in rows if r["ph"] == 50)
    starts = [t for i, t in enumerate(rp) if i == 0 or t - rp[i - 1] > 5000]
    calls = [[] for _ in starts]
    for r in rows:
        key = r["t0"] if r["ph"] in (60, 61, 70, 71) else r["t1"]
        k = max((i for i, s in enumerate(starts) if s <= key), default=None)
        if k is not None:
            calls[k].append(r)
    return [(starts[i], starts[i + 1], calls[i]) for i in range(len(starts) - 1)]


def timeline(start, nxt, rs, nc):
    by = lambda *ph: [r for r in rs if r["ph"] in ph]
    us = lambda t: (t - start) / 1e3
    rp, w13, w2 = by(50), by(0, 10), by(1, 11)
    p13, p2, c13, c2 = by(60), by(61), by(70), by(71)
    act, fin = by(53, 63), by(54)
    if not (rp and w13 and w2 and fin and c13 and c2):
        return None
    t = {
        "rp_end": max(r["t2"] for r in rp),
        "w13_ready": min(r["t1"] for r in w13 if r["ph"] == 0),
        "w13_issue": min(r["t0"] for r in p13),
        "w13_data0": min(r["t0"] for r in c13),
        "w13_data1": max(r["t1"] for r in c13),
        "w13_end": max(r["t2"] for r in w13),
        "act_ready": min(r["t1"] for r in act),
        "act_end": max(r["t2"] for r in act),
        "w2_issue": min(r["t0"] for r in p2),
        "w2_rows": min(r["t1"] for r in p2 if r["t1"]) if any(r["t1"] for r in p2) else 0,
        "w2_data0": min(r["t0"] for r in c2),
        "w2_data1": max(r["t1"] for r in c2),
        "w2_end": max(r["t2"] for r in w2),
        "fin_ready": min(r["t1"] for r in fin),
        "fin_end": max(r["t2"] for r in fin),
        "next": nxt,
    }
    out = {k: us(v) for k, v in t.items()}
    out["cold_w13_data1"] = us(max(r["t1"] for r in c13 if r["blk"] < (16 if nc == 1 else 24))) if nc else 0
    out["cold_w2_data1"] = us(max(r["t1"] for r in c2 if r["blk"] < (16 if nc == 1 else 24))) if nc else 0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", nargs="+", default=["0,1", "0,2", "0,3", "0,4", "9,2"])
    ap.add_argument("--out", default=str(HERE / "gap"))
    args = ap.parse_args()
    from vllm.model_executor.layers.fused_moe.tiered_decode import _extension
    ext = _extension()
    assert hasattr(ext, "td_dump"), "set VLLM_TIERED_DECODE_DEFINES=TD_CTA_TRACE"
    dev = torch.device("cuda:0")
    gen = torch.Generator().manual_seed(0)
    hot, _ = BF.tier("int4", 16, dev, False, 0, gen, 1000)
    cold, keep = BF.tier("int4", 8, dev, True, 0, gen, 100)
    x = (torch.randn((B.TOKENS, B.HIDDEN), generator=gen) * 0.3).to(torch.bfloat16).to(dev)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    for cell in args.grid:
        h, c = map(int, cell.split(","))
        calls = [B.routing(h, c, 1000, 100, r, gen, dev) for r in range(20)]
        for ids, w, hmap, cmap in calls:
            on = hmap >= 0
            hmap[on] = torch.randint(0, 1000, (int(on.sum()),), device=dev, dtype=hmap.dtype)
            on = cmap >= 0
            cmap[on] = torch.randint(0, 100, (int(on.sum()),), device=dev, dtype=cmap.dtype)
        g = capture(x, calls, hot, cold)
        for _ in range(30):
            g.replay()
        torch.cuda.synchronize()
        ext.td_dump()
        for _ in range(5):
            g.replay()
        recs = ext.td_dump()
        torch.save(recs, f"{args.out}/h{h}c{c}.pt")
        tl = [t for t in (timeline(*cl, c) for cl in split_calls(recs)) if t]
        med = {k: round(st.median(t[k] for t in tl), 2) for k in tl[0]}
        floor = c * EXPERT / LINK * 1e6
        print(json.dumps({"hot": h, "cold": c, "calls": len(tl), "floor_us": round(floor, 1),
                          "w13_floor_us": round(c * W13 / LINK * 1e6, 1),
                          "w2_floor_us": round(c * W2 / LINK * 1e6, 1), **med}), flush=True)
    del keep


if __name__ == "__main__":
    main()

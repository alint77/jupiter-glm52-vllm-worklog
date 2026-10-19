"""Replay a top-k dump (rank 0) through anchor_hot.cu for one anchor layer.

Source = a pinned host tensor of S x 656 random bytes (the anchor's main cache
on Grace, read through UVA). Per decode step: copy the step's converted
indices and new-token slots into static buffers, run the map (eager, or one
CUDA graph replay with --graph), then
  check   every entry: -1 -> -1, new token j -> N + j, else the buffer row
          holds exactly the source row's bytes; overflow stays 0; the
          where/slot_of tables stay inverse of each other (every --audit steps)
  time    torch profiler over the replays (--profile): per launch kernel time
          and per step span (first kernel start to last end)

  replay.py <dump dir> [--layer 0] [--rows 16384] [--steps 0] [--graph]
            [--profile 2000] [--check 1]
"""

import argparse
import re
import json
import os
import statistics as stt
import sys
from pathlib import Path

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile
from torch.utils.cpp_extension import load

HERE = Path(__file__).parent
ROW = 656
RES = 64
MAXE = 64 * 768
KPS = 1  # launches per step
OVF = 6  # ctl index of the overflow counter


def build(prof=False):
    os.environ.setdefault(
        "TORCH_EXTENSIONS_DIR", f"/e/fscratch/profound/{os.environ['USER']}/torch_ext_ahc"
    )
    return load(
        name="anchor_hot_prof" if prof else "anchor_hot",
        sources=[str(HERE / "anchor_hot.cu")],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo"]
        + (["-DAHC_PROF"] if prof else []),
        verbose=False,
    )


def steps(path, layer):
    idx = [json.loads(line) for line in open(os.path.join(path, "index.jsonl"))]
    data = np.memmap(os.path.join(path, "topk.bin"), dtype=np.int32, mode="r")
    for e in idx:
        if any(r.startswith("_warmup") for r in e["req_ids"]):
            continue
        L, t, w = len(e["layers"]), e["t"], e["w"]
        o = e["off"] // 4
        s = np.asarray(data[o + layer * t * w : o + (layer + 1) * t * w]).reshape(t, w)
        o += L * t * w
        yield t, w, s, np.asarray(data[o : o + t])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--rows", type=int, default=16384)
    ap.add_argument("--slots", type=int, default=262144)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--graph", action="store_true")
    ap.add_argument("--profile", type=int, default=0, help="profile this many steps")
    ap.add_argument("--check", type=int, default=1, help="check every k-th step (0 off)")
    ap.add_argument("--audit", type=int, default=500)
    ap.add_argument("--prof", action="store_true", help="in-kernel sweep timers")
    a = ap.parse_args()
    ext = build(a.prof)
    dev = torch.device("cuda")
    N, S = a.rows, a.slots
    g = torch.Generator().manual_seed(0)
    src = torch.randint(0, 256, (S, ROW), dtype=torch.uint8, generator=g).pin_memory()
    src_dev_check = src  # host copy for checks
    probe = torch.empty(1 << 30, dtype=torch.uint8).pin_memory()
    pd = torch.empty_like(probe, device=dev)
    pd.copy_(probe, non_blocking=True)
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(4):
        pd.copy_(probe, non_blocking=True)
    t1.record()
    torch.cuda.synchronize()
    print(f"C2C host->HBM {4 * (1 << 30) / t0.elapsed_time(t1) / 1e6:.0f} GB/s")
    del probe, pd
    dst = torch.zeros((N + RES) * ROW, dtype=torch.uint8, device=dev)
    where = torch.full((S,), -1, dtype=torch.int32, device=dev)
    slot_of = torch.full((N,), -1, dtype=torch.int32, device=dev)
    status = torch.zeros(N, dtype=torch.int32, device=dev)
    ctl = torch.zeros(20, dtype=torch.int32, device=dev)
    ctl[10:12] = -1  # AHC_PROF: min start time (u64)
    miss = torch.zeros(MAXE, dtype=torch.int32, device=dev)
    miss_row = torch.zeros(MAXE, dtype=torch.int32, device=dev)
    graphs = {}
    bufs = {}

    def get(t, w):
        if (t, w) not in bufs:
            idx = torch.full((t, w), -1, dtype=torch.int32, device=dev)
            new = torch.full((t,), -1, dtype=torch.int32, device=dev)
            out = torch.full((t, w), -1, dtype=torch.int32, device=dev)
            bufs[(t, w)] = (idx, new, out)
        return bufs[(t, w)]

    def run(t, w):
        idx, new, out = get(t, w)
        ext.anchor_hot_map(idx, new, out, src, dst, where, slot_of, status, ctl,
                           miss, miss_row)

    def replay(t, w):
        if not a.graph:
            run(t, w)
            return
        if (t, w) not in graphs:
            # warm-up on a scratch state would perturb the real one: capture
            # directly (the map has no lazy init); the capture itself does not
            # execute, so replay once for this step
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                run(t, w)
            graphs[(t, w)] = gr
        graphs[(t, w)].replay()

    nsteps = nmiss = nrows_read = 0
    bad = 0
    prof_steps = []
    prof = None
    hostdst = None
    for n, (t, w, s, nw) in enumerate(steps(a.dump, a.layer)):
        if a.steps and n >= a.steps:
            break
        idx, new, out = get(t, w)
        idx.copy_(torch.from_numpy(s.copy()))
        new.copy_(torch.from_numpy(nw.copy()))
        before = int(ctl[OVF].item()) if a.check else 0
        if a.profile and prof is None and n >= 200:
            prof = profile(activities=[ProfilerActivity.CUDA])
            prof.__enter__()
        replay(t, w)
        if prof is not None:
            prof_steps.append(t)
            if len(prof_steps) >= a.profile:
                torch.cuda.synchronize()
                prof.__exit__(None, None, None)
                report(prof, prof_steps)
                prof = None
                a.profile = 0
        nsteps += 1
        if a.check and n % a.check == 0:
            torch.cuda.synchronize()
            o = out.cpu().numpy()
            sv = s
            u = np.unique(sv[sv >= 0])
            u = u[~np.isin(u, nw[nw >= 0])]
            nrows_read += len(u)
            hostdst = dst.view(N + RES, ROW).cpu().numpy()
            srcn = src_dev_check.numpy()
            nwv = nw[nw >= 0]
            valid = sv >= 0
            isnew = valid & np.isin(sv, nwv)
            ok = np.where(valid, True, o == -1)
            if isnew.any():
                last = {int(v): N + j for j, v in enumerate(nw) if v >= 0}
                exp = np.vectorize(lambda v: last[int(v)])(sv[isnew])
                ok[isnew] = o[isnew] == exp
            old = valid & ~isnew
            r = o[old]
            inr = (r >= 0) & (r < N)
            same = np.zeros(len(r), bool)
            same[inr] = np.all(hostdst[r[inr]] == srcn[sv[old][inr]], axis=1)
            ok[old] = same
            if not ok.all():
                bad += int((~ok).sum())
                jj, kk = np.nonzero(~ok)
                for j, k in list(zip(jj, kk))[:5]:
                    print(f"step {n} entry ({j},{k}) slot {sv[j, k]} -> row {o[j, k]}",
                          file=sys.stderr)
            if int(ctl[OVF].item()) != before:
                print(f"step {n}: overflow {int(ctl[OVF].item())}", file=sys.stderr)
        if a.audit and n % a.audit == 0:
            wh = where.cpu().numpy()
            so = slot_of.cpu().numpy()
            rows = np.nonzero(wh >= 0)[0]
            inv = np.all(so[wh[rows]] == rows)
            filled = so >= 0
            inv2 = np.all(wh[so[filled]] == np.nonzero(filled)[0])
            if not (inv and inv2) or np.any(wh == -2):
                print(f"step {n}: table audit failed inv={inv} inv2={inv2} "
                      f"claimed={int(np.sum(wh == -2))}", file=sys.stderr)
                bad += 1
    torch.cuda.synchronize()
    c = ctl.cpu().tolist()
    print(f"claims {c[8]} = {c[8] / max(nsteps, 1):.1f}/step")
    print(f"steps {nsteps} bad {bad} overflow {c[OVF]} swept {c[7]} step {c[0]}")
    if a.prof:
        k = max(nsteps, 1)
        print(f"per step (ns): first block start -> last block {c[13] / k:.0f}, "
              f"sweep {c[15] / k:.0f} (round-0 scan -> end {c[12] / k:.0f}), "
              f"rounds {c[14] / k:.2f}")
        print(f"  sweep parts (ns): preload+hand {c[17] / k:.0f}, candidate loads "
              f"{c[18] / k:.0f}, scan {c[19] / k:.0f}, scan->end {c[12] / k:.0f}")


def report(prof, tsteps):
    ev = [e for e in prof.events() if e.device_type.name == "CUDA"]
    names = {}
    for e in ev:
        mm = re.search(r"(\w+_kernel)", e.name)
        names.setdefault(mm.group(1) if mm else e.name[:40], []).append(e.device_time)
    for k, v in sorted(names.items()):
        print(f"  {k:40s} n={len(v):6d} median {stt.median(v):7.2f} us  p90 "
              f"{np.percentile(v, 90):7.2f}")
    # per step span: kernels in groups of KPS in launch order
    ks = sorted((e.time_range.start, e.time_range.end, e.name) for e in ev
                if "kernel" in e.name)
    spans = collections_by_step(ks, len(tsteps))
    for m in sorted(set(tsteps)):
        sp = [s for s, t in zip(spans, tsteps) if t == m]
        if sp:
            print(f"  M={m:2d} steps {len(sp):5d} span median {stt.median(sp):7.2f} us  "
                  f"p90 {np.percentile(sp, 90):7.2f}  p99 {np.percentile(sp, 99):7.2f}")


def collections_by_step(ks, nsteps):
    spans = []
    for i in range(0, min(len(ks), KPS * nsteps), KPS):
        grp = ks[i : i + KPS]
        spans.append(grp[-1][1] - grp[0][0])
    return spans


if __name__ == "__main__":
    main()

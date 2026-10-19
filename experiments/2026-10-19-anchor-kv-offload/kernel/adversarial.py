"""Adversarial and concurrency tests for anchor_hot.cu (synthetic steps).

Every step is checked against the source (as in replay.py): -1 -> -1, this
step's token j -> N + j, any other slot -> a hot row holding exactly the
source row's current bytes; overflow stays 0; where/slot_of are inverse and
no slot stays claimed. Between steps the source rows of the step's new
tokens are rewritten (the KV write), so a stale hot row shows up as wrong
bytes.

  adversarial.py [--graph] [--stress]
"""

import argparse
import sys
import threading

import numpy as np
import torch

from replay import ROW, RES, build

N = 16384
S = 65536
MAXE = 64 * 768


class Cache:
    def __init__(self, ext, dev, graph):
        self.ext, self.dev, self.graph = ext, dev, graph
        self.g = torch.Generator().manual_seed(1)
        self.src = torch.randint(0, 256, (S, ROW), dtype=torch.uint8, generator=self.g).pin_memory()
        self.dst = torch.zeros((N + RES) * ROW, dtype=torch.uint8, device=dev)
        self.where = torch.full((S,), -1, dtype=torch.int32, device=dev)
        self.slot_of = torch.full((N,), -1, dtype=torch.int32, device=dev)
        self.status = torch.zeros(N, dtype=torch.int32, device=dev)
        self.ctl = torch.zeros(32, dtype=torch.int32, device=dev)
        self.ctl[10:12] = -1
        self.miss = torch.zeros(MAXE, dtype=torch.int32, device=dev)
        self.miss_row = torch.zeros(MAXE, dtype=torch.int32, device=dev)
        self.bufs, self.graphs = {}, {}
        self.steps = self.bad = 0
        self._claims = self._swept = 0

    def run(self, n, t, stream=None):
        idx, new, out = self.buf(n, t)
        self.ext.anchor_hot_map(idx, new, out, self.src, self.dst, self.where, self.slot_of,
                                self.status, self.ctl, self.miss, self.miss_row)

    def buf(self, n, t):
        if (n, t) not in self.bufs:
            self.bufs[(n, t)] = (torch.full((n,), -1, dtype=torch.int32, device=self.dev),
                                 torch.full((t,), -1, dtype=torch.int64, device=self.dev),
                                 torch.full((n,), -1, dtype=torch.int32, device=self.dev))
        return self.bufs[(n, t)]

    def step(self, slots, new, name):
        """slots: int32 [n] (-1 allowed); new: int32 [t] (-1 allowed)."""
        n, t = len(slots), len(new)
        idx, nw, out = self.buf(n, t)
        idx.copy_(torch.from_numpy(slots.astype(np.int32)))
        nw.copy_(torch.from_numpy(new.astype(np.int64)))
        if self.graph:
            if (n, t) not in self.graphs:
                gr = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gr):
                    self.run(n, t)
                self.graphs[(n, t)] = gr
            self.graphs[(n, t)].replay()
        else:
            self.run(n, t)
        torch.cuda.synchronize()
        self.steps += 1
        o = out.cpu().numpy()
        hd = self.dst.view(N + RES, ROW).cpu().numpy()
        sr = self.src.numpy()
        # expected reserved row per slot: N + the last j with new[j] == s
        resv = np.full(S, -1, np.int64)
        for j, v in enumerate(new):
            if v >= 0:
                resv[v] = N + j
        valid = slots >= 0
        isnew = valid & (resv[np.where(valid, slots, 0)] >= 0)
        ok = np.where(valid, True, o == -1)
        ok[isnew] = o[isnew] == resv[slots[isnew]]
        old = valid & ~isnew
        r = o[old]
        inr = (r >= 0) & (r < N)
        same = np.zeros(len(r), bool)
        same[inr] = np.all(hd[r[inr]] == sr[slots[old][inr]], axis=1)
        ok[old] = same
        c = self.ctl.cpu().tolist()
        if name != "random":
            print(f"  {name}: claims {c[8] - self._claims}, rows passed {c[7] - self._swept}")
        self._claims, self._swept = c[8], c[7]
        wh = self.where.cpu().numpy()
        so = self.slot_of.cpu().numpy()
        rows = np.nonzero(wh >= 0)[0]
        filled = so >= 0
        inv = (np.all(so[wh[rows]] == rows) and np.all(wh[so[filled]] == np.nonzero(filled)[0])
               and not np.any(wh == -2))
        if not ok.all() or c[6] or not inv:
            self.bad += 1
            print(f"FAIL {name} step {self.steps}: bad entries {int((~ok).sum())} "
                  f"overflow {c[6]} inverse {inv}", file=sys.stderr)
        # the KV write of this step's tokens changes their source rows
        nv = new[new >= 0]
        if len(nv):
            self.src[torch.from_numpy(nv.astype(np.int64))] = torch.randint(
                0, 256, (len(nv), ROW), dtype=torch.uint8, generator=self.g)
        return c

    def claims(self):
        return int(self.ctl[8].item())


def scenarios(cache, rng):
    none = np.full(16, -1, np.int32)
    # cold start at full capacity: n = N distinct slots, every row claimed
    a = rng.choice(S, N, replace=False).astype(np.int32)
    cache.step(a, none, "cold-full")
    # every row read this step (stamped, ref set)
    cache.step(rng.permutation(a), none, "all-hit")
    # N new slots: lap 1 clears every ref bit, lap 2 takes every row
    b = rng.choice(np.setdiff1d(np.arange(S), a), N, replace=False).astype(np.int32)
    cache.step(b, none, "all-ref-turnover")
    # half hits (protected), half misses (must take the other half only)
    keep = rng.choice(b, N // 2, replace=False)
    fresh = rng.choice(np.setdiff1d(np.arange(S), b), N // 2, replace=False)
    cache.step(rng.permutation(np.concatenate([keep, fresh])).astype(np.int32), none, "half")
    # heavy multiplicity and -1: 12288 entries over 40 slots
    few = rng.choice(S, 40, replace=False)
    d = rng.choice(few, 12288).astype(np.int32)
    d[rng.random(12288) < 0.3] = -1
    cache.step(d, none, "dups")
    # new tokens that are cached and also selected, then read again next step
    new = rng.choice(few, 16, replace=False).astype(np.int32)
    new[3] = -1  # a token another DCP rank owns
    cache.step(np.concatenate([d[:4096], new, new]).astype(np.int32), new, "new-cached")
    cache.step(np.concatenate([d[:4096], new]).astype(np.int32), none, "after-new")
    # 64 new tokens, one block (grid of 1)
    new64 = rng.choice(S, 64, replace=False).astype(np.int32)
    cache.step(np.concatenate([new64, rng.choice(S, 960)]).astype(np.int32), new64, "t64-alone")
    # random traffic with locality, sizes from 1 to 16 query tokens
    hot = rng.choice(S, 6000, replace=False)
    for _ in range(3000):
        m = int(rng.integers(1, 17))
        n = m * 768
        s = np.where(rng.random(n) < 0.8, rng.choice(hot, n), rng.integers(0, S, n)).astype(np.int32)
        s[rng.random(n) < 0.3] = -1
        t = int(rng.integers(0, min(m * 4, 64) + 1))
        nw = rng.choice(S, t, replace=False).astype(np.int32) if t else np.zeros(0, np.int32)
        if t:
            nw[rng.random(t) < 0.25] = -1
            s[: t] = nw
        if rng.random() < 0.02:
            hot = rng.choice(S, 6000, replace=False)  # request change: cold rows
        cache.step(s, nw, "random")


def stress(ext, dev):
    """Map launches on one stream while another stream keeps every SM busy.

    The barrier needs all of the map's blocks to start; the hog's blocks
    finish independently, so the map must always complete. A hang shows up as
    the watchdog firing.
    """
    cache = Cache(ext, dev, False)
    rng = np.random.default_rng(7)
    hog = torch.cuda.Stream()
    mp = torch.cuda.Stream()
    x = torch.randn(8192, 8192, device=dev, dtype=torch.bfloat16)
    done = threading.Event()

    def watchdog():
        if not done.wait(300):
            print("FAIL stress: no progress in 300 s (deadlock?)", file=sys.stderr, flush=True)
            import os
            os._exit(2)

    threading.Thread(target=watchdog, daemon=True).start()
    hot = rng.choice(S, 6000, replace=False)
    calls = 0
    for it in range(400):
        with torch.cuda.stream(hog):
            for _ in range(4):
                y = x @ x
        with torch.cuda.stream(mp):
            for _ in range(8):
                n = 16 * 768
                s = rng.choice(hot, n).astype(np.int32)
                idx, nw, out = cache.buf(n, 16)
                idx.copy_(torch.from_numpy(s), non_blocking=False)
                cache.run(n, 16)
                calls += 1
        if it % 50 == 0:
            torch.cuda.synchronize()
    torch.cuda.synchronize()
    done.set()
    del y
    print(f"stress: {calls} map launches alongside {400 * 4} 8192^3 GEMMs, no hang, "
          f"overflow {int(cache.ctl[6].item())}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", action="store_true")
    ap.add_argument("--stress", action="store_true")
    a = ap.parse_args()
    ext = build()
    dev = torch.device("cuda")
    if a.stress:
        stress(ext, dev)
        return
    cache = Cache(ext, dev, a.graph)
    scenarios(cache, np.random.default_rng(0))
    print(f"{'graph' if a.graph else 'eager'}: {cache.steps} steps, {cache.claims()} claims, "
          f"{cache.bad} failing steps, rounds-heavy steps swept {int(cache.ctl[7].item())}")
    sys.exit(1 if cache.bad else 0)


if __name__ == "__main__":
    main()

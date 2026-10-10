"""Convert a kdev unit trace (.pt, TD_UNIT_TRACE TD_CTA_TRACE) into a Chrome /
Perfetto trace JSON: one process per CTA, threads 'consumer w0' and
'producer'. Times in us from route_prep entry.   to_perfetto.py t.pt out.json"""
import json
import sys

import numpy as np
import torch

r = torch.load(sys.argv[1]).numpy().astype(np.int64)
ph = r[:, 0] & 0xFF
st = np.sort(r[ph == 50, 1])
g = np.nonzero(np.diff(st) > 20_000)[0]
T0 = st[g[-1] + 1] if len(g) else st[0]
blk = (r[:, 0] >> 8) & 0xFF
us = lambda x: (int(x) - T0) / 1e3  # noqa: E731
K = ["S0", "R0", "S1", "R1"]
ev = []
own = {}
for o in (0, 1):
    for row in r[(ph == o) & (r[:, 3] >= T0)]:
        own[int((row[0] >> 8) & 0xFF)] = o


def X(b, tid, name, a, z, cat, args=None):
    if z >= a >= T0:
        ev.append({"name": name, "cat": cat, "ph": "X", "pid": int(b), "tid": tid,
                   "ts": us(a), "dur": max(0.001, (int(z) - int(a)) / 1e3),
                   **({"args": args} if args else {})})


for row in r[r[:, 1] >= T0]:
    p, b = int(row[0] & 0xFF), int((row[0] >> 8) & 0xFF)
    t0, t1, t3 = row[1], row[2], row[3]
    hi = int((row[0] >> 32) & 0xFFFF)
    lo = int((row[0] >> 48) & 0xFFFF)
    if 120 <= p < 124:
        X(b, 1, f"wait {K[p - 120]}", t1, t3, "consumer wait",
          {"issued_us": us(t0), "ntok": hi, "tier": lo})
    elif p == 151:
        X(b, 1, "R0 math", t0, t1, "consumer")
    elif p == 153:
        X(b, 1, "R1 math", t0, t1, "consumer")
        X(b, 1, "R1 flush", t1, t3, "consumer")
    elif p == 140:
        X(b, 1, "w13 flush", t0, t1, "handoff")
        X(b, 1, "handoff bar.sync", t1, t3, "handoff")
    elif p in (141, 142):
        X(b, 1, "done count", t0, t1, "handoff")
        X(b, 1, "activate+release" if p == 142 else "not done", t1, t3, "handoff")
    elif p == 171:
        X(b, 2, f"empty wait {K[hi] if hi < 4 else hi}", t0, t1, "producer",
          {"chunk": lo})
        ev.append({"name": f"issue {K[hi] if hi < 4 else hi}", "ph": "i", "s": "t",
                   "pid": b, "tid": 2, "ts": us(t1)})
    elif p == 172:
        X(b, 2, "ready spin", t0, t1, "producer")
    elif p == 173:
        X(b, 2, "claim wait", t0, t1, "producer")
    elif p == 174:
        X(b, 2, "index math", t0, t1, "producer setup")
        X(b, 2, "record wait", t1, t3, "producer setup")
    elif p == 176:
        X(b, 3, f"claim {K[hi] if hi < 4 else hi}", t0, t1, "scheduler")
        X(b, 3, "decode+record+publish", t1, t3, "scheduler")
    elif p == 177:
        X(b, 3, "wait free slot", t0, t1, "scheduler")
    elif p == 170:
        ev.append({"name": "group", "ph": "i", "s": "t", "pid": b, "tid": 2, "ts": us(t0)})
for b, o in own.items():
    ev.append({"name": "process_name", "ph": "M", "pid": b,
               "args": {"name": f"CTA {b:3d} ({'cold' if o else 'hot'})"}})
    ev.append({"name": "process_sort_index", "ph": "M", "pid": b, "args": {"sort_index": b}})
    for tid, n in ((1, "consumer warp 0"), (2, "producer warp"), (3, "scheduler warp")):
        ev.append({"name": "thread_name", "ph": "M", "pid": b, "tid": tid, "args": {"name": n}})
json.dump({"traceEvents": ev, "displayTimeUnit": "ns"}, open(sys.argv[2], "w"))
print(len(ev), "events")

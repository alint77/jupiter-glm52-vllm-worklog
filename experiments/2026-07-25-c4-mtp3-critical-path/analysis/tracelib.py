"""Shared helpers: step segmentation, union busy, stream roles."""
import collections
from load import load

GPU = ("kernel", "gpu_memcpy", "gpu_memset")


def prep(rank):
    d = load(rank)
    ev = d["traceEvents"]
    t0 = min(e["ts"] for e in ev if e.get("ph") == "X" and e.get("cat") == "kernel")
    for e in ev:
        if e.get("ph") == "X":
            e["t"] = (e["ts"] - t0) / 1000.0  # ms
    return ev


def steps(ev):
    """Engine-step boundaries from the CPU-side execute_* annotation starts."""
    cpu = sorted(
        [e for e in ev if e.get("cat") == "user_annotation" and e["name"].startswith("execute_")],
        key=lambda e: e["t"],
    )
    steady = [e for e in cpu if "_generation_4(" in e["name"]]
    bounds = [e["t"] for e in steady]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def gpu_ops(ev, a=None, b=None):
    k = [e for e in ev if e.get("cat") in GPU]
    if a is not None:
        k = [e for e in k if a <= e["t"] < b]
    k.sort(key=lambda e: e["t"])
    return k


def union_busy(ops):
    iv = sorted((e["t"], e["t"] + e["dur"] / 1000.0) for e in ops)
    if not iv:
        return 0.0, []
    merged = [list(iv[0])]
    for s, en in iv[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], en)
        else:
            merged.append([s, en])
    return sum(e - s for s, e in merged), merged


def gaps(merged, a, b):
    g = []
    prev = a
    for s, e in merged:
        if s > prev:
            g.append((prev, s))
        prev = max(prev, e)
    if prev < b:
        g.append((prev, b))
    return g


def prep_depth(rank, depth):
    from load import load
    d = load(rank, depth)
    ev = d["traceEvents"]
    t0 = min(e["ts"] for e in ev if e.get("ph") == "X" and e.get("cat") == "kernel")
    for e in ev:
        if e.get("ph") == "X":
            e["t"] = (e["ts"] - t0) / 1000.0
    return ev

"""End-of-snapshot memory by allocating site for two snapshots and the
difference; plus reserved vs allocated per segment pool (CUDA graph pools are
not released by empty_cache).
Usage: memdiff.py <a.pickle> <b.pickle>"""
import collections, pickle, sys

GiB = 1 << 30


def owner(frames):
    keep = [f for f in frames if "/vllm/" in f["filename"] and "/torch/" not in f["filename"]]
    f = (keep or frames[:1] or [{"filename": "?", "line": 0, "name": "?"}])[0]
    return f"{f['filename'].split('/vllm/', 1)[-1]}:{f['line']} {f['name']}"


def load(path):
    snap = pickle.load(open(path, "rb"))
    live = {}
    for e in snap["device_traces"][0]:
        if e["action"] == "alloc":
            live[e["addr"]] = (e["size"], owner(e.get("frames", [])))
        elif e["action"] == "free_completed":
            live.pop(e["addr"], None)
    by = collections.Counter()
    for size, o in live.values():
        by[o] += size
    pools = collections.Counter()
    alloc = collections.Counter()
    for s in snap["segments"]:
        key = "graph pool" if s.get("segment_pool_id", (0, 0)) != (0, 0) else "default pool"
        pools[key] += s["total_size"]
        alloc[key] += sum(b["size"] for b in s["blocks"] if b["state"] == "active_allocated")
    return by, pools, alloc


a, b = load(sys.argv[1]), load(sys.argv[2])
for name, (by, pools, alloc) in (("A", a), ("B", b)):
    print(name, "  ".join(f"{k}: reserved {pools[k] / GiB:.2f} / allocated {alloc[k] / GiB:.2f} GiB" for k in pools))
keys = set(a[0]) | set(b[0])
print("live allocations made since recording started, by site (GiB): A, B, B - A")
for k in sorted(keys, key=lambda k: -abs(b[0][k] - a[0][k]))[:15]:
    print(f"  {a[0][k] / GiB:7.3f} {b[0][k] / GiB:7.3f} {(b[0][k] - a[0][k]) / GiB:+7.3f}  {k[:120]}")

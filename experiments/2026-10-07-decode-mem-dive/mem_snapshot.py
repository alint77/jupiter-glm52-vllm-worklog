"""Itemise a torch allocator snapshot (MEM_SNAPSHOT_DIR probe) by the vLLM
code that allocated each live block.

Key: the innermost frames under vllm/ (skipping generic helpers), `depth` of
them. Also: reserved-but-free bytes per pool, and CUDA-graph private pools.

    mem_snapshot.py <snap.pickle> [--depth 2] [--top 40] [--grep TEXT]
"""
import argparse
import collections
import pickle

GENERIC = ("/utils/", "_custom_ops.py", "torch_utils", "/parameter.py",
           "weight_utils.py", "/linear.py", "base_config.py")
G = 2 ** 30


def key(frames, depth):
    vf = [f for f in frames if "/vllm/vllm/" in f["filename"]]
    spec = [f for f in vf if not any(g in f["filename"] for g in GENERIC)]
    use = (spec or vf)[:depth]
    if not use:
        tail = frames[:2]
        return "<no vllm frame> " + " < ".join(f"{f['name']}" for f in tail)
    return " < ".join(f"{f['filename'].split('/vllm/vllm/')[1]}:{f['line']}:{f['name']}"
                      for f in use)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snap")
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--grep")
    a = ap.parse_args()
    s = pickle.load(open(a.snap, "rb"))
    by = collections.defaultdict(lambda: [0, 0])
    pool_res = collections.defaultdict(lambda: [0, 0, 0])
    for seg in s["segments"]:
        pid = tuple(seg["segment_pool_id"])
        pool_res[pid][0] += seg["total_size"]
        pool_res[pid][1] += seg["allocated_size"]
        pool_res[pid][2] += 1
        for b in seg["blocks"]:
            if b["state"] != "active_allocated":
                continue
            k = ("[graph pool] " if pid != (0, 0) else "") + key(b.get("frames", []), a.depth)
            if a.grep and a.grep not in k:
                continue
            by[k][0] += b["size"]
            by[k][1] += 1
    tot = sum(v[0] for v in by.values())
    print(f"live {tot / G:.3f} GiB in {sum(v[1] for v in by.values())} blocks")
    for pid, (t, al, n) in sorted(pool_res.items(), key=lambda x: -x[1][0]):
        print(f"  pool {pid}: reserved {t / G:.3f} GiB, allocated {al / G:.3f}, free {(t - al) / G:.3f}, {n} segments")
    for k, (b, n) in sorted(by.items(), key=lambda x: -x[1][0])[:a.top]:
        print(f"{b / G:8.3f} GiB {n:6d}  {k}")


if __name__ == "__main__":
    main()

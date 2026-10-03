"""Peak analysis of a torch.cuda.memory snapshot with allocation history.

Replays the alloc/free trace, finds the peak of allocated bytes (overall, or
after --after-gap: the first idle gap longer than N s, i.e. the request window
following startup), and lists the allocations live at the peak grouped by the
innermost vLLM frame that made them.
Usage: memsnap.py <snapshot.pickle> [--after-gap SECONDS] [--top N]"""
import collections, pickle, sys

args = sys.argv[1:]
path = args[0]
gap = float(args[args.index("--after-gap") + 1]) if "--after-gap" in args else None
top = int(args[args.index("--top") + 1]) if "--top" in args else 25
snap = pickle.load(open(path, "rb"))
trace = snap["device_traces"][0]
GiB = 1 << 30


def owner(frames):
    keep = [f for f in frames if "/vllm/" in f["filename"] and "/torch/" not in f["filename"]]
    if not keep:
        keep = frames[:1]
    f = keep[0]
    name = f["filename"].split("/vllm/", 1)[-1]
    caller = ""
    if len(keep) > 1:
        g = keep[1]
        caller = f"  <- {g['filename'].split('/vllm/', 1)[-1]}:{g['line']} {g['name']}"
    return f"{name}:{f['line']} {f['name']}{caller}"


# live allocations from segments existing before history started
live = {}
base = 0
for seg in snap["segments"]:
    pass  # segments describe the end state; history replay starts from 0 live
cur = 0
peak, peak_i = 0, -1
start_i = 0
if gap is not None:
    for i in range(1, len(trace)):
        if trace[i]["time_us"] - trace[i - 1]["time_us"] > gap * 1e6:
            start_i = i
            break
events = []
for i, e in enumerate(trace):
    a = e["action"]
    if a == "alloc":
        live[e["addr"]] = (e["size"], e.get("frames", []), i)
        cur += e["size"]
    elif a in ("free_completed",):
        v = live.pop(e["addr"], None)
        if v:
            cur -= v[0]
    if i >= start_i and cur > peak:
        peak, peak_i = cur, i
# rebuild the live set at the peak
live2 = {}
for i, e in enumerate(trace[: peak_i + 1]):
    a = e["action"]
    if a == "alloc":
        live2[e["addr"]] = (e["size"], e.get("frames", []), i)
    elif a == "free_completed":
        live2.pop(e["addr"], None)
t0 = trace[0]["time_us"]
print(f"{len(trace)} events; window from event {start_i} "
      f"(t+{(trace[start_i]['time_us'] - t0) / 1e6:.1f} s)")
print(f"peak of allocated-by-history bytes: {peak / GiB:.2f} GiB at event {peak_i} "
      f"(t+{(trace[peak_i]['time_us'] - t0) / 1e6:.1f} s)")
e = trace[peak_i]
print(f"allocation at the peak: {e['size'] / (1 << 20):.1f} MiB by {owner(e.get('frames', []))}")
groups = collections.defaultdict(lambda: [0, 0])
for size, frames, i in live2.values():
    g = groups[owner(frames)]
    g[0] += size; g[1] += 1
print(f"live at the peak, by allocating site (top {top}):")
for k, (b, n) in sorted(groups.items(), key=lambda kv: -kv[1][0])[:top]:
    print(f"  {b / GiB:7.3f} GiB {n:6d}x  {k[:170]}")
reserved = sum(s["total_size"] for s in snap["segments"])
alloc = sum(b["size"] for s in snap["segments"] for b in s["blocks"] if b["state"] == "active_allocated")
print(f"at dump: reserved {reserved / GiB:.2f} GiB, allocated {alloc / GiB:.2f} GiB "
      f"(history covers only allocations made after recording started)")

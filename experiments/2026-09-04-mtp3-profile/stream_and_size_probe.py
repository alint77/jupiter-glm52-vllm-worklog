import collections, gzip, json, sys
from pathlib import Path

SMEM = Path("agent_space/experiments/2026-07-29-marlin-smem-monopoly")
sys.path.insert(0, str(SMEM))
from analyze_step_budget import load_trace, step_windows, GPU_CATEGORIES, LAUNCH_CATEGORIES

root = Path("/e/project1/profound/alint77/traces/mtp3-profile-1665068")

for label in ("decode", "prefill"):
    path = sorted((root / label).glob("*_rank0.*trace.json.gz"))[0]
    events = load_trace(path)
    bycorr = collections.defaultdict(list)
    for e in events:
        if e.get("cat") in GPU_CATEGORIES and e.get("args", {}).get("correlation") is not None:
            bycorr[e["args"]["correlation"]].append(e)
    launches = sorted((e for e in events if e.get("cat") in LAUNCH_CATEGORIES), key=lambda e: e["t"])
    windows = step_windows(events)
    # middle step, representative
    start, end = windows[len(windows) // 2]
    corrs = [e["args"]["correlation"] for e in launches
             if start <= e["t"] < end and "correlation" in e.get("args", {})]
    ops = sorted((k for c in corrs for k in bycorr[c]), key=lambda e: e["t"])

    print(f"\n===== {label}  (rank0, step {len(windows)//2} of {len(windows)}) =====")
    marlin = [e for e in ops if "marlin_moe_wna16" in e["name"]]
    bystream = collections.defaultdict(lambda: {"n": 0, "us": 0.0})
    for e in marlin:
        s = bystream[e["args"].get("stream")]
        s["n"] += 1
        s["us"] += e["dur"]
    span = (max(e["t"] + e["dur"]/1000 for e in marlin) - min(e["t"] for e in marlin)) if marlin else 0
    print(f"marlin: {len(marlin)} launches, cumulative {sum(e['dur'] for e in marlin)/1000:.3f} ms, "
          f"wall span {span:.3f} ms")
    for stream, s in sorted(bystream.items(), key=lambda i: -i[1]["us"]):
        print(f"  stream {stream}: {s['n']:4d} launches  {s['us']/1000:8.3f} ms")
    if len(bystream) == 2:
        a, b = sorted((v["us"]/1000 for v in bystream.values()), reverse=True)
        print(f"  serial would be {a+b:.3f} ms; perfect overlap {a:.3f} ms; "
              f"actual marlin wall span {span:.3f} ms")

    # all-reduce message sizes from record_shapes on the launching op
    for needle in ("cross_device_reduce", "ncclDevKernel_AllReduce"):
        sel = [e for e in ops if needle in e["name"]]
        if not sel:
            continue
        durs = sorted(e["dur"] for e in sel)
        print(f"{needle}: {len(sel)} launches, {sum(durs)/1000:.3f} ms, "
              f"per-call min {durs[0]:.1f}us med {durs[len(durs)//2]:.1f}us max {durs[-1]:.1f}us")
    # shapes: find cpu_op ancestors carrying Input Dims for all_reduce
    shapes = collections.Counter()
    for e in events:
        if e.get("cat") == "cpu_op" and "all_reduce" in e.get("name", "").lower():
            dims = e.get("args", {}).get("Input Dims")
            if dims:
                shapes[json.dumps(dims)] += 1
    for dims, n in shapes.most_common(4):
        print(f"  all_reduce input dims {dims}  x{n} (whole trace)")

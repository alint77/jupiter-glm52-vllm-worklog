import collections, statistics, sys
from pathlib import Path
SMEM = Path("agent_space/experiments/2026-07-29-marlin-smem-monopoly")
sys.path.insert(0, str(SMEM))
from analyze_step_budget import load_trace, step_windows, GPU_CATEGORIES, LAUNCH_CATEGORIES
root = Path("/e/project1/profound/alint77/traces/mtp3-profile-1665068")

def union_ms(evts):
    iv = sorted((e["t"], e["t"] + e["dur"]/1000) for e in evts)
    total, cur_s, cur_e = 0.0, None, None
    for s, e in iv:
        if cur_e is None or s > cur_e:
            if cur_e is not None: total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    return total + (cur_e - cur_s if cur_e is not None else 0.0)

for label in ("decode", "prefill"):
    path = sorted((root / label).glob("*_rank0.*trace.json.gz"))[0]
    events = load_trace(path)
    bycorr = collections.defaultdict(list)
    for e in events:
        if e.get("cat") in GPU_CATEGORIES and e.get("args", {}).get("correlation") is not None:
            bycorr[e["args"]["correlation"]].append(e)
    launches = sorted((e for e in events if e.get("cat") in LAUNCH_CATEGORIES), key=lambda e: e["t"])
    windows = step_windows(events)
    picks = [len(windows)//2] if label == "prefill" else [len(windows)//3, len(windows)//2, 2*len(windows)//3]
    print(f"\n===== {label} =====")
    for wi in picks:
        start, end = windows[wi]
        corrs = [e["args"]["correlation"] for e in launches
                 if start <= e["t"] < end and "correlation" in e.get("args", {})]
        ops = sorted((k for c in corrs for k in bycorr[c]), key=lambda e: e["t"])
        marlin = [e for e in ops if "marlin_moe_wna16" in e["name"]]
        cum = sum(e["dur"] for e in marlin)/1000
        uni = union_ms(marlin)
        allgpu = union_ms(ops)
        print(f" step {wi}: GPU busy(union) {allgpu:7.3f} ms | marlin cum {cum:7.3f} "
              f"union {uni:7.3f} -> overlap saves {cum-uni:6.3f} ms "
              f"({100*(cum-uni)/cum:4.1f}%), marlin is {100*uni/allgpu:4.1f}% of busy")
        for needle in ("cross_device_reduce", "ncclDevKernel_AllReduce"):
            sel = [e for e in ops if needle in e["name"]]
            if not sel: continue
            d = sorted(e["dur"] for e in sel)
            n = len(d)
            head = sum(d[:int(n*0.9)])/1000
            tail = sum(d[int(n*0.9):])/1000
            print(f"   {needle}: n={n} total {sum(d)/1000:7.3f} ms | "
                  f"p50 {d[n//2]:6.1f}us p90 {d[int(n*0.9)]:7.1f}us p99 {d[int(n*0.99)]:7.1f}us "
                  f"max {d[-1]:7.1f}us | bottom-90% {head:6.3f} ms, top-10% {tail:6.3f} ms "
                  f"({100*tail/(head+tail):4.1f}%)")
            # exposed vs overlapped: is any other kernel running concurrently?
            others = [e for e in ops if needle not in e["name"]]
            oiv = sorted((e["t"], e["t"] + e["dur"]/1000) for e in others)
            exposed = 0.0
            j = 0
            for e in sorted(sel, key=lambda x: x["t"]):
                s0, e0 = e["t"], e["t"] + e["dur"]/1000
                cov = 0.0
                for s1, e1 in oiv:
                    if e1 <= s0: continue
                    if s1 >= e0: break
                    cov += min(e0, e1) - max(s0, s1)
                exposed += max(0.0, (e0 - s0) - min(cov, e0 - s0))
            print(f"   {needle}: exposed (no other kernel concurrent) {exposed:7.3f} ms "
                  f"of {sum(d)/1000:7.3f} ms ({100*exposed/(sum(d)/1000):4.1f}%)")

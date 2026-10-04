"""Shared expert (aux stream) vs routed MoE (main stream) in served decode
steps: for each MoE layer, each aux kernel's start / end relative to the
layer's router GEMM, its delay after becoming launchable (previous aux kernel
done), and how many us of it overlap each main-stream kernel.
Usage: overlap_trace.py <trace.json.gz> [first_step last_step]"""
import collections
import gzip
import json
import statistics as st
import sys

ev = json.load(gzip.open(sys.argv[1]))["traceEvents"]
a, b = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (5, 45)
ann = sorted([e for e in ev if e.get("cat") == "gpu_user_annotation"
              and e["name"].startswith("execute_context_0(0)")], key=lambda e: e["ts"])[a:b]
kern = sorted([e for e in ev if e.get("cat") == "kernel"], key=lambda e: e["ts"])


def short(n):
    for key, lab in (("cute_dsl_ll_bf16_splitk", "router"), ("grouped_topk", "topk"),
                     ("route_prep", "route_prep"), ("gemm_kernel<0", "w13"),
                     ("act_kernel", "act"), ("gemm_kernel<1", "w2"),
                     ("finalize_kernel", "finalize"), ("fused_add_mul", "add"),
                     ("allreduce_fusion", "AR+norm"), ("splitKreduce", "splitK_reduce"),
                     ("silu", "silu*up"), ("nvjet", "nvjet")):
        if key in n:
            return lab
    return n[:30]


aux_stream = collections.Counter(k["args"].get("stream") for k in kern).most_common(2)[1][0]
rows = []  # per layer
for w in ann:
    ks = [k for k in kern if w["ts"] <= k["ts"] < w["ts"] + w["dur"]]
    main = [k for k in ks if k["args"].get("stream") != aux_stream]
    aux = [k for k in ks if k["args"].get("stream") == aux_stream]
    routers = [i for i, k in enumerate(main) if short(k["name"]) == "router"]
    for i in routers:
        r = main[i]
        # main kernels from the router to the next AR+norm
        seq = []
        for k in main[i:]:
            seq.append(k)
            if short(k["name"]) == "AR+norm" and len(seq) > 1:
                break
        t_end = seq[-1]["ts"] + seq[-1]["dur"]
        # aux kernels of this layer: start at/after the previous AR, before this layer's end
        prev_ar = max((k["ts"] + k["dur"] for k in main[:i] if short(k["name"]) == "AR+norm"),
                      default=r["ts"] - 50)
        ax = [k for k in aux if prev_ar - 1 <= k["ts"] < t_end]
        if len(ax) != 4:
            continue
        labs = ["gate_up", "splitK_reduce", "silu*up", "down"]
        row = {"t0": r["ts"]}
        prev_end = None
        for lab, k in zip(labs, ax):
            s, e = k["ts"], k["ts"] + k["dur"]
            row[lab] = {"start": s - r["ts"], "dur": k["dur"],
                        "delay": (s - prev_end) if prev_end is not None else None,
                        "ov": {}}
            for m in seq:
                o = min(e, m["ts"] + m["dur"]) - max(s, m["ts"])
                if o > 0:
                    lm = short(m["name"])
                    row[lab]["ov"][lm] = row[lab]["ov"].get(lm, 0) + o
            row[lab]["alone"] = k["dur"] - sum(row[lab]["ov"].values())
            prev_end = e
        row["main"] = {short(m["name"]): (m["ts"] - r["ts"], m["dur"]) for m in seq}
        row["join_slack"] = row["main"].get("add", (None,))[0]
        rows.append(row)

print(f"{sys.argv[1].split('/')[-1][:24]}: {len(rows)} layers, aux stream {aux_stream}")
labs = ["gate_up", "splitK_reduce", "silu*up", "down"]
mains = ["router", "topk", "route_prep", "w13", "act", "w2", "finalize", "add", "AR+norm"]
print("main stream, median start / duration (us from router start):")
print("   " + "  ".join(f"{m} {st.median(r['main'][m][0] for r in rows if m in r['main']):.0f}"
                      f"+{st.median(r['main'][m][1] for r in rows if m in r['main']):.0f}"
                      for m in mains if any(m in r["main"] for r in rows)))
for lab in labs:
    d = [r[lab] for r in rows]
    dl = [x["delay"] for x in d if x["delay"] is not None]
    line = (f"{lab:14s} start {st.median(x['start'] for x in d):6.1f}  dur {st.median(x['dur'] for x in d):5.1f}"
            + (f"  waited {st.median(dl):5.1f} (p90 {sorted(dl)[9 * len(dl) // 10]:.0f})" if dl else "")
            + "  | overlap us (median over layers; % of layers with any): ")
    for m in mains + ["alone"]:
        vals = [x["ov"].get(m, 0) if m != "alone" else x["alone"] for x in d]
        frac = sum(v > 0.05 for v in vals) / len(vals)
        if frac > 0.02:
            line += f"{m} {st.median(vals):.1f} ({frac * 100:.0f}%)  "
    print(line)

"""Skip-layer KV staging in a served decode trace (torch profiler, rank 0).

Per decode step: the 78 FlashMLA sparse decode kernels on the main stream in
layer order, the staging kernels on the stager's stream (the stream that runs
`_gather_rows_kernel`), grouped by the anchor they follow. Reports per group
the staging span and gather bandwidth (rows from the device count are not in
the trace, so bytes come from the histogram run if given), the slack between
staging done and the first skip layer's FlashMLA start (negative = waited),
and the gap before each skip layer's FlashMLA against the same gap in a
reference trace (prod arm).
Usage: skipkv_trace.py <skip trace.json.gz> [<prod trace.json.gz>]"""
import gzip
import json
import re
import statistics as st
import sys

ANCHORS = {0, 1, 2} | set(range(6, 78, 4))
MLA = re.compile(r"flash_fwd_splitkv_mla_fp8_sparse")


def load(path):
    ev = json.load(gzip.open(path))["traceEvents"]
    steps = sorted(
        {e["ts"]: e for e in ev if e.get("cat") == "gpu_user_annotation"
         and e["name"].startswith("execute_context")}.values(),
        key=lambda e: e["ts"],
    )
    kern = sorted((e for e in ev if e.get("cat") == "kernel"), key=lambda e: e["ts"])
    return steps, kern


def per_step(steps, kern):
    # Steps from the FlashMLA calls themselves: 78 per step, split at the gaps
    # between steps (the annotation's GPU range can miss kernels that replay on
    # a forked graph branch).
    mla_all = [k for k in kern if MLA.search(k["name"])]
    groups, cur = [], []
    for k in mla_all:
        if cur and k["ts"] - cur[-1]["ts"] > 1000:
            groups.append(cur)
            cur = []
        cur.append(k)
    groups.append(cur)
    out = []
    j = 0
    for mla in groups:
        if len(mla) != 78:
            continue
        t0, t1 = mla[0]["ts"] - 2000, mla[-1]["ts"] + mla[-1]["dur"] + 2000
        while j < len(kern) and kern[j]["ts"] < t0:
            j += 1
        ks = []
        i = j
        while i < len(kern) and kern[i]["ts"] < t1:
            ks.append(kern[i])
            i += 1
        dur = mla[-1]["ts"] - mla[0]["ts"]
        # graph replay may place forked branches on other stream ids: take each
        # FlashMLA call's gap on its own stream
        by_stream: dict = {}
        for k in ks:
            by_stream.setdefault(k["args"].get("stream"), []).append(k)
        gaps = []
        for k in mla:
            same = by_stream[k["args"].get("stream")]
            i = same.index(k)
            prev = same[i - 1] if i else None
            gaps.append(k["ts"] - (prev["ts"] + prev["dur"]) if prev else 0.0)
        gather = [k for k in ks if "_gather_rows_kernel" in k["name"]]
        side = [k for k in ks if any(n in k["name"] for n in
                                      ("_gather_rows_kernel", "_claim_rows", "_remap_rows"))]
        out.append({"dur": dur, "mla": mla, "gaps": gaps, "side": side,
                    "gather": gather})
    return out


def report(skip, ref):
    print(f"decode steps with 78 FlashMLA calls: skip {len(skip)}"
          + (f", prod {len(ref)}" if ref is not None else ""))
    print(f"FlashMLA 0 -> 77 span median: skip {st.median(s['dur'] for s in skip) / 1e3:.2f}"
          f" ms" + (f", prod {st.median(s['dur'] for s in ref) / 1e3:.2f} ms"
                    if ref else ""))
    if not skip[0]["gather"]:
        print("no _gather_rows_kernel in the trace: staging did not run")
        return
    span, slack, gk = [], [], []
    for s in skip:
        mla, side = s["mla"], s["side"]
        for a in sorted(ANCHORS):
            if a + 1 in ANCHORS or a + 1 > 77:
                continue
            t0 = mla[a]["ts"]  # staging forks inside the anchor's attention
            t1 = mla[a + 1]["ts"]
            ks = [k for k in side if t0 - 200 <= k["ts"] < t1]
            if not ks:
                continue
            end = max(k["ts"] + k["dur"] for k in ks)
            span.append(end - ks[0]["ts"])
            slack.append(t1 - end)
        gk += [k["dur"] for k in s["gather"]]
    q = lambda v, p: sorted(v)[min(len(v) - 1, int(p * len(v)))]
    print(f"staging span per group (us): median {st.median(span):.1f}, p90 "
          f"{q(span, .9):.1f}, max {max(span):.1f}; side-stream kernels per step "
          f"{len(skip[0]['side'])}; gather kernel median {st.median(gk):.1f} us")
    print(f"slack staging-done -> first skip FlashMLA (us): min {min(slack):.1f}, "
          f"p10 {q(slack, .1):.1f}, median {st.median(slack):.1f}; "
          f"{sum(x < 0 for x in slack)} of {len(slack)} groups late")
    skip_layers = [i for i in range(78) if i not in ANCHORS]
    g = [s["gaps"][i] for s in skip for i in skip_layers]
    line = f"gap before skip-layer FlashMLA (us): skip median {st.median(g):.2f} p99 {q(g, .99):.2f}"
    if ref:
        r = [s["gaps"][i] for s in ref for i in skip_layers]
        line += f"; prod median {st.median(r):.2f} p99 {q(r, .99):.2f}"
    print(line)
    for name, sel in (("anchor", sorted(ANCHORS)), ("skip", skip_layers)):
        d = [s["mla"][i]["dur"] for s in skip for i in sel]
        line = f"FlashMLA {name} layers median {st.median(d):.1f} us"
        if ref:
            line += f" (prod {st.median(s['mla'][i]['dur'] for s in ref for i in sel):.1f})"
        print(line)


if __name__ == "__main__":
    skip = per_step(*load(sys.argv[1]))
    ref = per_step(*load(sys.argv[2])) if len(sys.argv) > 2 else None
    report(skip, ref)

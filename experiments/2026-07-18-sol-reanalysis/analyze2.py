"""Pass 2: GPU-annotation-aligned phases, concurrency, and solo-time attribution."""

import json
import collections
from analyze import classify, union_busy

RANKS = range(4)


def load(rank):
    with open(f"traces/rank{rank}.json") as f:
        events = json.load(f)["traceEvents"]
    kernels = [e for e in events if e.get("cat") in ("kernel", "gpu_memset", "gpu_memcpy")]
    for e in kernels:
        e["end"] = e["ts"] + e["dur"]
    # GPU-side step windows: generation annotations on the busiest tid
    gann = [
        e for e in events
        if e.get("cat") == "gpu_user_annotation" and e["name"].startswith("execute_context")
    ]
    by_tid = collections.Counter(e["tid"] for e in gann)
    main_tid = by_tid.most_common(1)[0][0]
    windows = sorted(
        ((e["ts"], e["ts"] + e["dur"]) for e in gann if e["tid"] == main_tid)
    )
    return kernels, windows


def phase_stats(kernels, windows):
    """Segment each GPU step window by the four vocab GEMMs."""
    acc = collections.defaultdict(lambda: collections.defaultdict(float))
    phase_windows = []
    steps = []
    for wi, (lo, hi) in enumerate(windows):
        ks = [e for e in kernels if lo <= e["ts"] < hi]
        vocab = sorted(
            (e for e in ks if "nvjet_sm90_tst_192x8" in e["name"]),
            key=lambda e: e["ts"],
        )
        if len(vocab) != 4:
            continue
        cuts = [lo] + [v["end"] for v in vocab] + [hi]
        names = ["target_verify", "draft1", "draft2", "draft3", "tail"]
        pw = {}
        for pname, plo, phi in zip(names, cuts, cuts[1:]):
            pks = [e for e in ks if plo <= e["ts"] < phi]
            busy, _ = union_busy([(e["ts"], e["end"]) for e in pks])
            acc[pname]["span"] += phi - plo
            acc[pname]["busy"] += busy
            acc[pname]["n"] += 1
            pw[pname] = (plo, phi)
            # component split inside phase
            for e in pks:
                comp, _ = classify(e["name"])
                acc[pname][f"c_{comp}"] += e["dur"]
        phase_windows.append(pw)
        steps.append((lo, hi, ks))
    return acc, phase_windows, steps


def concurrency(steps):
    """Time-weighted number of concurrently busy streams, and solo attribution."""
    conc = collections.Counter()
    solo = collections.defaultdict(float)
    for lo, hi, ks in steps:
        points = []
        for e in ks:
            comp, _ = classify(e["name"])
            points.append((e["ts"], 1, comp))
            points.append((e["end"], -1, comp))
        points.sort()
        active = collections.Counter()
        depth = 0
        prev = None
        for ts, delta, comp in points:
            if prev is not None and depth > 0:
                conc[min(depth, 4)] += ts - prev
                if len(active) == 1:
                    (only,) = active.keys()
                    solo[only] += ts - prev
            if delta > 0:
                active[comp] += 1
                depth += 1
            else:
                active[comp] -= 1
                if active[comp] == 0:
                    del active[comp]
                depth -= 1
            prev = ts
    return conc, solo


def main():
    for rank in RANKS:
        kernels, windows = load(rank)
        acc, pw, steps = phase_stats(kernels, windows)
        n_steps = len(steps)
        print(f"\n=== rank {rank}: {n_steps} GPU step windows ===")
        wall = sum(hi - lo for lo, hi, _ in steps) / n_steps
        print(f"GPU window {wall/1000:.3f} ms/step")
        for pname in ("target_verify", "draft1", "draft2", "draft3", "tail"):
            a = acc[pname]
            n = a["n"]
            if not n:
                continue
            comps = sorted(
                ((k[2:], v / n) for k, v in a.items() if k.startswith("c_")),
                key=lambda kv: -kv[1],
            )
            top = ", ".join(f"{c}:{v/1000:.2f}" for c, v in comps[:5])
            print(
                f"  {pname:14s} span {a['span']/n/1000:7.3f} busy {a['busy']/n/1000:7.3f} | {top}"
            )
        conc, solo = concurrency(steps)
        total = sum(conc.values())
        print("  concurrency (share of busy time at depth 1/2/3/4+):",
              " ".join(f"{d}:{conc[d]/total*100:.1f}%" for d in sorted(conc)))
        print("  solo time ms/step (component is the ONLY thing running):")
        for comp, v in sorted(solo.items(), key=lambda kv: -kv[1])[:10]:
            print(f"    {comp:22s} {v/1000/n_steps:7.3f}")


if __name__ == "__main__":
    main()

"""Pass 3: correct phase split, overlap quality, Marlin stats, solo attribution."""

import json
import collections
import statistics
from analyze import classify, union_busy

def load(rank):
    with open(f"traces/rank{rank}.json") as f:
        events = json.load(f)["traceEvents"]
    kernels = [e for e in events if e.get("cat") in ("kernel", "gpu_memset", "gpu_memcpy")]
    for e in kernels:
        e["end"] = e["ts"] + e["dur"]
    gann = [
        e for e in events
        if e.get("cat") == "gpu_user_annotation"
        and e["name"].startswith("execute_context")
        and e["tid"] == 19
    ]
    windows = sorted((e["ts"], e["ts"] + e["dur"]) for e in gann)
    return kernels, windows


def main():
    grand = {}
    for rank in range(4):
        kernels, windows = load(rank)
        # steps: target window w[i], draft/tail = [w[i].end, w[i+1].start]
        stats = collections.defaultdict(list)
        comp_target = collections.defaultdict(float)
        comp_gap = collections.defaultdict(float)
        marlin_durs = collections.defaultdict(list)
        conc_depth = collections.Counter()
        solo = collections.defaultdict(float)
        moe_solo_overlap = [0.0, 0.0]  # [marlin solo, marlin-with-marlin]

        for i in range(1, len(windows) - 1):  # skip first window (warmup-ish)
            tlo, thi = windows[i]
            glo, ghi = thi, windows[i + 1][0]
            step_wall = windows[i + 1][0] - tlo
            tks = [e for e in kernels if tlo <= e["ts"] < thi]
            gks = [e for e in kernels if glo <= e["ts"] < ghi]
            tbusy, _ = union_busy([(e["ts"], e["end"]) for e in tks])
            gbusy, _ = union_busy([(e["ts"], e["end"]) for e in gks])
            stats["step_wall"].append(step_wall)
            stats["target_span"].append(thi - tlo)
            stats["target_busy"].append(tbusy)
            stats["gap_span"].append(ghi - glo)
            stats["gap_busy"].append(gbusy)
            for e in tks:
                comp, _ = classify(e["name"])
                comp_target[comp] += e["dur"]
                if comp == "routed_moe_marlin":
                    marlin_durs[e["args"]["stream"]].append(e["dur"])
            for e in gks:
                comp, _ = classify(e["name"])
                comp_gap[comp] += e["dur"]

            # routed marlin union span within target window
            mivs = [(e["ts"], e["end"]) for e in tks if "marlin_moe_wna16" in e["name"]]
            mspan, _ = union_busy(mivs)
            stats["marlin_sum"].append(sum(b - a for a, b in mivs))
            stats["marlin_union"].append(mspan)

            # concurrency + solo over target window
            points = []
            for e in tks:
                comp, _ = classify(e["name"])
                points.append((e["ts"], 1, comp))
                points.append((e["end"], -1, comp))
            points.sort(key=lambda p: (p[0], -p[1]))
            active = collections.Counter()
            depth = 0
            prev = None
            for ts, delta, comp in points:
                if prev is not None and depth > 0:
                    dt = ts - prev
                    conc_depth[min(depth, 4)] += dt
                    if len(active) == 1:
                        (only,) = active.keys()
                        solo[only] += dt
                        if only == "routed_moe_marlin":
                            if active[only] == 1:
                                moe_solo_overlap[0] += dt
                            else:
                                moe_solo_overlap[1] += dt
                if delta > 0:
                    active[comp] += 1
                    depth += 1
                else:
                    active[comp] -= 1
                    if not active[comp]:
                        del active[comp]
                    depth -= 1
                prev = ts
        n = len(stats["step_wall"])
        m = lambda k: sum(stats[k]) / n / 1000
        print(f"\n=== rank {rank} ({n} steps) ===")
        print(f"step wall {m('step_wall'):.3f} ms = target window {m('target_span'):.3f}"
              f" (busy {m('target_busy'):.3f}) + draft/sample gap {m('gap_span'):.3f}"
              f" (busy {m('gap_busy'):.3f})")
        print(f"routed marlin: sum {m('marlin_sum'):.3f} ms -> union span {m('marlin_union'):.3f} ms"
              f" (overlap saves {m('marlin_sum')-m('marlin_union'):.3f})")
        total_c = sum(conc_depth.values())
        print("target-window concurrency:",
              " ".join(f"d{d}:{conc_depth[d]/total_c*100:.0f}%" for d in sorted(conc_depth)))
        print("marlin solo(single-instance) vs marlin||marlin:",
              f"{moe_solo_overlap[0]/n/1000:.3f} / {moe_solo_overlap[1]/n/1000:.3f} ms")
        print("solo time in target window (ms/step):")
        for comp, v in sorted(solo.items(), key=lambda kv: -kv[1])[:8]:
            print(f"  {comp:22s} {v/1000/n:7.3f}")
        print("target-window component busy (ms/step, top):")
        for comp, v in sorted(comp_target.items(), key=lambda kv: -kv[1])[:10]:
            print(f"  {comp:22s} {v/1000/n:7.3f}")
        print("draft-gap component busy (ms/step, top):")
        for comp, v in sorted(comp_gap.items(), key=lambda kv: -kv[1])[:8]:
            print(f"  {comp:22s} {v/1000/n:7.3f}")
        print("routed marlin per-call duration by stream (us): "
              + ", ".join(
                  f"s{s}: n={len(d)/n:.0f}/step med={statistics.median(d):.0f}"
                  f" p90={statistics.quantiles(d, n=10)[8]:.0f} max={max(d):.0f}"
                  for s, d in sorted(marlin_durs.items())))
        grand[rank] = dict(step=m("step_wall"), ar=comp_target["custom_allreduce"]/n/1000,
                           marlin=m("marlin_sum"))
    print("\ncross-rank: ", {r: {k: round(v, 2) for k, v in g.items()} for r, g in grand.items()})


if __name__ == "__main__":
    main()

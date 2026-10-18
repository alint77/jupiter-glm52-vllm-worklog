"""Engine steps of a c=4 timeline arm's nsys windows (device 0), joined with
the replay's request events: each step's kind (decode verify / prefill chunk
/ other = draft passes and small steps), wall, GPU busy, and how many
requests were decoding (first token seen, not finished) and waiting for their
first token at that moment (events.jsonl, client clock = nsys session clock).

Prints, over all windows: decode step wall by number of decoders, prefill
chunk wall, other steps per decode step, the engine's time split (decode /
prefill / other / gaps between steps), and per window the mix.

    swarm_steps.py <arm out dir> [--device 0] [--csv steps.csv]
"""
import argparse
import bisect
import collections
import json
import sqlite3
import statistics as st
from pathlib import Path


def steps_of(db, device):
    c = sqlite3.connect(db)
    names = dict(c.execute("select id, value from StringIds"))
    t_sess = c.execute("select utcEpochNs from TARGET_INFO_SESSION_START_TIME").fetchone()[0]
    ks = [(s, e, names[n], corr) for s, e, n, corr in c.execute(
        "select start, end, shortName, correlationId from CUPTI_ACTIVITY_KIND_KERNEL "
        "where deviceId=? order by start", (device,))]
    mc = c.execute("select start, end from CUPTI_ACTIVITY_KIND_MEMCPY where deviceId=? "
                   "order by start", (device,)).fetchall()
    per_corr = collections.Counter(k[3] for k in ks)
    marks = [k[0] for k in ks if k[2].startswith("_compute_slot_mappings")]
    kstarts = [k[0] for k in ks]
    mstarts = [m[0] for m in mc]
    out = []
    for lo, hi in zip(marks, marks[1:]):
        sk = ks[bisect.bisect_left(kstarts, lo):bisect.bisect_left(kstarts, hi)]
        sm = mc[bisect.bisect_left(mstarts, lo):bisect.bisect_left(mstarts, hi)]
        big = max((per_corr[k[3]] for k in sk), default=0)
        kind = ("prefill" if any(k[2] in ("gemm_kernel", "gemm_xres_kernel") for k in sk)
                else "decode" if big >= 1500 else "other")
        iv = sorted([(k[0], k[1]) for k in sk] + list(sm))
        busy, cur = 0, lo
        for s, e in iv:
            s, e = max(s, cur), min(e, hi)
            if e > s:
                busy += e - s
                cur = e
        last_end = max((e for _, e in iv), default=lo)
        out.append(dict(t=(t_sess + lo) / 1e9, kind=kind, wall=(hi - lo) / 1e6,
                        busy=busy / 1e6, tail=(hi - min(hi, last_end)) / 1e6, nk=len(sk)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--csv")
    a = ap.parse_args()
    d = Path(a.dir)
    t0 = json.loads(open(d / "timings.jsonl").readline())["t0_unix"]
    ev = [json.loads(l) for l in open(d / "events.jsonl")]

    def state(t):
        r = t - t0
        dec = [e for e in ev if e["first"] is not None and e["first"] <= r < e["end"]]
        wait = sum(1 for e in ev if e["send"] <= r < (e["first"] or e["end"]))
        ctx = st.mean(e["input"] or 0 for e in dec) / 1e3 if dec else 0
        return len(dec), wait, ctx

    allsteps = []
    for db in sorted(d.glob("nsys.*.sqlite"), key=lambda p: int(p.name.split(".")[1])):
        ss = steps_of(db, a.device)
        for s in ss:
            s["win"] = int(db.name.split(".")[1])
            s["dec"], s["wait"], s["ctx"] = state(s["t"])
        allsteps += ss
        if not ss:
            continue
        span = ss[-1]["t"] + ss[-1]["wall"] / 1e3 - ss[0]["t"]
        mix = collections.Counter(s["kind"] for s in ss)
        tm = collections.Counter()
        for s in ss:
            tm[s["kind"]] += s["wall"]
        print(f"window {s['win']}: +{ss[0]['t'] - t0:.0f} s, {span:.2f} s, "
              + ", ".join(f"{k} {mix[k]} ({tm[k] / 1e3 / span * 100:.0f}%)" for k in mix)
              + f" | decoders {collections.Counter(s['dec'] for s in ss if s['kind'] == 'decode')}"
              + f" waiting {collections.Counter(s['wait'] for s in ss)}")
    if a.csv:
        with open(a.csv, "w") as f:
            f.write("win,t,kind,wall_ms,busy_ms,tail_ms,decoders,waiting,ctx_k\n")
            for s in allsteps:
                f.write(f"{s['win']},{s['t'] - t0:.4f},{s['kind']},{s['wall']:.3f},"
                        f"{s['busy']:.3f},{s['tail']:.3f},{s['dec']},{s['wait']},{s['ctx']:.0f}\n")

    def q(v):
        v = sorted(v)
        return (f"n {len(v):4d}  med {st.median(v):7.2f}  p10 {v[len(v) // 10]:7.2f}  "
                f"p90 {v[len(v) * 9 // 10]:7.2f}  mean {st.mean(v):7.2f}") if v else "n 0"

    print("\ndecode verify step wall (ms) by decoders at that moment and their mean "
          "context (K tokens); idle = wall - GPU busy:")
    by = collections.defaultdict(list)
    for s in allsteps:
        if s["kind"] == "decode":
            by[(s["dec"], min(3, int(s["ctx"] // 100)))].append(s)
    for (k, c), ss in sorted(by.items()):
        cb = ("<100K", "100-200K", "200-300K", ">300K")[c]
        print(f"  {k} decoding, ctx {cb:<9} wall {q([s['wall'] for s in ss])}"
              f" | idle {st.mean(s['wall'] - s['busy'] for s in ss):.2f}")
    pf = [s for s in allsteps if s["kind"] == "prefill"]
    print(f"\nprefill chunk wall (ms): {q([s['wall'] for s in pf])} | idle "
          f"{st.mean(s['wall'] - s['busy'] for s in pf) if pf else 0:.2f}")
    ot = [s for s in allsteps if s["kind"] == "other"]
    nd = sum(1 for s in allsteps if s["kind"] == "decode")
    print(f"other steps (draft passes etc.): {q([s['wall'] for s in ot])}, "
          f"{len(ot) / max(1, nd):.2f} per decode step")
    tot = collections.Counter()
    for s in allsteps:
        tot[s["kind"]] += s["wall"]
    T = sum(tot.values())
    print("engine time over all windows: " + ", ".join(
        f"{k} {v / T * 100:.1f}%" for k, v in tot.most_common()))


if __name__ == "__main__":
    main()

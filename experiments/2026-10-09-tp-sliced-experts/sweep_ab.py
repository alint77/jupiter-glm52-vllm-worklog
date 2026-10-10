"""EP vs tp_sliced on the interactivity sweep (chain_sweep.sh sweep): per
context and requests in flight, per-user decode tok/s, output tok/s per GPU,
step time and acceptance, each kind averaged over its arms; step-time
difference paired within holds. The chart's EP run (m32-sweep/
mtp3-c8-400k-pool1600k) is the reference line.

    sweep_ab.py
"""
import collections
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path("/e/fscratch/profound/naeimitabiei1/m32-sweep")
GPUS = 4
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2026-09-26-mimo-routing-profile"))


def load(d):
    f = d / "sweep.jsonl"
    return [json.loads(l) for l in f.read_text().splitlines() if l.strip()] if f.exists() else []


def cells(rows):
    out = {}
    for key in sorted({(r["ctx_target"], r["n"]) for r in rows}):
        rs = [r for r in rows if (r["ctx_target"], r["n"]) == key]
        out[key] = dict(user=st.mean(r["decode_tps"] for r in rs),
                        gpu=st.mean(r["agg_tps"] for r in rs) / GPUS,
                        step=st.mean(r["step_ms"] for r in rs),
                        acc=st.mean(r["acc_len"] for r in rs))
    return out


def collect():
    """{kind: {(hold, i): cells}}, {kind: mean cells}, chart reference cells."""
    arms = collections.defaultdict(dict)
    for d in sorted(ROOT.glob("mtp3-c8-pool1600k-*-*-*")):
        _, _, _, kind, job, i = d.name.split("-")
        rows = load(d)
        if rows and {r["n"] for r in rows} >= {1, 2, 4, 8}:
            arms[kind][(job, i)] = cells(rows)
    agg = {k: {key: {m: st.mean(c[key][m] for c in v.values()) for m in ("user", "gpu", "step", "acc")}
               for key in next(iter(v.values()))} for k, v in arms.items()}
    return arms, agg, cells(load(ROOT / "mtp3-c8-400k-pool1600k"))


def paired(arms, key):
    diffs = []
    for job in {j for j, _ in arms["ep"]} & {j for j, _ in arms["sl"]}:
        pe = st.mean(c[key]["step"] for (jj, _), c in arms["ep"].items() if jj == job)
        ps = st.mean(c[key]["step"] for (jj, _), c in arms["sl"].items() if jj == job)
        diffs.append(ps - pe)
    se = st.stdev(diffs) / len(diffs) ** 0.5 if len(diffs) > 1 else float("nan")
    return (st.mean(diffs) if diffs else float("nan")), se, len(diffs)


def main():
    arms, agg, ref = collect()
    print({k: len(v) for k, v in arms.items()})
    if not {"ep", "sl"} <= arms.keys():
        return
    print(f"{'ctx':>5} {'n':>2} | {'tok/s/user ep  sl  chart':>24} | {'tok/s/GPU ep  sl':>17} | "
          f"{'acc ep  sl':>10} | step ms ep   sl   sl-ep (paired over holds, +-se)")
    for key in sorted(agg["ep"]):
        e, s, r = agg["ep"][key], agg["sl"][key], ref.get(key, {})
        dm, se, n = paired(arms, key)
        print(f"{key[0] // 1000:>4}K {key[1]:>2} | {e['user']:7.1f} {s['user']:6.1f} {r.get('user', 0):6.1f} | "
              f"{e['gpu']:7.1f} {s['gpu']:6.1f} | {e['acc']:4.2f} {s['acc']:4.2f} | "
              f"{e['step']:6.2f} {s['step']:6.2f} {dm:+6.2f} +-{se:.2f} [{n}] "
              f"({(s['step'] / e['step'] - 1) * 100:+.1f}%)")


if __name__ == "__main__":
    main()

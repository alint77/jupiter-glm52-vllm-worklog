"""Aggregate a py-spy speedscope recording (--idle --threads): per thread, share
of samples by leaf frame (where the thread is) and by inclusive frame (what it
is inside), idle leaves (waits, polls, socket reads) split out.

    pyspy_agg.py pyspy-enginecore.json [--top 25] [--thread SUBSTR]
                 [--skip-idle] [--from S --to S]
"""
import argparse
import collections
import json
from pathlib import Path

IDLE = {"wait", "select", "poll", "epoll", "_recv", "recv", "recv_into", "acquire",
        "sleep", "get", "_poll", "readinto", "accept", "do_wait", "_wait_for_tstate_lock",
        "run_forever", "_run_once", "recv_multipart", "_worker", "join", "read"}
SPIN = {"sched_yield", "acquire_read", "check", "timeout_ms", "should_warn",
        "acquire_write"}


def short(f):
    p = Path(f["file"])
    tail = "/".join(p.parts[-2:]) if p.parts else f["file"]
    return f"{f['name']} ({tail}:{f['line']})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--thread", default="")
    ap.add_argument("--skip-idle", action="store_true")
    ap.add_argument("--from", dest="t_from", type=float, default=0.0)
    ap.add_argument("--to", type=float, default=1e9)
    a = ap.parse_args()
    d = json.load(open(a.json))
    frames = d["shared"]["frames"]
    for p in d["profiles"]:
        if a.thread not in p["name"]:
            continue
        n = len(p["samples"])
        dur = p["endValue"] - p["startValue"]
        dt = dur / max(n, 1)
        leaf = collections.Counter()
        incl = collections.Counter()
        busy = 0
        for i, s in enumerate(p["samples"]):
            t = i * dt
            if not a.t_from <= t < a.to:
                continue
            if not s:
                leaf["<no python frame>"] += 1
                continue
            lf = frames[s[-1]]
            spin = lf["name"] in SPIN
            idle = lf["name"] in IDLE or spin
            if idle and a.skip_idle:
                continue
            leaf[("SPIN " if spin else "IDLE " if idle else "") + short(lf)] += 1
            if not idle:
                busy += 1
                for fi in set(s):
                    incl[short(frames[fi])] += 1
        tot = sum(leaf.values())
        if not tot:
            continue
        print(f"\n=== {p['name']}  {dur:.1f}s, {n} samples, busy (non-idle leaf) "
              f"{100 * busy / tot:.1f}%")
        print("  leaf:")
        for k, v in leaf.most_common(a.top):
            print(f"    {100 * v / tot:5.1f}%  {k}")
        if busy:
            print("  inclusive (busy samples, % of all):")
            for k, v in incl.most_common(a.top):
                print(f"    {100 * v / tot:5.1f}%  {k}")


if __name__ == "__main__":
    main()

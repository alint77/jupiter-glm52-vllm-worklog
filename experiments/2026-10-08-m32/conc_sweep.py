"""Decode speed with n = 1, 2, 4, 8 (<= --max-conc) concurrent requests, at
~5K and ~50K tokens of context: distinct prompts per request (slices of this
repo's source, prefilled and cached first), 400 forced output tokens,
temperature 1.0 / top_p 0.95, streamed (../2026-10-08-c2-k3/conc_probe.py's
stream / counters). Per case: per-request decode tok/s, aggregate tok/s,
acceptance length, step time (acceptance / per-request tok/s), TTFT.

    conc_sweep.py --max-conc 8 [--reps 3] --out sweep.jsonl
"""
import argparse
import json
import statistics as st
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2026-10-08-c2-k3"))
from conc_probe import call, prompt, spec_counters, stream  # noqa: E402

TAIL = "\n\nExplain in detail how the code above is organised."


def run_case(name, texts, seeds):
    outs = [dict() for _ in texts]
    c0 = spec_counters()
    th = [threading.Thread(target=stream, args=(name, t, s, o)) for t, s, o in zip(texts, seeds, outs)]
    t0 = time.time()
    for t in th:
        t.start()
    for t in th:
        t.join()
    wall = time.time() - t0
    c1 = spec_counters()
    drafts = c1["num_drafts"] - c0["num_drafts"]
    acc = 1 + (c1["num_accepted_tokens"] - c0["num_accepted_tokens"]) / max(drafts, 1)
    return outs, sum(o["tokens"] for o in outs) / wall, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-conc", type=int, required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    name = json.loads(call("/v1/models", method="GET").read())["data"][0]["id"]
    corpus = "".join(f"\n### {f}\n{f.read_text(errors='ignore')}"
                     for f in sorted(Path("vllm").rglob("*.py")))
    ns = [n for n in (1, 2, 4, 8) if n <= a.max_conc]
    k = max(ns)
    prompts = {5000: [prompt(name, 5000, corpus[i * 1_000_000:]) for i in range(k)],
               50000: [prompt(name, 50000, corpus[(i * 3 + 10) * 1_000_000:]) for i in range(k)]}
    for ps in prompts.values():
        for t in ps:  # prefill + cache
            call("/v1/chat/completions", {"model": name, "max_tokens": 1, "messages": [
                {"role": "user", "content": "Below is source code from a project.\n" + t + TAIL}]}).read()
    rows = []
    with open(a.out, "w") as f:
        for rep in range(a.reps):
            for ctx, ps in prompts.items():
                for n in ns:
                    outs, agg, acc = run_case(name, ps[:n], [rep * 100 + i for i in range(n)])
                    for o in outs:
                        o.update(rep=rep, ctx_target=ctx, n=n, agg_tps=agg, acc_len=acc,
                                 step_ms=1000 * acc / o["decode_tps"])
                        f.write(json.dumps(o) + "\n")
                        rows.append(o)
                    f.flush()
    print(f"{'ctx':>6s} {'n':>2s} {'tok/s per req':>13s} {'aggregate':>9s} {'acc len':>7s} "
          f"{'step ms':>7s} {'TTFT s':>6s}")
    for ctx in prompts:
        for n in ns:
            rs = [r for r in rows if r["ctx_target"] == ctx and r["n"] == n]
            print(f"{ctx:6d} {n:2d} {st.mean(r['decode_tps'] for r in rs):13.1f} "
                  f"{st.mean(r['agg_tps'] for r in rs):9.1f} {st.mean(r['acc_len'] for r in rs):7.2f} "
                  f"{st.mean(r['step_ms'] for r in rs):7.2f} {st.mean(r['ttft'] for r in rs):6.2f}")


if __name__ == "__main__":
    main()

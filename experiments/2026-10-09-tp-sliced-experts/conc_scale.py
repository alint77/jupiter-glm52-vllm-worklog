"""Aggregate decode throughput past c=8: conc_sweep.py's cases (distinct
cached prompts, 400 forced tokens, temperature 1.0, per-case acceptance) at
more requests in flight: n in --ns at ~5K and n in --ns50 at ~50K context.

    conc_scale.py --out scale.jsonl [--ns 8,16,24,32,48,64] [--ns50 8,16,24] [--reps 2]
"""
import argparse
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2026-10-08-m32"))
from conc_sweep import TAIL, call, prompt, run_case  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ns", default="8,16,24,32,48,64")
    ap.add_argument("--ns50", default="8,16,24")
    ap.add_argument("--reps", type=int, default=2)
    a = ap.parse_args()
    ns = {5000: [int(x) for x in a.ns.split(",")], 50000: [int(x) for x in a.ns50.split(",")]}
    name = json.loads(call("/v1/models", method="GET").read())["data"][0]["id"]
    corpus = "".join(f"\n### {f}\n{f.read_text(errors='ignore')}"
                     for f in sorted(Path("vllm").rglob("*.py")))
    prompts = {5000: [prompt(name, 5000, corpus[i * 500_000:]) for i in range(max(ns[5000]))],
               50000: [prompt(name, 50000, corpus[5_000_000 + i * 1_000_000:])
                       for i in range(max(ns[50000]))]}
    for ps in prompts.values():
        for t in ps:
            call("/v1/chat/completions", {"model": name, "max_tokens": 1, "messages": [
                {"role": "user", "content": "Below is source code from a project.\n" + t + TAIL}]}).read()
    rows = []
    with open(a.out, "w") as f:
        for rep in range(a.reps):
            for ctx, ps in prompts.items():
                for n in ns[ctx]:
                    outs, agg, acc = run_case(name, ps[:n], [rep * 1000 + i for i in range(n)])
                    for o in outs:
                        o.update(rep=rep, ctx_target=ctx, n=n, agg_tps=agg, acc_len=acc,
                                 step_ms=1000 * acc / o["decode_tps"])
                        f.write(json.dumps(o) + "\n")
                        rows.append(o)
                    f.flush()
                    print(f"{ctx:6d} n={n:3d} agg {agg:7.1f} tok/s ({agg / 4:6.1f}/GPU) per-user "
                          f"{st.mean(o['decode_tps'] for o in outs):6.1f} acc {acc:.2f} step "
                          f"{st.mean(o['step_ms'] for o in outs):6.2f} ms", flush=True)


if __name__ == "__main__":
    main()

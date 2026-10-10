"""Is the sweep's 50K-over-5K speed context length or prompt content? The
sweep's 5K prompts start at corpus offsets i * 1M and its 50K prompts at
(3i + 10) * 1M. Here both offset sets at both lengths (a 5K prompt is the
first 5K tokens of the 50K one at the same offset), one request at a time,
400 tokens, the sweep's sampling: per request acceptance (counter delta),
decode tok/s and step ms.

    ctx_probe.py --out probe.jsonl [--k 8] [--reps 2]
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
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--reps", type=int, default=2)
    a = ap.parse_args()
    name = json.loads(call("/v1/models", method="GET").read())["data"][0]["id"]
    corpus = "".join(f"\n### {f}\n{f.read_text(errors='ignore')}"
                     for f in sorted(Path("vllm").rglob("*.py")))
    offs = {"A": [i * 1_000_000 for i in range(a.k)], "B": [(i * 3 + 10) * 1_000_000 for i in range(a.k)]}
    cellp = {(s, ctx): [prompt(name, ctx, corpus[o:]) for o in offs[s]]
             for s in offs for ctx in (5000, 50000)}
    for ps in cellp.values():
        for t in ps:
            call("/v1/chat/completions", {"model": name, "max_tokens": 1, "messages": [
                {"role": "user", "content": "Below is source code from a project.\n" + t + TAIL}]}).read()
    rows = []
    with open(a.out, "w") as f:
        for rep in range(a.reps):
            for (s, ctx), ps in cellp.items():
                for i, t in enumerate(ps):
                    outs, _, acc = run_case(name, [t], [rep * 100 + i])
                    o = outs[0]
                    o.update(set=s, ctx_target=ctx, prompt=i, rep=rep, acc_len=acc,
                             step_ms=1000 * acc / o["decode_tps"])
                    f.write(json.dumps(o) + "\n")
                    f.flush()
                    rows.append(o)
    print(f"{'set':>3} {'ctx':>6} {'acc':>5} {'step ms':>7} {'tok/s':>6}")
    for (s, ctx) in cellp:
        rs = [r for r in rows if r["set"] == s and r["ctx_target"] == ctx]
        print(f"{s:>3} {ctx:6d} {st.mean(r['acc_len'] for r in rs):5.2f} "
              f"{st.mean(r['step_ms'] for r in rs):7.2f} {st.mean(r['decode_tps'] for r in rs):6.1f}")


if __name__ == "__main__":
    main()

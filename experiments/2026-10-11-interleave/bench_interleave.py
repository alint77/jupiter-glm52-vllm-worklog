"""Prefill/decode interleaving windows (c4 config), each run twice on fresh text:

- mixed:   3 decoders at ~10K context (cached) stream 1500 tokens; after 4 s a
           delta8k@60k prefill (cached 60K prefix + 8K new, 1 output token).
           Prefill TTFT; decoder tok/s and longest gap while it prefills.
- swarm:   4 fresh 24K prompts arriving 1 s apart, 300 tokens each.
- fanout:  4 prompts at once sharing a cached 20K prefix plus 4K new each,
           300 tokens each (subagents spawned together).

    bench_interleave.py --out bench.jsonl
"""
import argparse
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2026-10-08-c2-k3"))
from conc_probe import call, prompt  # noqa: E402

HEAD = "Below is source code from a project.\n"
TAIL = "\n\nExplain in detail how the code above is organised."


def stream(name, text, seed, out, max_tokens):
    body = {"model": name, "max_tokens": max_tokens, "ignore_eos": True, "temperature": 1.0,
            "top_p": 0.95, "seed": seed, "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": HEAD + text + TAIL}]}
    t0 = time.time()
    arrivals, usage = [], None
    with call("/v1/chat/completions", body) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            if d.get("usage"):
                usage = d["usage"]
            if d.get("choices") and any(v for k, v in d["choices"][0].get("delta", {}).items()
                                        if k != "role"):
                arrivals.append(time.time())
    toks = usage["completion_tokens"]
    out.update(t0=t0, ttft=arrivals[0] - t0, end=arrivals[-1], arrivals=arrivals, tokens=toks,
               ctx=usage["prompt_tokens"],
               decode_tps=(toks - 1) / (arrivals[-1] - arrivals[0]) if toks > 1 else None)


def run(name, jobs):
    """jobs: (delay_s, text, max_tokens); returns the outs, started at delays."""
    outs, threads = [], []
    t0 = time.time()
    for i, (delay, text, mt) in enumerate(jobs):
        time.sleep(max(0.0, t0 + delay - time.time()))
        o = {}
        th = threading.Thread(target=stream, args=(name, text, i, o, mt))
        th.start()
        outs.append(o)
        threads.append(th)
    for th in threads:
        th.join()
    return outs


def summary(o):
    return {k: o[k] for k in ("ttft", "decode_tps", "tokens", "ctx")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    name = json.loads(call("/v1/models", method="GET").read())["data"][0]["id"]
    corpus = "".join(f"\n### {f}\n{f.read_text(errors='ignore')}"
                     for f in sorted(Path("vllm").rglob("*.py")))
    mb = 1_000_000
    warm = lambda text: run(name, [(0, text, 1)])  # noqa: E731
    rows = []

    def emit(row):
        rows.append(row)
        print(json.dumps(row), flush=True)

    base60 = prompt(name, 60000, corpus)
    base20 = prompt(name, 20000, corpus[int(1.5 * mb):])
    warm(base60)
    warm(base20)
    for rep in range(2):
        off = (2 + 10 * rep) * mb
        ctx = [prompt(name, 10000, corpus[off + i * mb // 2:]) for i in range(3)]
        for t in ctx:
            warm(t)
        delta = prompt(name, 8000, corpus[off + 2 * mb:])
        outs = run(name, [(0, t, 1500) for t in ctx] + [(4, base60 + delta, 1)])
        pre = outs[-1]
        p0, p1 = pre["t0"], pre["t0"] + pre["ttft"]
        dec = []
        for o in outs[:-1]:
            inside = [t for t in o["arrivals"] if p0 <= t <= p1]
            span = [t for t in o["arrivals"] if p0 - 1 <= t <= p1 + 1]
            gaps = [b - a for a, b in zip(span, span[1:])]
            dec.append({"chunks_in_prefill": len(inside), "max_gap_s": max(gaps, default=None),
                        **summary(o)})
        emit({"window": "mixed", "rep": rep, "prefill_ttft": pre["ttft"],
              "prefill_ctx": pre["ctx"], "decoders": dec})

        fresh = [prompt(name, 24000, corpus[off + (3 + i) * mb // 2:]) for i in range(4)]
        outs = run(name, [(i, t, 300) for i, t in enumerate(fresh)])
        t0 = min(o["t0"] for o in outs)
        emit({"window": "swarm", "rep": rep, "makespan": max(o["end"] for o in outs) - t0,
              "reqs": [summary(o) for o in outs]})

        news = [prompt(name, 4000, corpus[off + (6 + i) * mb // 2:]) for i in range(4)]
        outs = run(name, [(0, base20 + t, 300) for t in news])
        t0 = min(o["t0"] for o in outs)
        emit({"window": "fanout", "rep": rep, "makespan": max(o["end"] for o in outs) - t0,
              "reqs": [summary(o) for o in outs]})
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()

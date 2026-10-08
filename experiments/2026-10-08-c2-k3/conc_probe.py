"""Per-request decode speed alone vs in pairs: chat requests over ~5K / ~50K
tokens of this repo's source (each prompt prefilled and cached first), 400
output tokens forced (ignore_eos), temperature 1.0 / top_p 0.95, streamed;
decode tok/s = (tokens - 1) / (last chunk - first chunk); mean acceptance
length per case from the server's spec-decode counters. --quad adds 4-way
cases; --with-profile first takes the --profile-only windows (not counted).

    conc_probe.py --out probe.jsonl [--quad] [--profile | --profile-only | --with-profile]
"""
import argparse
import json
import threading
import time
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8027"


def call(path, body=None, method="POST"):
    req = urllib.request.Request(f"{BASE}{path}", method=method,
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=3600)


def prompt(name, n, corpus):
    chars = n * 3
    while True:
        text = corpus[:chars]
        got = json.loads(call("/tokenize", {"model": name, "prompt": text}).read())["count"]
        if abs(got - n) < 0.02 * n:
            return text
        chars = int(chars * n / got)


def stream(name, text, seed, out, max_tokens=400):
    body = {"model": name, "max_tokens": max_tokens, "ignore_eos": True, "temperature": 1.0,
            "top_p": 0.95, "seed": seed, "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": "Below is source code from a project.\n" + text
                          + "\n\nExplain in detail how the code above is organised."}]}
    t0 = time.time()
    first = last = None
    usage = None
    with call("/v1/chat/completions", body) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            if d.get("usage"):
                usage = d["usage"]
            # any text the server streams (content, reasoning under whichever
            # key the reasoning parser uses, tool-call arguments)
            if d.get("choices") and any(v for k, v in d["choices"][0].get("delta", {}).items()
                                        if k != "role"):
                last = time.time()
                first = first or last
    toks = usage["completion_tokens"]
    assert first is not None and last > first, ("no streamed tokens", usage)
    out.update(ttft=first - t0, decode_tps=(toks - 1) / (last - first), tokens=toks,
               ctx=usage["prompt_tokens"], wall=time.time() - t0)


def spec_counters():
    got = {}
    for line in call("/metrics", method="GET").read().decode().splitlines():
        for k in ("num_drafts", "num_accepted_tokens"):
            if line.startswith(f"vllm:spec_decode_{k}_total"):
                got[k] = got.get(k, 0.0) + float(line.split()[-1])
    return got


def case(name, label, texts, seeds, rep, profile=False, max_tokens=400, trace_root=None):
    c0 = spec_counters()
    outs = [dict() for _ in texts]
    th = [threading.Thread(target=stream, args=(name, t, s, o, max_tokens))
          for t, s, o in zip(texts, seeds, outs)]
    t0 = time.time()
    for t in th:
        t.start()
    if profile:
        before = set(Path(trace_root).glob("*.gz"))
        time.sleep(1)
        call("/start_profile").read()
        time.sleep(2)
        call("/stop_profile").read()
        time.sleep(5)
        dest = Path(trace_root) / label.replace(" ", "-").replace("+", "_")
        dest.mkdir(parents=True, exist_ok=True)
        for f in set(Path(trace_root).glob("*.gz")) - before:
            f.replace(dest / f.name)
    for t in th:
        t.join()
    wall = time.time() - t0
    agg = sum(o["tokens"] for o in outs) / wall
    c1 = spec_counters()
    drafts = c1.get("num_drafts", 0) - c0.get("num_drafts", 0)
    acc = 1 + (c1.get("num_accepted_tokens", 0) - c0.get("num_accepted_tokens", 0)) / max(drafts, 1)
    for o in outs:
        o.update(case=label, rep=rep, n=len(texts), agg_tps=agg, acc_len=acc)
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--profile-only", action="store_true",
                    help="only two profiled windows (lone 4-token step, 5K+5K pair), 1200 tokens")
    ap.add_argument("--with-profile", action="store_true")
    ap.add_argument("--quad", action="store_true", help="also 4 concurrent requests")
    ap.add_argument("--trace-root")
    a = ap.parse_args()
    name = json.loads(call("/v1/models", method="GET").read())["data"][0]["id"]
    corpus = "".join(f"\n### {f}\n{f.read_text(errors='ignore')}" for f in sorted(Path("vllm").rglob("*.py")))
    p5a, p5b = prompt(name, 5000, corpus), prompt(name, 5000, corpus[700000:])
    p50a, p50b = prompt(name, 50000, corpus), prompt(name, 50000, corpus[900000:])
    q5 = [p5a, p5b] + ([prompt(name, 5000, corpus[o:]) for o in (1400000, 2100000)] if a.quad else [])
    q50 = [p50a, p50b] + ([prompt(name, 50000, corpus[o:]) for o in (3000000, 4000000)] if a.quad else [])
    for t in [p5a, p5b, p50a, p50b] + (q5[2:] + q50[2:] if a.quad else []):  # prefill + cache
        call("/v1/chat/completions", {
            "model": name, "max_tokens": 1, "messages": [{"role": "user", "content":
            "Below is source code from a project.\n" + t + "\n\nExplain in detail how the code above is organised."}]}).read()
    rows = []
    if a.profile_only or a.with_profile:
        prof = case(name, "alone 5K", [p5a], [0], 0, True, 1200, a.trace_root)
        prof += case(name, "pair 5K+5K", [p5a, p5b], [0, 10], 0, True, 1200, a.trace_root)
        if a.quad:
            prof += case(name, "quad 5K", q5, [0, 10, 20, 30], 0, True, 1200, a.trace_root)
        if a.profile_only:
            rows += prof
    for rep in range(0 if a.profile_only else 2):
        rows += case(name, "alone 5K", [p5a], [rep], rep)
        rows += case(name, "alone 50K", [p50a], [rep], rep)
        rows += case(name, "pair 5K+5K", [p5a, p5b], [rep, rep + 10], rep, profile=a.profile and rep == 1)
        rows += case(name, "pair 50K+50K", [p50a, p50b], [rep, rep + 10], rep)
        rows += case(name, "pair 5K+50K", [p5a, p50b], [rep, rep + 10], rep)
        if a.quad:
            seeds = [rep, rep + 10, rep + 20, rep + 30]
            rows += case(name, "quad 5K", q5, seeds, rep)
            rows += case(name, "quad 50K", q50, seeds, rep)
            rows += case(name, "quad 2x5K+2x50K", q5[:2] + q50[2:], seeds, rep)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    import collections
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["case"], round(r["ctx"], -3))].append(r)
    print(f"{'case':16s} {'ctx':>7s} {'decode tok/s per req':>21s} {'aggregate tok/s':>16s} {'acc len':>8s}")
    for (c, ctx), rs in by.items():
        print(f"{c:16s} {ctx:7.0f} {sum(r['decode_tps'] for r in rs) / len(rs):21.1f} "
              f"{sum(r['agg_tps'] for r in rs) / len(rs):16.1f} {sum(r['acc_len'] for r in rs) / len(rs):8.2f}")


if __name__ == "__main__":
    main()

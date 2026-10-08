"""Draft acceptance on a fixed request set: 8 sequential chat requests over
~3K-token slices of this repo's source (prefilled first), 400 forced output
tokens, temperature 1.0 / top_p 0.95, fixed seeds. Prints mean acceptance
length (server spec-decode counters) and mean per-request decode tok/s.

    acc_probe.py [--n 8] [--out probe.json]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2026-10-08-c2-k3"))
from conc_probe import call, prompt, spec_counters, stream  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out")
    a = ap.parse_args()
    name = json.loads(call("/v1/models", method="GET").read())["data"][0]["id"]
    corpus = "".join(f"\n### {f}\n{f.read_text(errors='ignore')}"
                     for f in sorted(Path("vllm").rglob("*.py")))
    texts = [prompt(name, 3000, corpus[i * 1_500_000:]) for i in range(a.n)]
    for t in texts:
        call("/v1/chat/completions", {"model": name, "max_tokens": 1, "messages": [
            {"role": "user", "content": "Below is source code from a project.\n" + t
             + "\n\nExplain in detail how the code above is organised."}]}).read()
    c0, rows = spec_counters(), []
    for i, t in enumerate(texts):
        o = {}
        stream(name, t, 100 + i, o)
        rows.append(o)
    c1 = spec_counters()
    drafts = c1["num_drafts"] - c0["num_drafts"]
    acc = 1 + (c1["num_accepted_tokens"] - c0["num_accepted_tokens"]) / drafts
    tps = sum(r["decode_tps"] for r in rows) / len(rows)
    print(f"acceptance length {acc:.3f} over {drafts:.0f} drafts | "
          f"decode {tps:.1f} tok/s per request | per request: "
          + " ".join(f"{r['decode_tps']:.0f}" for r in rows))
    if a.out:
        Path(a.out).write_text(json.dumps({"acc_len": acc, "decode_tps": tps, "rows": rows}))


main()

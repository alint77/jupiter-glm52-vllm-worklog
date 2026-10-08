"""Profiled decode windows with n = 1, 4, 8 requests in flight at ~5K context
and 8 at ~50K: distinct cached prompts, 1200 tokens each, a 2 s torch-profiler
window 1 s after the requests start (../2026-10-08-c2-k3/conc_probe.case),
traces moved to <trace_root>/<n>x<ctx>.

    prof_load.py --trace-root DIR
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2026-10-08-c2-k3"))
from conc_probe import call, case, prompt  # noqa: E402

TAIL = "\n\nExplain in detail how the code above is organised."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-root", required=True)
    a = ap.parse_args()
    name = json.loads(call("/v1/models", method="GET").read())["data"][0]["id"]
    corpus = "".join(f"\n### {f}\n{f.read_text(errors='ignore')}"
                     for f in sorted(Path("vllm").rglob("*.py")))
    p5 = [prompt(name, 5000, corpus[i * 1_000_000:]) for i in range(8)]
    p50 = [prompt(name, 50000, corpus[(i * 3 + 10) * 1_000_000:]) for i in range(8)]
    for t in p5 + p50:
        call("/v1/chat/completions", {"model": name, "max_tokens": 1, "messages": [
            {"role": "user", "content": "Below is source code from a project.\n" + t + TAIL}]}).read()
    for label, texts in (("1x5K", p5[:1]), ("4x5K", p5[:4]), ("8x5K", p5), ("8x50K", p50)):
        outs = case(name, label, texts, list(range(len(texts))), 0, True, 1200, a.trace_root)
        print(label, f"agg {outs[0]['agg_tps']:.1f} tok/s, acc {outs[0]['acc_len']:.2f}, "
              f"per-req {sum(o['decode_tps'] for o in outs) / len(outs):.1f}", flush=True)


if __name__ == "__main__":
    main()

"""Profiled c=1 decode windows for the EP vs tp_sliced A/B: one request at
~5K and one at ~50K context (distinct cached prompts, 1200 tokens), a 2 s
torch-profiler window 1 s in, traces moved to <trace_root>/<label>.

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
    shapes = (("1x5K", prompt(name, 5000, corpus)),
              ("1x50K", prompt(name, 50000, corpus[10_000_000:])))
    for _, t in shapes:
        call("/v1/chat/completions", {"model": name, "max_tokens": 1, "messages": [
            {"role": "user", "content": "Below is source code from a project.\n" + t + TAIL}]}).read()
    for label, t in shapes:
        outs = case(name, label, [t], [0], 0, True, 1200, a.trace_root)
        print(label, f"{outs[0]['agg_tps']:.1f} tok/s, acc {outs[0]['acc_len']:.2f}", flush=True)


if __name__ == "__main__":
    main()

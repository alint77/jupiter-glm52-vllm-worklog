"""Un-profiled TTFT of pure prefill steps (max_tokens 1, no decoders) at
several sizes, fresh (context = N) and N new tokens after a cached 60K prefix.
Sizes are interleaved within each round so node drift hits all of them alike.

    ttft_sizes.py --sizes 320,384,448,512 --reps 8 --out ttft.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2026-10-08-c2-k3"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2026-10-11-prefill-sizes"))
from conc_probe import call, prompt  # noqa: E402
from prof_sizes import ttft  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", required=True)
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sizes = [int(s) for s in a.sizes.split(",")]
    name = json.loads(call("/v1/models", method="GET").read())["data"][0]["id"]
    corpus = "".join(f"\n### {f}\n{f.read_text(errors='ignore')}"
                     for f in sorted(Path("vllm").rglob("*.py")))
    mb = 1_000_000
    base = prompt(name, 60000, corpus)
    ttft(name, base)
    off = 1
    with open(a.out, "w") as f:
        for r in range(a.reps + 1):
            for n in sizes:
                for ctx in ("fresh", "@60k"):
                    t = prompt(name, n, corpus[off * mb // 4:])
                    off += 1
                    s, k = ttft(name, t if ctx == "fresh" else base + t)
                    row = {"label": f"{ctx}{n}", "rep": r, "ttft_s": s,
                           "prompt_tokens": k}
                    print(json.dumps(row), flush=True)
                    if r:
                        f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()

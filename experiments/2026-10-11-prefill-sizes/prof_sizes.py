"""One pure prefill step per Nsight Systems window (served with
VLLM_SCHED_INTERLEAVE_PREFILL=1, no decoders, max_tokens 1), at several step
sizes: fresh prompts of ~N tokens (context = N) and N new tokens after a cached
60K prefix. Each size is first timed twice un-profiled on fresh text; the
profiled windows run in the order printed (nsys.<i> = i-th profiled window).

    prof_sizes.py --sizes 500,1000,2000,4000 --out timings.jsonl
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2026-10-08-c2-k3"))
from conc_probe import call, prompt  # noqa: E402

HEAD = "Below is source code from a project.\n"
TAIL = "\n\nExplain in detail how the code above is organised."


def ttft(name, text):
    t0 = time.time()
    r = json.loads(call("/v1/chat/completions", {"model": name, "max_tokens": 1, "messages": [
        {"role": "user", "content": HEAD + text + TAIL}]}).read())
    return time.time() - t0, r["usage"]["prompt_tokens"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sizes = [int(s) for s in a.sizes.split(",")]
    name = json.loads(call("/v1/models", method="GET").read())["data"][0]["id"]
    corpus = "".join(f"\n### {f}\n{f.read_text(errors='ignore')}"
                     for f in sorted(Path("vllm").rglob("*.py")))
    mb = 1_000_000
    base = prompt(name, 60000, corpus)
    ttft(name, base)
    cases = []
    off = 1
    for n in sizes:
        for ctx in ("fresh", "@60k"):
            texts = []
            for _ in range(3):
                t = prompt(name, n, corpus[off * mb // 4:])
                texts.append(t if ctx == "fresh" else base + t)
                off += 1
            cases.append((f"{ctx}{n}", texts))
    rows = []

    def emit(row):
        rows.append(row)
        print(json.dumps(row), flush=True)

    for label, texts in cases:
        for t in texts[:2]:
            s, n = ttft(name, t)
            emit({"label": label, "profiled": False, "ttft_s": s, "prompt_tokens": n})
    for i, (label, texts) in enumerate(cases, 1):
        call("/start_profile").read()
        s, n = ttft(name, texts[2])
        time.sleep(0.3)
        call("/stop_profile").read()
        time.sleep(3)
        emit({"label": label, "profiled": True, "window": i, "ttft_s": s, "prompt_tokens": n})
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()

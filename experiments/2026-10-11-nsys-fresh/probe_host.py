"""Where the ~105 ms of non-GPU TTFT at 60K context goes (../2026-10-11-
prefill-sizes finding 1). Per prompt size, one request at a time (idle
server), after a warm-up request that caches the prefix:
- client TTFT (max_tokens 1, streamed);
- server-side deltas of vllm:time_to_first_token / request_queue_time /
  request_prefill_time (sum over the request);
- /tokenize of the same chat messages (chat template + tokenizer, API server);
- /tokenize of the raw prompt string.

    probe_host.py --out host.jsonl
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
METRICS = ("vllm:time_to_first_token_seconds_sum", "vllm:request_queue_time_seconds_sum",
           "vllm:request_prefill_time_seconds_sum", "vllm:e2e_request_latency_seconds_sum")


def metrics():
    got = {}
    for line in call("/metrics", method="GET").read().decode().splitlines():
        for k in METRICS:
            if line.startswith(k):
                got[k] = got.get(k, 0.0) + float(line.split()[-1])
    return got


def ttft(name, text, suffix=""):
    body = {"model": name, "max_tokens": 1, "stream": True, "temperature": 0.0,
            "messages": [{"role": "user", "content": HEAD + text + TAIL + suffix}]}
    t0 = time.time()
    with call("/v1/chat/completions", body) as r:
        for line in r:
            if line.decode().startswith("data: {"):
                d = json.loads(line.decode()[6:])
                if d.get("choices") and any(v for k, v in d["choices"][0].get("delta", {}).items()
                                            if k != "role"):
                    t1 = time.time()
    return t1 - t0


def tok_time(name, text, chat):
    body = ({"model": name, "messages": [{"role": "user", "content": HEAD + text + TAIL}]}
            if chat else {"model": name, "prompt": HEAD + text + TAIL})
    t0 = time.time()
    n = json.loads(call("/tokenize", body).read())["count"]
    return time.time() - t0, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    name = json.loads(call("/v1/models", method="GET").read())["data"][0]["id"]
    corpus = "".join(f"\n### {f}\n{f.read_text(errors='ignore')}"
                     for f in sorted(Path("vllm").rglob("*.py")))
    rows = []
    for i, n in enumerate((1000, 10000, 30000, 60000, 100000)):
        text = prompt(name, n, corpus[i * 1_500_000:])
        ttft(name, text)  # cache the prefix
        for rep in range(3):
            m0 = metrics()
            # a new short suffix each time: cached prefix + a small delta
            c = ttft(name, text, f"\nVariant {rep}.")
            time.sleep(0.5)
            m1 = metrics()
            d = {k.split(":")[1].replace("_seconds_sum", ""): m1[k] - m0.get(k, 0.0)
                 for k in METRICS if k in m1}
            tc, ntok = tok_time(name, text, True)
            tr, _ = tok_time(name, text, False)
            row = {"tokens": ntok, "rep": rep, "client_ttft": c, **d,
                   "tokenize_chat": tc, "tokenize_raw": tr}
            rows.append(row)
            print(json.dumps({k: round(v, 4) if isinstance(v, float) else v
                              for k, v in row.items()}), flush=True)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()

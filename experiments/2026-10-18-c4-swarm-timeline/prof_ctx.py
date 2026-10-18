"""Long-context engine windows at the c=4 config (prof_load_n.py at the
swarm's contexts), one Nsight Systems report per window under NSYS_OUT:
  decode  for each ctx (K tokens) with n * ctx inside the 800K pool: n
          distinct cached prompts decoding 1200 tokens, a 2 s window 1 s in
  prefill (--prefill K) one fresh K-token prompt alone, 2 s windows when its
          chunked prefill passes ~1/3, ~2/3 and the end of the context
          (chunk rate from a 4K-token step at the start)

    prof_ctx.py --n N --ctx 100,190 --trace-root DIR [--prefill 300]
"""
import argparse
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2026-10-08-c2-k3"))
from conc_probe import call, case, prompt  # noqa: E402

TAIL = "\n\nExplain in detail how the code above is organised."
POOL_K = 780


def msgs(t):
    return [{"role": "user", "content": "Below is source code from a project.\n" + t + TAIL}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--ctx", default="")
    ap.add_argument("--prefill", type=int, default=0)
    ap.add_argument("--trace-root", required=True)
    a = ap.parse_args()
    name = json.loads(call("/v1/models", method="GET").read())["data"][0]["id"]
    corpus = "".join(f"\n### {f}\n{f.read_text(errors='ignore')}"
                     for f in sorted(Path("vllm").rglob("*.py")))
    off = 0
    for k in [int(x) for x in a.ctx.split(",") if x]:
        if a.n * k > POOL_K:
            print(f"skip {a.n}x{k}K (> pool)", flush=True)
            continue
        texts = []
        for _ in range(a.n):
            texts.append(prompt(name, k * 1000, corpus[off:]))
            off += len(texts[-1])
        for t in texts:
            call("/v1/chat/completions", {"model": name, "max_tokens": 1,
                                          "messages": msgs(t)}).read()
        label = f"{a.n}x{k}K"
        outs = case(name, label, texts, list(range(len(texts))), 0, True, 1200, a.trace_root)
        print(label, f"agg {outs[0]['agg_tps']:.1f} tok/s, acc {outs[0]['acc_len']:.2f}, "
              f"per-req {sum(o['decode_tps'] for o in outs) / len(outs):.1f}", flush=True)
    if a.prefill:
        k = a.prefill * 1000
        text = prompt(name, k, corpus[off:])
        res = {}

        def go():
            t = time.time()
            call("/v1/chat/completions", {"model": name, "max_tokens": 1,
                                          "messages": msgs(text)}).read()
            res["ttft"] = time.time() - t

        th = threading.Thread(target=go)
        t0 = time.time()
        th.start()
        # windows by fraction of an estimated total (~0.5 s per 4K chunk, rising
        # with context)
        est = 0.5 * (k / 4096) * (1 + k / 1.2e6)
        for frac in (0.33, 0.66, 0.95):
            time.sleep(max(0.0, t0 + frac * est - time.time()))
            if not th.is_alive():
                break
            s = time.time() - t0
            call("/start_profile").read()
            time.sleep(2)
            call("/stop_profile").read()
            print(f"prefill {a.prefill}K window at +{s:.1f} s", flush=True)
        th.join()
        print(f"prefill {a.prefill}K TTFT {res['ttft']:.1f} s (est {est:.1f})", flush=True)


if __name__ == "__main__":
    main()

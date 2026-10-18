"""Server-side request phases from the /metrics dumps of a timeline arm
(metrics/<t>.txt): per histogram, count / mean / p50 / p90 (bucket upper
bounds) over the replay (last dump minus first), plus counters (tokens,
prefix cache, spec decode acceptance, preemptions).

    metrics_report.py <arm out dir>
"""
import collections
import re
import sys
from pathlib import Path

HIST = ("time_to_first_token_seconds", "e2e_request_latency_seconds",
        "request_queue_time_seconds", "request_prefill_time_seconds",
        "request_decode_time_seconds", "request_inference_time_seconds",
        "inter_token_latency_seconds", "request_time_per_output_token_seconds",
        "iteration_tokens_total", "request_prompt_tokens", "request_generation_tokens",
        "request_prefill_kv_computed_tokens")
LINE = re.compile(r"^vllm:([a-z_]+?)(_bucket|_sum|_count|_total)?(\{[^}]*\})? ([0-9.eE+-]+|NaN)$")


def parse(path):
    h = collections.defaultdict(lambda: {"b": collections.Counter(), "sum": 0.0, "n": 0.0})
    c = collections.Counter()
    for line in path.open():
        m = LINE.match(line)
        if not m:
            continue
        name, suf, lab, v = m.groups()
        v = float(v) if v != "NaN" else 0.0
        if suf == "_bucket":
            le = re.search(r'le="([^"]+)"', lab).group(1)
            h[name]["b"][float(le)] += v
        elif suf == "_sum":
            h[name]["sum"] += v
        elif suf == "_count":
            h[name]["n"] += v
        else:
            c[name + (suf or "")] += v
    return h, c


def pct(b, n, p):
    for le in sorted(b):
        if b[le] >= p * n:
            return le
    return float("inf")


def main():
    d = Path(sys.argv[1]) / "metrics"
    files = sorted(d.glob("*.txt"), key=lambda p: float(p.stem))
    (h0, c0), (h1, c1) = parse(files[0]), parse(files[-1])
    print(f"{len(files)} dumps, {files[0].stem} .. {files[-1].stem} s")
    print(f"  {'histogram':<40} {'count':>6} {'mean':>9} {'p50<=':>8} {'p90<=':>8} {'p99<=':>8}")
    for name in HIST:
        if name not in h1:
            continue
        b = {le: h1[name]["b"][le] - h0[name]["b"].get(le, 0) for le in h1[name]["b"]}
        n = h1[name]["n"] - h0[name]["n"]
        s = h1[name]["sum"] - h0[name]["sum"]
        if n <= 0:
            continue
        print(f"  {name:<40} {n:6.0f} {s / n:9.3f} {pct(b, n, .5):8.3g} "
              f"{pct(b, n, .9):8.3g} {pct(b, n, .99):8.3g}")
    print("  counters (delta):")
    for k in sorted(c1):
        dv = c1[k] - c0.get(k, 0)
        if dv and any(x in k for x in ("tokens", "prefix", "spec", "preempt", "request_success")):
            print(f"    {k:<50} {dv:,.0f}")
    acc = c1.get("spec_decode_num_accepted_tokens_total", 0) - c0.get("spec_decode_num_accepted_tokens_total", 0)
    drafts = c1.get("spec_decode_num_drafts_total", 0) - c0.get("spec_decode_num_drafts_total", 0)
    if drafts:
        print(f"  acceptance length {1 + acc / drafts:.2f} tokens per verify step")


if __name__ == "__main__":
    main()

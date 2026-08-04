"""Summarize the 16K-prefill / 512-output PyTorch code-generation runs.

Reports prefill and decode separately, per concurrency, with the spread across
repetitions and the MTP3 acceptance the server recorded for the same window.
"""

import json
import statistics
from pathlib import Path

RESULT_DIR = Path(__file__).parent
ARMS = ["c1", "c4"]
REPEATS = [1, 2]


def mean(values):
    return statistics.fmean(values)


def read_metrics(name):
    path = RESULT_DIR / f"metrics-{name}.txt"
    values = {}
    for line in path.read_text().splitlines():
        key, _, value = line.rpartition(" ")
        key = key.split("{")[0].replace("vllm:", "")
        values[key] = float(value)
    return values


def acceptance(tag):
    before = read_metrics(f"{tag}-before")
    after = read_metrics(f"{tag}-after")
    draft = after["spec_decode_num_draft_tokens_total"] - (
        before["spec_decode_num_draft_tokens_total"]
    )
    accepted = after["spec_decode_num_accepted_tokens_total"] - (
        before["spec_decode_num_accepted_tokens_total"]
    )
    generated = after["generation_tokens_total"] - before["generation_tokens_total"]
    steps = generated - accepted
    return {
        "draft_tokens": draft,
        "accepted_tokens": accepted,
        "generated_tokens": generated,
        "target_steps": steps,
        "acceptance_rate": accepted / draft if draft else float("nan"),
        "tokens_per_step": generated / steps if steps else float("nan"),
    }


rows = []
for arm in ARMS:
    per_repeat = []
    for repeat in REPEATS:
        tag = f"{arm}-r{repeat}"
        result = json.loads((RESULT_DIR / f"{tag}.json").read_text())
        input_lens = result["input_lens"]
        ttfts = result["ttfts"]
        # Prefill rate per request, from that request's own input length.
        prefill_rates = [n / t for n, t in zip(input_lens, ttfts)]
        acc = acceptance(tag)
        per_repeat.append(
            {
                "tag": tag,
                "concurrency": result["max_concurrency"],
                "requests": result["completed"],
                "input_tokens_mean": mean(input_lens),
                "output_tokens_mean": mean(result["output_lens"]),
                "duration_s": result["duration"],
                "ttft_mean_s": mean(ttfts),
                "ttft_median_s": statistics.median(ttfts),
                "ttft_p99_s": result["p99_ttft_ms"] / 1000,
                "prefill_tok_s_mean": mean(prefill_rates),
                "tpot_mean_ms": result["mean_tpot_ms"],
                "tpot_median_ms": result["median_tpot_ms"],
                "itl_median_ms": result["median_itl_ms"],
                "decode_tok_s_per_request": 1000 / result["mean_tpot_ms"],
                "output_tok_s_aggregate": result["output_throughput"],
                "total_tok_s": result["total_token_throughput"],
                **acc,
            }
        )
    rows.extend(per_repeat)

summary = {"repetitions": rows, "arms": {}}
for arm in ARMS:
    arm_rows = [r for r in rows if r["tag"].startswith(f"{arm}-")]

    def spread(key):
        values = [r[key] for r in arm_rows]
        return {
            "mean": mean(values),
            "min": min(values),
            "max": max(values),
            "spread_pct": (max(values) - min(values)) / mean(values) * 100,
        }

    summary["arms"][arm] = {
        key: spread(key)
        for key in (
            "ttft_mean_s",
            "prefill_tok_s_mean",
            "tpot_mean_ms",
            "decode_tok_s_per_request",
            "output_tok_s_aggregate",
            "total_tok_s",
            "acceptance_rate",
            "tokens_per_step",
            "duration_s",
        )
    }

(RESULT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

print(f"{'':22} {'c=1':>14} {'c=4':>14}")
labels = [
    ("TTFT mean (s)", "ttft_mean_s", 3),
    ("Prefill tok/s/req", "prefill_tok_s_mean", 0),
    ("TPOT mean (ms)", "tpot_mean_ms", 3),
    ("Decode tok/s/req", "decode_tok_s_per_request", 2),
    ("Output tok/s total", "output_tok_s_aggregate", 2),
    ("Total tok/s", "total_tok_s", 2),
    ("Draft acceptance", "acceptance_rate", 4),
    ("Tokens/target step", "tokens_per_step", 3),
    ("Wall clock (s)", "duration_s", 1),
]
for label, key, digits in labels:
    c1 = summary["arms"]["c1"][key]
    c4 = summary["arms"]["c4"][key]
    print(f"{label:22} {c1['mean']:>14.{digits}f} {c4['mean']:>14.{digits}f}")
print()
for arm in ARMS:
    for key, _, _ in [(k, 0, 0) for _, k, _ in labels]:
        pass
print("repeat spread (max-min as % of mean):")
for label, key, _ in labels:
    c1 = summary["arms"]["c1"][key]["spread_pct"]
    c4 = summary["arms"]["c4"][key]["spread_pct"]
    print(f"  {label:22} c1 {c1:5.2f}%   c4 {c4:5.2f}%")

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def mean(values):
    return statistics.fmean(values)


result_dir = Path(__file__).parent
tag = sys.argv[1] if len(sys.argv) > 1 else "mixed-c4"
prompt_file = sys.argv[2] if len(sys.argv) > 2 else "prompts.jsonl"
concurrency = 4
prompts = [
    json.loads(line)
    for line in (result_dir / prompt_file).read_text().splitlines()
]
paths = sorted(result_dir.glob(f"{tag}-r*.json"))
if not paths:
    raise SystemExit(f"no {tag}-r*.json results in {result_dir}")
results = [json.loads(path.read_text()) for path in paths]

repetitions = []
domains = defaultdict(list)
for repeat, result in enumerate(results, 1):
    first_tokens = [
        start + ttft for start, ttft in zip(result["start_times"], result["ttfts"])
    ]
    finishes = [
        first + sum(itls) for first, itls in zip(first_tokens, result["itls"])
    ]
    decode_tokens = sum(length - 1 for length in result["output_lens"])
    decode_interval = max(finishes) - min(first_tokens)
    repetitions.append(
        {
            "repeat": repeat,
            "output_throughput_tps": result["output_throughput"],
            "decode_interval_throughput_tps": decode_tokens / decode_interval,
            "mean_tpot_ms": result["mean_tpot_ms"],
            "mean_ttft_ms": result["mean_ttft_ms"],
            "mtp_acceptance_percent": result["spec_decode_acceptance_rate"],
            "mtp_tokens_per_step": result["spec_decode_acceptance_length"],
            "duration_s": result["duration"],
        }
    )
    for index, (prompt, output_len, ttft, itls) in enumerate(
        zip(prompts, result["output_lens"], result["ttfts"], result["itls"])
    ):
        tpot = sum(itls) / (output_len - 1)
        domains[prompt["domain"]].append(
            {
                "submission_index": index,
                "drains": index >= len(prompts) - concurrency + 1,
                "tpot_ms": tpot * 1000,
                "ttft_ms": ttft * 1000,
            }
        )

summary = {
    "configuration": {
        "concurrency": concurrency,
        "mtp_depth": 3,
        "prompts": len(prompts),
        "prompt_file": prompt_file,
        "submission_order": [prompt["domain"] for prompt in prompts],
        "output_tokens_per_prompt": 256,
        "repetitions": len(results),
    },
    "mean": {
        key: mean([repeat[key] for repeat in repetitions])
        for key in repetitions[0]
        if key != "repeat"
    },
    "repetitions": repetitions,
    "domains": {
        domain: {
            "requests": len(rows),
            "mean_tpot_ms": mean([row["tpot_ms"] for row in rows]),
            # Derived from the mean TPOT so the rate and the latency columns
            # cannot disagree; averaging 1/tpot biases the rate upward.
            "decode_tps_from_mean_tpot": 1000 / mean([row["tpot_ms"] for row in rows]),
            "mean_ttft_ms": mean([row["ttft_ms"] for row in rows]),
            "submission_indices": sorted({row["submission_index"] for row in rows}),
            "draining_requests": sum(1 for row in rows if row["drains"]),
        }
        for domain, rows in domains.items()
    },
    # Requests submitted in the final `concurrency - 1` slots finish with fewer
    # co-running requests and post better TPOT for that reason alone. If the
    # submission order groups domains, this is indistinguishable from a domain
    # effect; interleave the order before reading the per-domain table.
    "steady_vs_draining": {
        state: {
            "requests": sum(
                1
                for rows in domains.values()
                for row in rows
                if row["drains"] is drains
            ),
            "mean_tpot_ms": mean(
                [
                    row["tpot_ms"]
                    for rows in domains.values()
                    for row in rows
                    if row["drains"] is drains
                ]
            ),
        }
        for state, drains in (("steady", False), ("draining", True))
    },
}
(result_dir / f"summary-{tag}.json").write_text(json.dumps(summary, indent=2) + "\n")
json.dump(summary, sys.stdout, indent=2)
print()

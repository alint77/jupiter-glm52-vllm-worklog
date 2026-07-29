import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def mean(values):
    return statistics.fmean(values)


result_dir = Path(__file__).parent
prompts = [
    json.loads(line)
    for line in (result_dir / "prompts.jsonl").read_text().splitlines()
]
paths = [result_dir / f"mixed-c4-r{repeat}.json" for repeat in (1, 2)]
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
    for prompt, output_len, ttft, itls in zip(
        prompts,
        result["output_lens"],
        result["ttfts"],
        result["itls"],
    ):
        tpot = sum(itls) / (output_len - 1)
        domains[prompt["domain"]].append(
            {
                "decode_tps": 1 / tpot,
                "tpot_ms": tpot * 1000,
                "ttft_ms": ttft * 1000,
            }
        )

summary = {
    "configuration": {
        "concurrency": 4,
        "mtp_depth": 3,
        "prompts": len(prompts),
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
            "mean_per_request_decode_tps": mean(
                [row["decode_tps"] for row in rows]
            ),
            "mean_tpot_ms": mean([row["tpot_ms"] for row in rows]),
            "mean_ttft_ms": mean([row["ttft_ms"] for row in rows]),
        }
        for domain, rows in domains.items()
    },
}
(result_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
json.dump(summary, sys.stdout, indent=2)
print()

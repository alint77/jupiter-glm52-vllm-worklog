#!/usr/bin/env python3
"""Compare two logprob captures: KL divergence, top-1 agreement, rank overlap.

Reports several statistics rather than one, because each fails differently:

  - **KL divergence** over the renormalised top-k. Top-k truncation makes this
    approximate, so it is a comparison against a control, not an absolute.
  - **Top-1 agreement.** The token the model would actually emit greedily. This
    is what changes user-visible output.
  - **Top-k set overlap.** Catches a distribution that has shifted broadly
    even where the argmax happens to survive.
  - **Max logprob delta** on shared tokens, which localises the worst position.

Interpretation needs a control. Two runs of the *same* configuration differ by
some amount from float non-determinism alone; the tiered-vs-reference number
is only meaningful relative to that floor.
"""

import argparse
import json
import math
import statistics
import sys
from pathlib import Path


def softmax_from_logprobs(lps: list[float]) -> list[float]:
    m = max(lps)
    e = [math.exp(x - m) for x in lps]
    s = sum(e)
    return [x / s for x in e]


def kl(p_ids, p_lps, q_ids, q_lps) -> float | None:
    """KL(P||Q) over the union of top-k, with a floor for tokens absent from Q."""
    q = dict(zip(q_ids, q_lps))
    if not q:
        return None
    floor = min(q_lps) - 2.0  # unseen tokens sit below Q's observed tail
    p_probs = softmax_from_logprobs(p_lps)
    total = 0.0
    for tid, pp in zip(p_ids, p_probs):
        qlp = q.get(tid, floor)
        plp = math.log(pp) if pp > 0 else -60.0
        total += pp * (plp - qlp)
    return total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="capture A (e.g. tiered)")
    ap.add_argument("--b", required=True, help="capture B (e.g. reference)")
    ap.add_argument("--out")
    args = ap.parse_args()

    A = json.loads(Path(args.a).read_text())
    B = json.loads(Path(args.b).read_text())

    kls, top1, overlap, deltas = [], 0, [], []
    n = 0
    for ca, cb in zip(A["captures"], B["captures"]):
        for pa, pb in zip(ca["positions"], cb["positions"]):
            n += 1
            d = kl(pa["ids"], pa["lps"], pb["ids"], pb["lps"])
            if d is not None:
                kls.append(max(d, 0.0))
            if pa["ids"] and pb["ids"]:
                top1 += pa["ids"][0] == pb["ids"][0]
                overlap.append(len(set(pa["ids"]) & set(pb["ids"])) / len(pa["ids"]))
            qb = dict(zip(pb["ids"], pb["lps"]))
            for tid, lp in zip(pa["ids"], pa["lps"]):
                if tid in qb:
                    deltas.append(abs(lp - qb[tid]))

    res = {
        "a": A["label"],
        "b": B["label"],
        "positions_compared": n,
        "kl_mean": round(statistics.mean(kls), 6) if kls else None,
        "kl_median": round(statistics.median(kls), 6) if kls else None,
        "kl_p99": round(sorted(kls)[int(0.99 * len(kls))], 6) if kls else None,
        "kl_max": round(max(kls), 6) if kls else None,
        "top1_agreement": round(top1 / n, 4) if n else None,
        "topk_overlap_mean": round(statistics.mean(overlap), 4) if overlap else None,
        "logprob_delta_mean": round(statistics.mean(deltas), 6) if deltas else None,
        "logprob_delta_max": round(max(deltas), 6) if deltas else None,
    }
    print(json.dumps(res, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2))
    print(
        "\nInterpretation needs the same-config control run: float "
        "non-determinism alone gives a non-zero KL, and only a value well "
        "above that floor indicates the tiered path computes something different."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

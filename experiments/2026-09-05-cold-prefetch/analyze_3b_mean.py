# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A logprob gate that survives the noise floor, from the phase-3b records.

Comparing single reps was inconclusive: the distance between two runs of the
same configuration matched the distance between configurations. Averaging the
top-20 logprob vector over an arm's three reps cuts that noise by root-3, so
the question becomes whether the arm-to-arm distance between *means* exceeds
what the reps' own scatter predicts.

Also reports the decisive-argmax check: on a prompt whose top-1 is several nats
clear of top-2, a staging defect would have to be large to flip it, so
agreement there is meaningful where agreement on a near-tie is not.

Caveat established by running it: the scatter threshold has no authority. The
same-config control (baseline1 vs baseline2) exceeds it on two of three
prompts, so a cross-config pair exceeding it means nothing on its own. Only the
comparison against that control, and the decisive-argmax check, carry weight.
"""

import json
from itertools import combinations
from pathlib import Path
from statistics import fmean, pstdev

HERE = Path(__file__).parent
ARMS = ("baseline1", "baseline2", "staged_noverify", "staged_verify")


def load() -> dict:
    out = {}
    for arm in ARMS:
        path = HERE / f"p3b-{arm}.json"
        if path.exists():
            out[arm] = json.loads(path.read_text())
    return out


def mean_vector(reps: list[dict]) -> dict[str, float]:
    """Mean logprob per token, over tokens every rep ranked."""
    shared = set(reps[0]["top_logprobs"])
    for r in reps[1:]:
        shared &= set(r["top_logprobs"])
    return {t: fmean(r["top_logprobs"][t] for r in reps) for t in shared}


def scatter(reps: list[dict]) -> float:
    """Largest per-token standard deviation across reps."""
    shared = set(reps[0]["top_logprobs"])
    for r in reps[1:]:
        shared &= set(r["top_logprobs"])
    if not shared:
        return float("nan")
    return max(pstdev([r["top_logprobs"][t] for r in reps]) for t in shared)


def main() -> int:
    data = load()
    if len(data) < 2:
        print("need at least two arms")
        return 1
    prompts = sorted({row["prompt"] for rows in data.values() for row in rows})

    for prompt in prompts:
        print(f"\n=== {prompt} ===")
        means, scat, control = {}, {}, None
        for arm, rows in data.items():
            reps = [r for r in rows if r["prompt"] == prompt]
            if len(reps) < 2:
                continue
            means[arm] = mean_vector(reps)
            scat[arm] = scatter(reps)
            firsts = {r["first_token"] for r in reps}
            margin = "n/a"
            ranked = sorted(means[arm].values(), reverse=True)
            if len(ranked) > 1:
                margin = f"{ranked[0] - ranked[1]:.3f}"
            print(
                f"  {arm:16s} rep scatter(max sd)={scat[arm]:.3f}  "
                f"top1-top2={margin}  first tokens={sorted(firsts)}"
            )

        print("  -- distance between arm means --")
        for left, right in combinations(means, 2):
            shared = set(means[left]) & set(means[right])
            if not shared:
                continue
            dist = max(abs(means[left][t] - means[right][t]) for t in shared)
            # What the reps' own scatter predicts for a mean-of-3 difference.
            predicted = 1.4 * max(scat[left], scat[right]) / (3**0.5)
            kind = (
                "same-config" if left[:8] == right[:8] == "baseline" else "cross-config"
            )
            if kind == "same-config":
                control = dist
            print(
                f"    {left:16s} vs {right:16s} d={dist:.3f} "
                f"predicted<={predicted:.3f}  [{kind}]"
            )
        # The same-config pair is the control. Any cross-config distance has to
        # be read against it, not against the invented scatter threshold --
        # which the control itself fails on two of the three prompts.
        if control is not None:
            cross = [
                max(
                    abs(means[a][t] - means[b][t])
                    for t in set(means[a]) & set(means[b])
                )
                for a, b in combinations(means, 2)
                if not (a[:8] == b[:8] == "baseline")
                and set(means[a]) & set(means[b])
            ]
            print(
                f"  CONTROL same-config d={control:.3f} vs cross-config "
                f"{min(cross):.3f}-{max(cross):.3f} -> "
                + ("separable" if min(cross) > control else "NOT separable")
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

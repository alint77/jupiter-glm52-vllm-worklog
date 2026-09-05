# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare prefill output distributions across the phase-3b arms.

The phase-3 gate asked whether two completions were byte-equal, which a greedy
argmax can fail on a near-tie without anything being wrong. This asks the
question the gate meant to ask: how far apart are the distributions the prefill
actually produced, and is any argmax flip a tie or a shift.
"""

import json
import sys
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).parent
ARMS = ("baseline1", "baseline2", "staged_noverify", "staged_verify")


def load() -> dict:
    records = {}
    for arm in ARMS:
        path = HERE / f"p3b-{arm}.json"
        if not path.exists():
            print(f"missing: {path.name}")
            continue
        for row in json.loads(path.read_text()):
            records[(row["arm"], row["prompt"], row["rep"])] = row
    return records


def distance(a: dict, b: dict) -> tuple[float, int]:
    """Max |delta| over tokens both arms ranked, and the count of shared ones."""
    shared = set(a) & set(b)
    if not shared:
        return float("nan"), 0
    return max(abs(a[t] - b[t]) for t in shared), len(shared)


def tie_margin(top: dict) -> float:
    """Gap between the best and second-best token: how close the argmax was."""
    ranked = sorted(top.values(), reverse=True)
    return ranked[0] - ranked[1] if len(ranked) > 1 else float("inf")


def main() -> int:
    records = load()
    if not records:
        print("no results yet")
        return 1
    prompts = sorted({key[1] for key in records})

    print("== reproducibility within an arm (same server, 3 reps) ==")
    for arm in ARMS:
        for prompt in prompts:
            reps = [records[k] for k in records if k[0] == arm and k[1] == prompt]
            if len(reps) < 2:
                continue
            texts = {r["text"] for r in reps}
            worst = max(
                distance(x["top_logprobs"], y["top_logprobs"])[0]
                for x, y in combinations(reps, 2)
            )
            flag = "stable" if len(texts) == 1 else f"{len(texts)} DISTINCT TEXTS"
            print(
                f"  {arm:16s} {prompt:6s} tokens={reps[0]['prompt_tokens']:>7} "
                f"max|dlogp|={worst:.3e}  {flag}"
            )

    print("\n== across arms (rep 0) ==")
    for prompt in prompts:
        print(f"  -- {prompt} --")
        for left, right in combinations(ARMS, 2):
            a = records.get((left, prompt, 0))
            b = records.get((right, prompt, 0))
            if a is None or b is None:
                continue
            delta, shared = distance(a["top_logprobs"], b["top_logprobs"])
            same_first = a["first_token"] == b["first_token"]
            same_text = a["text"] == b["text"]
            margin = min(tie_margin(a["top_logprobs"]), tie_margin(b["top_logprobs"]))
            verdict = "match" if same_text else ("TIE-FLIP" if margin < delta else "SHIFT")
            print(
                f"    {left:16s} vs {right:16s} max|dlogp|={delta:.3e} "
                f"shared={shared:2d} margin={margin:.3e} "
                f"first={'=' if same_first else 'X'} text={'=' if same_text else 'X'} "
                f"-> {verdict}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())

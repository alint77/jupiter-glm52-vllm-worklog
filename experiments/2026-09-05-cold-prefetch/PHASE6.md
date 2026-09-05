# Phase 6 — the accuracy gate on the single-stream path

Job `1671632`. Code `27099cd923`.

## Why the two deferred tuning items were dropped

Neither survived measurement, so neither got an allocation.

**Launch-policy retune.** `_apply_tier_launch_policy` gives the cold tier 1
block/SM against the hot tier's 2, an asymmetry chosen when cold was
Grace-bound. Now that the staged tier reads HBM, 2 looked worth trying. It is
not reachable: `marlin_moe.py:142` applies the policy only when
`launch_policy.max_tokens >= M`, and `max_tokens` is the 2048 overlap cap, so
at a full 8192-token chunk Marlin's own heuristic runs instead. The setting
cannot affect the regime staging exists for.

**Wrap-around staging to 75/75.** Worth 4.82 ms/chunk from the roofline --
0.29% of prefill wall, against +/-0.77% wall-clock noise, so only a trace could
confirm it. It also costs an invariant: today the slot is only ever read inside
the forward pass that filled it, and wrap-around would leave layer 0's bytes
live in the slot *across requests* with nothing to invalidate them. An
in-place reload of the Grace buffers would then serve stale weights silently
under `VERIFY=0`, and the `register()` guard would not catch it because
in-place reload never re-registers. It would also route around
`chunk_complete()`, stopping the per-chunk log. Not worth 0.29%.

## The gap this closed

Phase 5's GSM8K prompts are 643 tokens: one chunk, above `MIN_TOKENS=256` and
below the 2048 overlap cap, so that eval gated the **fork branch** -- hot on
the compute stream, staged cold on the tier stream. The **single-stream
M=8192 path**, which is what the prefetch was built for, had byte verification
and the roofline but no accuracy eval, and phase 3b's one unexplained flag sat
on it.

Padding each prompt with 36000 characters of coding-transcript text, fenced off
with a rule, gives 8973 tokens: chunk 1 at 8192 -- staged, and single-stream
because 8192 exceeds the overlap cap -- on every one of 1000 questions.

**Correction to the design note.** The plan said chunk 2 would land at ~1349
tokens and so be staged and forked too. It does not: the real prompt is 8973
tokens, not the 9541 the 4-chars-per-token estimate predicted, so chunk 2 is
781 tokens, below `MIN_TOKENS=1024`, and ran unstaged. That is exactly why the
chunk counter read 1004 for ~1002 requests rather than double. No coverage was
lost -- phase 5 already gated the fork branch -- but the run covered one branch,
not two.

## Result

| | baseline | staged |
| --- | --- | --- |
| accuracy | 76.20% (762/1000) | **76.30% (763/1000)** |
| invalid | 0.000 | 0.001 |
| eval wall | 3230 s | **2979 s** |

```
concordant 767   baseline-only 116   staged-only 117
McNemar exact two-sided p = 1.0000
accuracy delta +0.0010
```

The 15-point drop from the unpadded 91.13% is long-context distraction and
applies to both arms identically. It is also a useful sanity check: had padding
left accuracy untouched, the long prefill would not have been getting attended
to.

**Throughput:** the staged arm finished the same 1000 questions 7.8% faster
(2979 s against 3230 s). Each question is an 8192-token prefill plus 256 tokens
of decode, so this is the prefetch's benefit diluted by decode -- a fairer
end-to-end figure for a mixed workload than the prefill-only -16.1%.

## The bound is *weaker* than phase 5's, not stronger

233 discordant pairs, so the smallest asymmetry this would have flagged is a
101/132 split -- 31 questions, **3.10 accuracy points**. Phase 5 resolved 1.21.

The padded task is far noisier (23% discordance against 3.8% unpadded) because
it sits near the model's competence edge, where many questions are coin-flips.
More discordant pairs mean a *coarser* bound, not a finer one; an earlier note
in this campaign had that backwards.

So the two evals bound different paths at different resolutions:

| path | eval | bound |
| --- | --- | --- |
| fork branch (M <= 2048) | phase 5, 1319 unpadded | 1.21 points |
| single-stream (M = 8192) | phase 6, 1000 padded | 3.10 points |

Tightening the single-stream bound needs a long-context task the model is not
near the edge of -- the noise, not the sample size, is what limits it.

## Phase 3b's `chunk` flag

Retired as far as it can be. The flag was a cross-config logprob distance
exceeding the same-config control on a 3725-token prompt -- single-stream,
staged, the same path this eval covers. A defect there large enough to matter
would have to survive 1000 paired questions at p = 1.000. It remains formally
unexplained, and it remains most plausibly the 0.042-nat argmax tie it always
looked like.

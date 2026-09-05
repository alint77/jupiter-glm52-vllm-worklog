# Phase 5 — does the prefetch change what the model answers?

Job `1669612`. Code: `f6a7e8727f` (phases 3 and 4 merged).

## Why not a token diff

Phase 3b measured this configuration's run-to-run spread at 0.6-2.3 nats on the
top-20 logprobs, **in the baseline**. Repeated identical requests to one server
produce different greedy completions. Token comparison therefore carries no
information here in either direction, which is why phase 3's "gate failed" was
meaningless and why phase 2 agreeing with phase 3 was luck.

Accuracy over many samples is stable where any single completion is not, and
`AGENTS.md` requires an eval for an output-affecting change regardless.

## Design

Both arms answer the **identical** 1319 questions (GSM8K's full test set,
5-shot) and per-question outcomes are recorded, so the comparison is paired.
McNemar's test then asks whether the discordant pairs are *asymmetric*.
Run-to-run noise produces symmetric discordance; a defect does not. That
distinction is what makes a within-arm control unnecessary.

`MIN_TOKENS` is 256 here, not the 1024 of phases 3/3b: GSM8K's 5-shot prompts
measure ~539 tokens, so 512 would have been marginal and the staged path might
never have engaged. Engagement was confirmed from the log rather than assumed
-- `74 from slot, 1 from Grace` across 1323 chunks.

Staging 49.4 GiB behind a ~539-token prefill is pure overhead. That is
deliberate: this run gates correctness, and short prompts are the harsher test
of it. The speed case is the roofline in `PHASE3.md`.

## Result

| | baseline | staged |
| --- | --- | --- |
| accuracy | 91.13% (1202/1319) | **91.28% (1204/1319)** |
| invalid | 0 | 0 |
| eval wall | 1644 s | 1630 s |
| residency | 2325 hot / 2475 cold | 2284 hot / 2516 cold |

```
concordant 1269   baseline-only 24   staged-only 26
McNemar exact two-sided p = 0.8877
accuracy delta +0.0015
```

Fifty discordant pairs, split 24 against 26. That is as symmetric as fifty
coin flips get, and it is what a correct implementation under a noisy runtime
looks like.

## What the result does and does not bound

With 50 discordant pairs, the smallest asymmetry this test would have flagged
at p < 0.05 is a 17/33 split -- a systematic loss of 16 questions, or **1.21
accuracy points**. So:

* **Bounded:** the prefetch does not cost more than ~1.2 accuracy points.
* **Not bounded:** anything smaller. A defect costing half a point would have
  passed. Tightening this needs more questions, not a different test; the
  discordant count is what sets the resolution.

The earlier 300-question design would have resolved only ~5 points, which is
why it was abandoned once the real question rate (0.81 q/s, not the 0.06 q/s
extrapolated from the 96K prompt) showed the full set was affordable.

## Correctness evidence, assembled

No single line of this is proof; together they are a reasonable case.

| evidence | strength |
| --- | --- |
| 1924 byte compares of the staged slot against Grace, 0 mismatched | strong, but only that the *bytes* are right |
| GSM8K paired McNemar, p = 0.888 over 1319 questions | rules out degradation above 1.2 points |
| `mid` prompt, decisive 4.7-nat argmax, 12/12 arms and reps agree | weak but clean |
| staged execution confirmed engaged (74 of 75 layers, 1323 chunks) | rules out an eval that measured nothing |

The one loose end from phase 3b stays loose: on the `chunk` prompt the
cross-config logprob distance exceeded the same-config control. That prompt's
argmax is a numerical tie (0.042 nats), and the eval finds no accuracy cost, so
it is most likely tie instability. It is recorded rather than explained away.

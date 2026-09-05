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

## Which code path the eval actually exercised

Worth being precise, because it is the harder one. GSM8K's prompts are ~539
tokens, and `tiered_overlap_max_tokens` is 2048, so `256 <= 539 <= 2048` puts
every one of these prefills **inside the overlap window**: hot Marlin on the
compute stream, the staged cold kernel on `tier_stream`, both cross-stream
hazards live at once. That is the configuration where a missing `wait_stream`
would show, and it passed at p = 0.888 over 1319 questions.

The converse is the honest caveat. The single-stream chunked path at M=8192 --
the regime the prefetch was actually built for -- has the 1924/0 byte
verification and the roofline behind it, but **no accuracy eval**. The one
unexplained flag from phase 3b, the `chunk` prompt at 3725 tokens, sits on that
path. It is most likely tie instability (a 0.042-nat margin, and the baselines
varied across all three reps there too), but the coincidence belongs in the
record rather than in a footnote. A long-context eval would close it; MRCR
needs its dataset pre-materialised first, since it streams from HuggingFace and
the compute nodes have no network.

## Throughput, with the placement cost included

Three runs of the 96K prompt per arm, on the same allocation:

| arm | runs | mean |
| --- | --- | --- |
| baseline | 23.31, 23.32, 23.44 s | 23.36 s |
| staged | 19.57, 19.60, 19.62 s | **19.60 s** |

**-16.1%**, with each arm tight to +/-0.15 s. This is the honest end-to-end
figure: it is measured on the merged code, so the 41 hot experts phase 4 gives
up to budget the slot are already paid for inside it.

GSM8K wall time was 1630 s staged against 1644 s baseline -- unchanged, as
expected. Staging 49.4 GiB behind a 539-token prefill is overhead, but it is
small next to 256 tokens of decode, so short-prompt serving is not harmed
either.

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

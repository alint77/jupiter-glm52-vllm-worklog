# Phase 3 — execute the cold tier from the staged slot

Commit `047195e43c`. Job `1669068` (Booster, full node, both arms in one
allocation).

## What changed

Phase 2 staged every layer's cold experts into an HBM slot and verified the
bytes, but nothing read the slot — the cold tier still ran from Grace. Phase 3
points the cold tier at the slot.

Two things had to move, not one. `apply_tiered_moe` builds a `tiers` list
carrying only `w13_weight_packed` / `w2_weight_packed`, so swapping that list
would have redirected the weights alone. The **scales are baked into the
kernel's `quant_config` at setup time** by `make_wna16_moe_quant_config`, and
Marlin reads weights and scales per tile inside the same CTA — leaving the
scales on Grace would have gated every tile exactly as the weights did. Scales
are 2.36 MB of the 21.23 MB per expert, but the gating is per-tile, not
proportional, so the expected gain from staging weights alone is near zero.

Because the scales live in the kernel, the staged tier needs its own kernel.
It is built once per layer at first use (`_staged_cold_tier`), cached on the
method, and carries the same launch policy as the tier it stands in for, so
the two kernels differ in their operands and in nothing else.

## Fallback

`staged_views()` returns `None` when the slot does not hold the layer being
run. That happens at the first MoE layer of every chunk, which has nothing
staged behind it. The fallback is the existing Grace tier, decided per layer.
The per-chunk log now reports the split (`N from slot, M from Grace`); in
steady state `M` should be 1, and anything higher is a staging gap.

## Ordering under the overlap branch

`VLLM_TIERED_MOE_COLD_PREFETCH_MIN_TOKENS=1024` and
`tiered_overlap_max_tokens=2048` overlap for prompts of 1024–2048 tokens: such
a batch runs the prefetch *and* forks hot/cold onto separate streams. The
phase-2 commit message justified the single slot by "the copy cannot begin
until this layer's cold Marlin has finished", which is only self-evident when
prefill is single-stream.

Checked rather than assumed: `modular_kernel.py:1665` runs
`main_stream.wait_stream(tier_stream)` inside `apply_tiered`, before it
returns. The join is not deferred to the consumer. So `stage()`'s
`self._stream.wait_stream(compute_stream)` — where `compute_stream` is the
main stream — transitively orders the copy after the cold Marlin on the tier
stream. Symmetrically, `await_staged` blocks the main stream on the copy, and
`tier_stream.wait_stream(main_stream)` inside `apply_tiered` carries that to
the cold read. The window is safe. A ~1500-token prompt goes into the phase-5
gate on both arms to keep this honest.

## Gate

Byte-identical completions between arms on the same prompt at temperature 0,
run in one allocation:

- baseline: `MIN_TOKENS=0` — no staging at all
- prefetch: `MIN_TOKENS=1024 VERIFY=1` — staged, verified, executed from slot

Both a short prompt and the ~96K prompt (≈12 prefill chunks). `VERIFY=1` stays
on: it costs ~26 ms/chunk and turns a mismatch into a log line instead of
wrong tokens. Prefill wall time is recorded as the cheap read on whether cold
Marlin actually got faster — the formal number (cold Marlin 583 ms/chunk →
below 383) is the phase-5 kill criterion.

## Results — job `1669068`

The mechanism works. Every number that measures *the staging itself* is clean:

| | |
| --- | --- |
| slot | 830 MiB on ranks 0/2/3, 810 MiB on rank 1, 75 layers each |
| staged | 48.5 GiB per chunk |
| byte verifies | **1924, 0 mismatched** |
| tier selection | **74 from slot, 1 from Grace** per chunk, every chunk |
| prefill wall | 23.53 s baseline -> **21.69 s** prefetch (-7.8%) |

`74 from slot / 1 from Grace` is exactly the predicted steady state: the one
fallback is the first MoE layer of each chunk, which has nothing staged behind
it. So the staged kernel is genuinely being selected and genuinely being read.

## The gate failed, and the gate was wrong

The short completion matched. The long one diverged **from the first token** --
too gross to be float noise, which sent me looking for a view-offset bug.

It is not a bug in the change. Phase 2 ran the same prompt with the cold tier
executing **from Grace** -- staging happened but nothing read the slot, so the
arithmetic was identical to the baseline's. Its completion:

```
 Show the complete file.\n\nWrite a pytest test for verify_observed_hbm_reserve ...
```

which is **character-for-character the phase-3 prefetch arm's output**. The
phase-3 *baseline* is the one that stands alone:

```
</think>```python\n# SPDX-License-Identifier: Apache-2.0 ...
```

What this does establish, and all it establishes: **the execution swap does
not move the output.** Phase 2 (staging on, cold tier read from Grace) and
phase 3 (staging on, cold tier read from the slot) agree character for
character, on different server processes. That is the thing phase 3 changed,
and it is clean.

What it does *not* establish is why the staging-off baseline differs. Two
readings fit the same three runs:

* the runs are not reproducible, and A/B are two attractors on a near-tie; or
* `MIN_TOKENS > 0` deterministically shifts prefill output whether or not the
  slot is ever read -- which would be a real finding, since phase 2 read
  nothing from it.

Note the evidence leans against the first: two *independent* server processes
produced character-identical text, which is evidence **for** reproducibility
within a configuration. Calling this non-determinism would be the same mistake
in the other direction.

The error was in the gate, not the conclusion: bitwise identity presumes
run-to-run reproducibility across a configuration change, and I asserted it
rather than measuring it. Phase 3b measures it.

## Phase 3b — a gate that measures what it claims to

1. **Determinism control.** One server, the long prompt three times; then a
   restart of the same arm and once more. This is the missing measurement:
   whether the baseline reproduces itself within a process and across restarts.
2. **A numerical gate instead of a token gate.** Compare the top-20 logprobs
   after the 96K prefill (`max_tokens=1, logprobs=20`) between arms. That reads
   the prefill's output directly rather than through a greedy argmax that can
   flip on a tie, and it produces a distance, not a boolean.
3. The 1924/0 byte verify already establishes that what reaches the kernel is
   the right bytes. What is still unmeasured is whether the staged *kernel*
   computes the same function -- which is what the logprob distance settles.
4. A `VERIFY=0` staging arm, to separate the per-layer host sync from the slot
   allocation itself, and a one-chunk and a ~1500-token prompt beside the 96K
   one. Twelve chunks of a 96K prefill is where ties are likeliest; a
   single-chunk prompt gives the logprob gate teeth.

Job `1669268`: four server loads in one allocation -- two identical baselines
(the second is the across-restart control), then staging without and with
verify.

## Ruled out: KV cache geometry

The memory-profiling pass runs 8192 dummy tokens, which is above
`MIN_TOKENS`, so the slot is allocated lazily *during* profiling and the
staging arm measures less free HBM:

| arm | available KV cache | GPU KV cache size |
| --- | --- | --- |
| baseline | 23.02 GiB | 400,064 tokens |
| prefetch | 21.94 GiB | 400,064 tokens |

The 1.08 GiB gap is the slot (830 MiB) plus allocator slack. It is a real cost
and worth knowing -- at a larger `max_model_len` it would cut KV capacity --
but here `max_model_len=400000` caps the cache first, so **both arms run the
identical KV geometry**. It cannot be what moved the output, and it means the
phase-3b arms are not confounded by cache size either.

Phase 4 makes this cost explicit rather than incidental: the planner reserves
the slot up front, so it comes out of expert residency where it can be seen,
not out of whatever the profiling pass happened to measure.

## The speedup is about half of what the mechanism should give

580 ms/chunk of cold Marlin at the hot tier's per-expert rate would be ~210 ms;
over 12 chunks, less verify and copy contention, that is ~3.9 s. The measured
drop was 1.84 s, which puts staged cold Marlin near ~400 ms/chunk -- at or
above the 383 ms phase-5 kill line. The `74 from slot` counter proves the
staged kernel is selected; it does not prove it reads at HBM rate. That is the
first phase-5 question: one profiled chunk through `prefill_roofline.py` shows
whether staged cold Marlin now sits at the hot tier's ~250 GB/s or somewhere
between.

## Phase 3b result: the serving configuration is non-deterministic

**Correction.** An earlier revision of this file claimed phase 3b had found a
race that `VERIFY` was masking. That was wrong, and it was wrong because it
generalised from the monitor's filtered output -- which printed only `rep0` and
`rep2` first tokens -- instead of the recorded results. The baselines agreed on
those two lines and disagreed everywhere else.

The full records show **every arm varies, baseline included**:

| arm | prompt | rep0 | rep1 | rep2 |
| --- | --- | --- | --- | --- |
| baseline1 | long | `</think>` | ` Show` | `</think>` |
| baseline1 | chunk | 3 distinct texts | | |
| baseline2 | chunk | 3 distinct texts | | |
| staged_noverify | chunk | 3 distinct texts | | |

Within-arm logprob spread, max over the shared top-20:

| arm | chunk | long | mid |
| --- | --- | --- | --- |
| baseline1 | 1.557 | 0.585 | 0.748 |
| baseline2 | 2.268 | 0.690 | 0.694 |
| staged_noverify | **1.384** | 0.914 | 0.680 |

The staged arm's variation sits inside the baselines' range, and on the chunk
prompt it is the *lowest* of the three. There is a cleaner demonstration still
inside a single rep: each rep issues two separate prefills of the same prompt
(the `max_tokens=1` logprob probe and the 32-token text request), and in the
baseline those two disagree on the first token.

So this serving configuration -- GLM-5.3 W4A16, MTP3, chunked prefill, TP4 --
does not reproduce itself run to run, at a noise floor of roughly 0.6 to 2.3
nats on the top-20 logprobs. That predates the prefetch entirely.

### What this does and does not settle

* **Settled:** token-level comparison cannot gate this change, in either
  direction. Phase 3's "gate failed" carried no information, and phase 2
  agreeing with phase 3 was luck rather than evidence.
* **Settled:** staging does not make output *less* stable than the baseline.
* **Not settled:** whether staging is correct. A noise floor this high would
  hide a real defect, so "within noise" is not proof. The 1924/0 byte compares
  remain genuine evidence that the bytes reaching the kernel are right, but they
  say nothing about the staged kernel computing the right function.

### Next: an eval, which is the right instrument anyway

`AGENTS.md` requires a model eval for output-affecting changes, and an eval is
exactly the tool for a system with a per-token noise floor: accuracy over many
samples is stable where any single completion is not. Phase 5 runs one on both
arms rather than diffing text.

## The measured win, which is unaffected

Cold Marlin 580.4 -> 223.9 ms/chunk at 333/351 GB/s against the hot tier's
335/353 -- per-expert parity, the 4.6x access-pattern gap closed, and the
383 ms/chunk kill line cleared. Prefill wall 1902.5 -> 1634.3 ms/chunk. End to
end on the 96K prompt, 23.4 s -> 19.8 s with `VERIFY=0`.

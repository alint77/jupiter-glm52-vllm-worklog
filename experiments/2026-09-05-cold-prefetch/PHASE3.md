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

## Results

Pending job `1669068`.

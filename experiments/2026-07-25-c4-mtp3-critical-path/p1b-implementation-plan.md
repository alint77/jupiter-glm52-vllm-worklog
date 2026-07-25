# P1b implementation plan — make the hot and cold MoE tiers overlap

Target: the routed MoE layer span, measured at **25.7 ms/step** against an
isolated-cost ideal of **13.2 ms**. Upper bound ~12.5 ms/step, ~20% of the
62.43 ms step. This is the largest remaining lever in the c4 MTP3 configuration.

## 0. Correction that reopens this work

[2026-07-25-marlin-decode-tuning](../2026-07-25-marlin-decode-tuning/README.md)
concluded that splitting the SM budget between tiers "does nothing" (0.0–0.4%
across a 4x4 sweep). **That conclusion is void.** In
`csrc/libtorch_stable/moe/marlin_moe_wna16/ops.cu`:

```cpp
if (thread_k != -1 && thread_n != -1) {
    if (blocks_per_sm == -1) blocks_per_sm = 1;
    exec_cfg = exec_config_t{blocks_per_sm, thread_tfg};   // honoured
} else {
    exec_cfg = determine_exec_config(...);                 // blocks_per_sm DISCARDED
}
```

`blocks_per_sm` is only honoured when **both** `thread_k` and `thread_n` are
also supplied. Every `(blocks_per_sm, -1, -1)` config in that sweep fell into
the auto branch and produced an identical launch, which is why the timings were
within 0.4% of each other — that was measurement noise on identical work, not a
null result. The only two configs that did set thread dimensions also changed
the thread tile, confounding them.

So the leading hypothesis for why the tiers do not overlap has never been
tested.

## 1. What is established

| Fact | Source |
| --- | --- |
| hot 115.3 us, cold 104.4 us, union 194.0 us, ideal 115.3 us (M=16, w13, production routing) | grace-bandwidth job 1040910 |
| serial would be 219.7 us — so overlap saves only 25.7 us, **12% of serial, 25% of what is available** | same |
| cold runs at 88–95% of the 421 GB/s C2C roof | same |
| hot runs at 61–62% of the 3.5 TB/s HBM roof | same |
| every Marlin launch is grid 396 = 3 blocks/SM x 132 SMs, 76,458 B smem (3 x 76,458 = 229,374 of 233,472 per SM) | c4 MTP3 trace |
| the grid is identical at 8, 12 and 16 tokens | trace, all three depths |
| in-situ hot is 1.48x its isolated cost | marlin-decode-tuning |

The two tiers are bound by *different* resources — HBM for hot, C2C for cold —
so on paper they should overlap almost perfectly. They do not: the combined
region conserves total work rather than hiding one behind the other.

## 2. Mechanism hypothesis

Each Marlin launch requests exactly one full occupancy wave, and occupancy is
**shared-memory limited to 3 blocks/SM**. Whichever kernel is enqueued first
fills all 396 block slots; the second kernel's blocks cannot become resident
until the first kernel's blocks retire. Marlin blocks are long-lived (they
grid-stride over expert/N tiles), so there is little interleaving.

The consequence is specific and testable: **while a cold block stalls on a C2C
read, its SM has no co-resident hot block to issue from**, so the stall is
exposed rather than hidden. That is precisely the latency-hiding that
concurrency was supposed to buy.

If this is right, partitioning the SM budget — cold at 1 block/SM, hot at 2 —
should let both be resident from the start and recover most of the 12.5 ms.

Competing hypotheses to rule out (Phase A):

- **H2 memory-system interference.** Cold's C2C misses hold MSHRs for long
  latencies and throttle hot's HBM stream regardless of residency.
- **H3 launch-order effect.** `apply_tiered` enqueues cold first
  (`with torch.cuda.stream(tier_stream): run_tier(1)`) then hot. The shorter
  kernel grabs the GPU first, which is the wrong order for tail hiding.
- **H4 stream priority.** Both streams are default priority; no preemption.
- **H5 graph-replay serialization.** The fork/join becomes graph dependencies;
  replay may not honour concurrency. Weakened by the fact that poor overlap
  reproduces in the isolated *eager* benchmark, but not eliminated.

## 3. Phase A — diagnosis (no source changes, ~1 Booster job)

Extend `benchmarks/kernels/benchmark_moe_wna16_marlin_decode.py`.

**A1. Establish the auto thread config.** `determine_exec_config` picks
`num_threads = 128`, so `thread_k * thread_n = 8192` — either `(64, 128)` or
`(128, 64)`. Sweep both against auto at the production shape and pick the one
that reproduces auto's timing to within noise. Everything downstream must pin
this config, otherwise a `blocks_per_sm` sweep confounds tile shape with
occupancy.

**A2. Real SM-partition sweep.** With the thread config pinned, sweep
`blocks_per_sm ∈ {1,2,3}` for each tier independently (9 combinations), single
tier and overlapped. Report union span against `max(hot_solo, cold_solo)`.
Success criterion: any combination bringing union within 25% of ideal.

Note `max_shared_mem = max_shared_mem / blocks_per_sm - 1024` when
`blocks_per_sm > 1`, so higher values reduce per-block shared memory and may
fail `is_valid_config`. Catch and report rather than assume support — and see
the block-size lesson in §7.

**A3. Concurrency evidence.** Profile the isolated two-stream case and measure
the actual overlap of the two kernels' `[start, end]` intervals on the GPU
timeline, plus per-kernel dilation versus solo. This distinguishes H1
(kernels barely co-resident) from H2 (co-resident but mutually throttled).
Decisive, and cheap.

**A4. Launch order and priority.** Two one-line variants: enqueue hot first;
and create the cold stream with `torch.cuda.Stream(priority=...)`. Both are
free to test and settle H3/H4.

Phase A ends with a decision: if A2/A3 show partitioning works, go to Phase B.
If the kernels are co-resident and still slow, H2 holds and the fix is
structural — go to Phase D.

## 4. Phase B — plumb a per-tier launch config (if A2 succeeds)

Each tier already gets its **own** kernel instance:
`compressed_tensors_moe_wna16_marlin.py` builds them in a
`for tier_name in ("hot", "cold")` loop via `make_wna16_moe_kernel`. So a
per-instance attribute is the minimal plumbing point — no new call-site
threading.

1. `compressed_tensors_moe_wna16_marlin.py`, in the tier loop: set
   `kernel.impl.marlin_launch_config = (blocks_per_sm, thread_k, thread_n)`
   per tier, from a config knob with the auto behaviour as default.
2. `vllm/model_executor/layers/fused_moe/experts/marlin_moe.py`: read that
   attribute in `_fused_marlin_moe` and forward it to both
   `ops.moe_wna16_marlin_gemm` calls (w13 near line 135, w2 near line 207),
   defaulting to `-1, -1, -1` so every non-tiered path is untouched.
3. Workspace: `marlin_make_workspace_new(device, 4)` already covers up to
   4 blocks/SM, so no change. Confirm the two tiers keep independent workspace
   views — the branch already allocates from one buffer.

Gate the whole thing behind an off-by-default knob so the qualified
configuration is unchanged until measured.

## 5. Phase C — preferred alternative: fix the C++ contract

Phase B works around an API defect. The cleaner change, and an upstreamable one:
in the auto branch of `marlin_mm`, if the caller supplied `blocks_per_sm != -1`,
keep `determine_exec_config`'s thread config but override its block count, then
re-run `is_valid_config` against the adjusted shared-memory budget and fall back
to the auto value if invalid.

That makes `blocks_per_sm` mean what its signature says, removes the need to pin
thread dimensions, and benefits any caller. It is a C++ change, so it requires a
full rebuild rather than `VLLM_USE_PRECOMPILED=1` — budget for that.

Prefer C over B if Phase A confirms partitioning helps; do B first only if a
fast in-flight answer is needed.

## 6. Phase D — structural fallback (if H2 holds)

If the tiers are co-resident and still mutually throttled, per-kernel
partitioning cannot help and the fix is to stop having two kernels.

**D1. Single fused launch over both tiers.** Marlin indexes weights by expert.
If one launch covered all activated experts with per-expert base pointers —
some in HBM, some in Grace — then latency hiding happens *within* a block's
warp scheduler rather than between kernels, which is where it actually works.
Requires the kernel to accept a per-expert pointer array instead of one packed
`[E, ...]` tensor. Substantial kernel work; the payoff is the full 12.5 ms and
it removes the fork/join entirely.

**D2. Interleaved chunking.** Split each tier into k chunks and alternate their
launches so blocks from both tiers are pending at all times. Crude, no kernel
changes, worth one measurement as a cheap probe of D1's premise.

## 7. Gates, risks, and process

**Correctness gates**, unchanged from the qualified configuration: the semantic
smoke completion, the exact-400K golden SHA
`d594e4d4268600a0d9e2d51355913907c638ef0c61615547598615384dcfc528` at c1, and
the 24-prompt realistic suite at c4. Any tier-split change alters only
scheduling, so output must stay byte-identical; a SHA change means a real bug.

**Baseline to beat** (post-draft-sync, job 1040935): realistic c4 warmed
185.05 tok/s, 4K c4 99.13/110.25 tok/s, 396K c4 median ITL 58.13 ms.

**Measure the layer span, not kernel durations.** Both tiers' per-kernel times
are contention-distorted; comparing them against each other is what produced
the bogus "39–40% overlap saving" in two earlier reports. Use
`analysis/s7_layers.py`, which reports the per-layer span.

**Do not trust an unexercised knob.** This plan exists because a swept
parameter was silently discarded. Every sweep must include a positive control
that proves the parameter reached the kernel — e.g. one config whose timing
*must* differ.

**Expect async CUDA failures on invalid configs.** `moe_block_size=32` raises an
`AcceleratorError` that poisons the context and cannot be caught (job 1040908).
Validate configurations in a subprocess, or enumerate known-good ones, so one
bad combination does not lose the whole sweep.

**Interaction with the all-reduce skew.** The post-MoE skew (8.2 ms/step) scales
with per-expert cost. If the routed span shrinks, the skew shrinks with it —
so do not add the two savings independently, and re-measure the skew after any
win rather than assuming it carries over.

**Headroom is an upper bound.** The 13.2 ms ideal comes from isolated w13+w2 at
19 hot / 3 cold activated experts with `cold_share=0.13`. The trace's per-layer
distribution varies, and the isolated figure excludes the MoE epilogue that sits
inside the measured layer span. Realistic capture is likely 8–10 ms, not 12.5.

## 8. Sequence

| Step | Work | Cost | Exit criterion |
| --- | --- | --- | --- |
| A1 | pin the auto thread config | shares one job | reproduces auto within noise |
| A2 | real 3x3 blocks_per_sm sweep | same job | union within 25% of ideal, or not |
| A3 | profile the two-stream case | same job | kernels co-resident: yes/no |
| A4 | launch order + stream priority | same job | either helps, or not |
| B/C | plumb per-tier config, or fix the C++ | 1–2 days + rebuild for C | SHA holds, layer span drops |
| D | fused launch or chunk interleave | only if H2 holds | — |

Phase A is one Booster job and answers whether P1b is reachable at all. Run it
before committing to B, C or D.

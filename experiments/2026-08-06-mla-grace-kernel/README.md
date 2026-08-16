# Saturating C2C from the sparse MLA kernel

Status: **Stopped at the Phase 1 gate, premise refuted.** The kernel is not
leaving Grace bandwidth unused — it is occupancy-bound by its shared-memory
footprint, and it already extracts more from C2C at its occupancy than a plain
gather does. All source changes were reverted; the patches are kept here.

## Goal

Read the MLA KV cache directly from Grace LPDDR over C2C inside the attention
kernel, and modify the kernel so it extracts most of the link's bandwidth
instead of a third of it. Success frees ~19.3 GiB of HBM per rank for expert
residency, which is the top lever the profile identified.

**Target:** raise the sparse kernel's Grace read bandwidth from **157 GB/s** to
**≥300 GB/s** (the plain-gather ceiling is 361-384). That turns the c4 attention
penalty from **+8.5 ms/step** into **≤+3 ms**, against an expert-residency gain
estimated at **~9.4 ms/step** — i.e. from a wash into a clear net win.

**Stop condition:** if Phase 1 cannot beat 220 GB/s, stop. Below that the
arithmetic does not close and the remaining work is not worth it.

## What is already established

All measured this session on Booster, jobs `1388432`, `1388502`, `1388513`,
`1388518`, `1388524`.

### The link is not the problem

| probe | Grace GB/s |
| --- | ---: |
| Triton streaming read, ≥264 CTAs | **400-412** |
| Triton scattered gather, ≥512 B granularity | **361-384** |
| **TMA descriptor load** | **419.7** |
| copy engine H2D | 417-419 |

C2C delivers ~400 GB/s to ordinary SM loads and 419.7 GB/s through TMA.
**TMA works on host-mapped UVA memory and is full speed** — that hypothesis is
dead. Scattered access reaches 94% of the streaming roof provided the
granularity is ≥512 B; the MLA cache row is 656 B, comfortably inside that.

### The kernels are the problem, asymmetrically

At MLA's ~656 B granularity:

| | HBM | Grace |
| --- | ---: | ---: |
| FlashMLA sparse achieves | 763 GB/s | **157 GB/s** |
| plain Triton gather achieves | 784 GB/s | **361 GB/s** |
| headroom | ~1.0x (at ceiling) | **2.3x** |

**On HBM the kernel is already at the achievable gather bandwidth; on Grace it
is 2.3x below it.** That asymmetry is the signature of insufficient
memory-level parallelism to cover C2C's higher latency: HBM's latency fits
inside the kernel's existing pipeline, Grace's does not.

Earlier readings of 52 GB/s (FA3 GQA) and 157 GB/s (FlashMLA sparse) were
mistaken for hardware limits. They are kernel limits. That error is why this
experiment exists at all — the idea was refuted twice on bad grounds.

### The pipeline is two-deep and cannot be deepened

`csrc/sm90/decode/sparse_fp8/config.h:34`:

```cpp
static constexpr int NUM_K_BUFS = 2;
```

The producer warpgroup runs at most 2 KV tiles ahead of the consumer. Raising
it is the obvious fix and it is **structurally blocked by shared memory**:

| NUM_K_BUFS | K bufs | + Q | + S | total | Hopper cap |
| ---: | ---: | ---: | ---: | ---: | ---: |
| **2** | 144 KiB | 72 | 8 | **224 KiB** | 227 KiB |
| 3 | 216 | 72 | 8 | 296 | ✗ |
| 4 | 288 | 72 | 8 | 368 | ✗ |

One K buffer is `TOPK_BLOCK_SIZE(64) x HEAD_DIM_K(576) x bf16` = 72 KiB. The
kernel is within 3 KiB of the cap. **So every candidate below has to buy
latency tolerance without buying shared memory.**

### How the gather actually works

`splitkv_mla.cuh:509-560`. It is a manual gather, not TMA: per thread, per
round, it computes a row address from a token index and issues
`load_128b_from_gmem<float4, EVICT_LAST, L2PrefetchHint::B128>`. It already
prefetches the *next block's token indices* into registers
(`nxt_token_indexs`), so the index dependency is hidden — but the *data* loads
are issued only once the current tile's buffer is free.

## Candidate interventions, cheapest first

### A. L2 prefetch the next block's rows

The kernel already knows the next block's token indices one iteration ahead.
Issue `prefetch.global.L2` for those row addresses while the current tile
computes. **L2 becomes the extra pipeline stage at zero shared-memory cost** —
which is exactly the resource we do not have. GH200 has 50 MB of L2 and the
per-block working set is 64 rows x 656 B = 41 KiB, so the prefetch distance
could go well beyond one block.

Precedent exists in the file: the scales load already passes
`L2PrefetchHint::B128`.

This is the first thing to try and the most likely to work.

### B. Widen the per-thread load batch

`NUM_TOKENS_PER_THREAD` rounds are `CUTE_UNROLL`ed, so several loads can be in
flight per thread already. Check the generated SASS for how many loads are
actually issued before the first dependent use, and restructure the loop into
an explicit load-all-then-consume-all shape if the compiler is serialising.

### C. Split the K tile to buy stages

K is 576 dims = 512 nope + 64 rope. Separate, smaller buffers would allow more
pipeline stages within the same shared memory. Costs a restructure of the WGMMA
consumption, so only if A and B fall short.

### D. More CTAs

The ceiling probe needed ~1,056 CTAs for Grace to reach 361 GB/s. Measure the
kernel's actual grid at the production shape; if it is well below that, more
split-K would add parallelism across SMs rather than within a CTA.

### E. Cache hint audit

Rows are read once per step with no reuse across blocks, so `EVICT_LAST` on the
data loads may be wrong. Cheap to test alongside A.

## Phases and gates

**Phase 0 — rebuild loop.** Prove we can modify `splitkv_mla.cuh`, rebuild
`_flashmla_extension_C`, and observe the change. Gate: a deliberate no-op edit
(e.g. a changed constant with no semantic effect) is visibly rebuilt and the
existing benchmark still reproduces 157 GB/s. Nothing else matters until this
works.

**Phase 1 — intervention A.** L2 prefetch of next-block rows. Measure with
`mla_cache_full_footprint.py --query-tokens 16`. Gate: **≥220 GB/s** from
Grace, output bit-identical to HBM. Below that, stop per the stop condition.

**Phase 2 — B, D, E** as needed to reach 300 GB/s.

**Phase 3 — C** only if 1-2 fall short and the arithmetic still looks winnable.

**Phase 4 — end to end.** `--mla-cache-tier host` on a real server, same-node
A/B, acceptance-free protocol. Gate: net step-time improvement at c4, and the
exact-400K golden SHA. **Note TTFT: the 2026-07-17 retry measured +122% at
400K.** Decode winning is necessary, not sufficient — prefill has to be
measured too, and on the 16K coding shape prefill is 71% of wall clock.

## Correctness

Every kernel change is gated on bit-identical output against the HBM path at a
fixed grid, which `mla_cache_full_footprint.py` already asserts
(`rtol=0, atol=0`) across random, sorted and clustered patterns. A prefetch or
pipelining change must not alter arithmetic at all — unlike a grid change, there
is no reduction-order excuse available here, so any difference is a bug.

## Risks

- **Shared memory is full.** Every idea must be smem-neutral. This is the
  binding constraint on the whole design.
- **The C2C link is already saturated by the MoE cold tier** — Phase 26
  measured 379 GB/s against a 373 GB/s achievable rate. KV gathers would
  contend with cold expert weight streams. The isolated numbers here are
  therefore optimistic, and Phase 4 is where that shows up.
- **Vendored dependency.** `.deps/flashmla-src` is fetched by CMake; changes
  there are not tracked by this repo and will be lost on a clean rebuild. Phase
  0 must establish where the patch lives permanently.
- **We are modifying a third-party kernel** whose upstream may not want a
  far-memory path.


---

# Result

## Phase 0 — rebuild loop: PASSED

`ninja _flashmla_C` rebuilds the four sparse instantiations in ~55 s and
`rebuild.sh` installs the `.so` and snapshots `.deps/flashmla-src` to
`kernel.patch`. Rebuilt from unmodified source, the benchmark reproduced
**157.9 GB/s** against 156.8 before — within noise, so the loop is sound.

## Phase 1 — L2 prefetch: no effect

| build | Grace GB/s (random, tok=16) | HBM GB/s |
| --- | ---: | ---: |
| baseline, rebuilt | 157.9 | 771.8 |
| + next-block L2 prefetch | **159.0** | 767.0 |

+0.7%, i.e. nothing, against a **≥220 GB/s** gate.

**A near-miss worth recording.** The first build compiled cleanly and changed
nothing: `_flashmla_C` takes `COMPILE_FLAGS ${VLLM_GPU_FLAGS}`, but the define
had been appended to `VLLM_FLASHMLA_GPU_FLAGS`, which only
`_flashmla_extension_C` uses, so `#if FLASHMLA_GRACE_L2_PREFETCH` silently
evaluated false. Caught by diffing SASS `CCTL` counts against the backed-up
pre-patch binary (10 vs 10; after the fix, 10 vs 22). Without that check the
run would have been recorded as "prefetching does not help far memory" — a
false refutation. **Verifying the instruction reached the binary belongs in the
loop, not after it.**

## Why: the kernel is occupancy-bound, and the "headroom" was an artifact

The plan claimed 2.3x headroom from comparing the kernel's 157 GB/s against a
plain Triton gather's 361 GB/s. That gather was running **1056-2112 CTAs**. The
MLA kernel runs **132** — one per SM, because 224 KiB of shared memory admits
only one CTA, at 384 threads, i.e. **19% warp occupancy**.

Measured at matched parallelism, 512 B granularity:

| CTAs | HBM GB/s | Grace GB/s |
| ---: | ---: | ---: |
| **132** (the kernel's operating point) | 111 | **73** |
| 264 | 218 | 139 |
| 528 | 412 | 252 |
| 1056 | 768 | 382 |
| 2112 | 1356 | 368 |

**A plain gather at 132 CTAs reaches 73 GB/s from Grace; the MLA kernel reaches
159.** It is already 2.2x *better* than a straightforward gather at its own
occupancy, not 2.3x worse than achievable. The comparison that motivated this
experiment was against a ceiling the kernel structurally cannot reach.

Two consequences kill the remaining interventions:

- Reaching ~382 GB/s needs ~1056 CTAs, i.e. cutting shared memory roughly 8x to
  ~28 KiB/CTA. That is a different kernel — smaller tiles, Q out of shared
  memory, fewer stages — not a modification, and it would move HBM performance
  too, so the win is not even directionally safe.
- **Higher occupancy makes the ratio worse, not better.** HBM keeps scaling
  while Grace saturates near 385, so HBM/Grace goes from 1.5x at 132 CTAs to
  6.5x at 2112. A successfully re-occupied kernel would face a *larger*
  relative penalty.

Interventions A (measured, nil), B, C and D all fail for the same reason: they
need shared memory the kernel does not have, or parallelism it cannot reach.

## Status of the source

Everything was reverted. The work is preserved as patches:

| file | what |
| --- | --- |
| `kernel.patch` | `prefetch_l2` helper + next-block prefetch in `splitkv_mla.cuh` |
| `cmake.patch` | `FLASHMLA_GRACE_L2_PREFETCH` plumbed to `_flashmla_C` |

`.deps/flashmla-src` is a CMake FetchContent checkout, so the kernel edit would
have been discarded by a clean rebuild regardless; the patch is the durable
form. The pre-experiment `_flashmla_C.abi3.so` was restored from
`/e/scratch/profound/naeimitabiei1/_flashmla_C.abi3.so.bak-20260816`.

## What would still be worth knowing

The measurement that would change this verdict is a sparse MLA kernel designed
for high occupancy — small tiles, no 72 KiB K buffers. That is a research
project, not a tuning pass, and the ratio table above says the payoff shrinks as
occupancy rises. Recorded so nobody re-derives the 2.3x-headroom framing.

Unrelated and still open: **FA3 GQA reaches only 52 GB/s from Grace**, flat
across batch and context, which no measurement here explains. TMA is not the
cause (a TMA descriptor load on host memory hits 419.7 GB/s). If GQA models
matter, that is its own investigation.

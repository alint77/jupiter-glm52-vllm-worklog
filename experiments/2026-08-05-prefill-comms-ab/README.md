# Prefill collectives: NCCL protocol and chunk size

Status: **Complete — the premise was wrong.** The collectives already run at
hardware speed; 75% of their measured time is waiting, not moving bytes.

## The question

The [production profile](../2026-08-05-prod-profile/README.md) measured prefill
spending **593 ms/chunk, 23.8%, in collectives**, with two properties that make
it look misconfigured rather than expensive:

- `vllm::all_reduce` on `(8192, 6144)` bf16 = **96 MiB**, 160 times per chunk,
  2,064 us each — a **70 GB/s** ring bus rate.
- The kernel is `ncclDevKernel_AllReduce_Sum_bf16_RING_LL`: NCCL's
  **low-latency** protocol, whose 8-byte flits are half flag bytes, on a 96 MiB
  message across 24 channels. Nothing in `jupiter-env.sh` sets `NCCL_PROTO`, so
  NCCL's own tuner chose it.

A follow-up measurement (`analyze_prefill_overlap.py`) showed **0.0% overlap**
between communication and compute on all four ranks — comm 592.9 + compute
1902.6 + idle 123.7 = 2619.2 ms against a 2619.1 ms wall, so the three partition
the timeline exactly. 99.9% of both is issued to the **main stream**, so the GPU
cannot run a collective and a GEMM at once.

That zero is what makes this test worth running first: with nothing hidden
behind compute, **any bandwidth improvement converts 1:1 into wall clock**.

## Design

Four arms, 2x2, all on one node — cross-node variance is ~2.5% on Marlin alone,
larger than one of the effects being measured.

| arm | `NCCL_PROTO` | `max_num_batched_tokens` |
| --- | --- | ---: |
| `baseline` | unset (LL) | 8192 |
| `proto-simple` | `Simple` | 8192 |
| `chunk16k` | unset | 16384 |
| `chunk16k-simple` | `Simple` | 16384 |

The grid separates the two effects and shows whether they stack. They could
interact either way: a 16K chunk doubles each all-reduce to 192 MiB, which may
push NCCL further from LL's comfort zone, or the larger message may improve LL
enough to narrow the gap.

**Metric.** A 16K prompt with `max_tokens=1`, so TTFT *is* the prefill and no
decode contaminates it. Prefix caching off, 16 unique prompts, one discarded
warmup, two measured repetitions per arm.

**Plus one profiled prefill per arm.** The trace is the guard against fooling
ourselves: if `Simple` improves TTFT but the kernel names still read `RING_LL`,
the protocol is not what changed and the delta means something else.
`compare.py` reads the protocol out of the kernel names and reports it beside
the timing.

**Chunk-size caveat.** Doubling `max_num_batched_tokens` roughly doubles the
peak activation (2.15 GB at 8192) against a 7 GB reserve, so the 16K arms may
fail the post-warmup HBM audit. The arm loop logs and continues rather than
killing the job.

**Scope caveat.** `NCCL_PROTO` is global, so it also applies to decode's DCP
all-gathers, where LL may be the right choice for small messages. This test
speaks only to prefill; if Simple wins, whether it can be scoped is a separate
question.

## Two failed attempts before this one

**`1245124`** — cancelled by hand at 57 s, not a failure. A fourth arm
(`chunk16k-simple`) was added and same-node comparability was worth more than
the minute already spent.

**`1245137` — died on an inode quota, and the cause was self-inflicted.**
Inductor failed on every rank with:

```text
OSError: [Errno 122] Disk quota exceeded:
  '/e/project1/profound/alint77/.marlin-caches/...'
```

`/e/project1` was 33% full of 11 PB, but a bare `touch` failed, so this is a
**file-count quota, not a space quota** — the same failure mode
[`2026-07-29-marlin-smem-monopoly`](../2026-07-29-marlin-smem-monopoly/README.md)
recorded for `/e/scratch`, now reached on the filesystem the caches were moved
*to* as that fix.

The cause was this job creating **one cache root per arm**, which is exactly
what commit `8a9de12` ("Share one compile-cache root per shape and guard the
quota") exists to prevent. Deleting the two roots this job had created restored
file creation immediately.

The job now uses **one shared cache root** for all four arms — reusing
`vllm-cache-prod-profile`, already warm for this exact server config, so the
8192 arms need almost no compilation. `NCCL_PROTO` does not affect compilation
and Inductor keys on the graph hash, so sharing is safe.

**Standing risk:** 30 cache roots have accumulated under `.marlin-caches/`,
about 26 of them from finished experiments (16 from the 07-29 shared-memory
work, 6 from grid-fit, 4 older). They are regenerable compiler caches and they
are what is pressing a **project-wide** quota, so this can affect other users of
`profound`, not just this branch. They were left in place rather than deleted.

## Results

Job `1245581` (jobs `1245124`, `1245137`, `1245333` failed for the reasons
above). Two arms completed; the 16K arms were refused by the tiered validator.

| arm | TTFT | vs base | spread | protocol | comm/chunk | AR/chunk |
| --- | ---: | ---: | ---: | :-: | ---: | ---: |
| `baseline` | 5.127 s | 1.000x | 0.17% | LL | 396.6 ms | 225.3 ms |
| `proto-simple` | 5.114 s | 1.003x | 0.16% | LL | 397.0 ms | 225.9 ms |
| `chunk16k` | refused | | | | | |
| `chunk16k-simple` | refused | | | | | |

**The protocol never changed.** Both arms show
`ncclDevKernel_AllReduce_Sum_bf16_RING_LL` and identical comm time; the 0.3%
TTFT difference is inside the 0.17% repeat spread. `NCCL_PROTO` exported in the
shell did not reach the communicator. Note vLLM itself sets it *from Python*
(`batch_invariant.py:975`) rather than relying on the environment.

**The 16K arms were refused by design**, not crashed:
`vllm/config/vllm.py:2316` raises "Tiered MoE initially requires
max_num_batched_tokens=8192".

## What the standalone probe showed

[`2026-08-05-nccl-proto-probe`](../2026-08-05-nccl-proto-probe/README.md)
measured the identical 96 MiB all-reduce outside vLLM:

| `NCCL_PROTO` | 96 MiB | busbw |
| --- | ---: | ---: |
| unset | 514 us | 294 GB/s |
| Simple | 520 us | 290 GB/s |
| LL128 | 557 us | 271 GB/s |
| LL (forced) | 1,105 us | 137 GB/s |

`NCCL_PROTO` *is* honoured there, and **NCCL's default already picks the fast
protocol**. There was never a protocol fix to make.

## The real finding: the median is already optimal, the tail is not

Per-rank all-reduce duration in the production prefill trace:

| rank | n | mean | p10 | **p50** | p90 | max |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 320 | 2,064 us | 472 | **520** | 6,541 | 11,067 |
| 1 | 319 | 2,008 | 472 | **491** | 6,422 | 11,841 |
| 2 | 319 | 2,029 | 470 | **482** | 6,365 | 11,239 |
| 3 | 319 | 1,983 | 485 | **500** | 6,109 | 9,893 |

**The median all-reduce (482-520 us) equals the standalone hardware optimum
(514 us).** The bandwidth was never bad. The "70 GB/s bus rate" reported in the
production profile was an artifact of averaging wait time into transfer time.

p90 is 6.1-6.5 ms, twelve times the median. Cross-rank spread of the means is
**4.0%**, so no rank is slow — every rank sees the same fast-median, heavy-tail
distribution, which is the signature of **symmetric arrival skew**: whoever
reaches the collective first blocks until the straggler arrives, and the
straggler rotates.

Pricing it: 160 all-reduces at the p50 would be ~80 ms/chunk against the
330 ms measured, so **~250 ms/chunk — 9.5% of prefill — is waiting, not
communication.**

### Unresolved

Production's kernels are named `RING_LL` yet reach a p50 that matches *Simple*
standalone, while forced-LL standalone is twice as slow. The two are therefore
not the same operating point — channel count is the likely difference, since the
production launch is `grid=(24,1,1)` and several communicators (TP, DCP, EP)
coexist. This does not affect the conclusion, which rests on the median/tail
split, but it is not explained.

## Consequence: the lever moves

Prefill's collective cost is not bandwidth and not protocol. It is **EP load
imbalance across ranks**, which is the same quantity replica assignment was
built for — and which cut decode's skew 61.5% and all-reduce residency 55.4%
in [`2026-07-31-replica-scheduling-v2`](../2026-07-31-replica-scheduling-v2/README.md).

Replica assignment does not run in prefill.
`tiered_moe_execution.py:221` returns early when
`num_tokens > tiered_overlap_max_tokens`, which is 16.

That is the fourth decode-only gate found in this area, all with the same shape:

| gate | site | effect in prefill |
| --- | --- | --- |
| tight shared-memory launch policy | `marlin_moe.py:142` | legacy launch (measured: worth little) |
| hot/cold stream overlap | `apply_tiered` | tiers run serially |
| **replica assignment** | `tiered_moe_execution.py:221` | **no EP balancing** |
| `max_num_batched_tokens` pin | `vllm/config/vllm.py:2316` | chunk size fixed at 8192 |

## Follow-up: the skew is per-layer, and replica assignment cannot fix it

The obvious next step from the finding above was to ungate replica assignment
for prefill. Two measurements say no.

**Aggregate rank load is already balanced.** Per-rank totals over a chunk:

| rank | Marlin ms | all compute ms |
| ---: | ---: | ---: |
| 0 | 883.7 | 1,902.9 |
| 1 | 890.5 | 1,917.1 |
| 2 | 886.8 | 1,915.5 |
| 3 | 894.5 | 1,920.3 |

Spread is **1.21% on Marlin and 0.91% on all compute** — max-minus-mean of
6.3 ms, which cannot explain a 250 ms tail.

**The imbalance is per-layer, and it rotates.** Grouping Marlin launches into
routed layers and comparing ranks at each one:

| | |
| --- | ---: |
| per-layer rank spread, p50 | **45.2%** |
| p90 | 74.5% |
| max | 89.4% |
| summed per-layer max-minus-mean | **185.7 ms/chunk** (60 of 75 layers in window; ~232 ms scaled) |

That accounts for most of the ~250 ms of all-reduce time above the p50, and it
reconciles the two facts: each layer is badly imbalanced, every barrier cashes
that in, and because the straggler rotates the totals still come out level.

**Why the existing mechanism does not apply.** `tiered_moe_scheduler.py` states
its own assumption: *"Every active cold expert costs the same Grace weight read
regardless of how many tokens route to it, so the objective is purely to
minimise the maximum per-rank count of active cold experts. No cost constants
appear."* That holds at decode, which is weight-streaming bound at 45.32 us per
active cold expert. It is false at prefill, which
[the profile](../2026-08-05-prod-profile/README.md) measured as compute-bound by
**8.1x over its streaming floor** — cost there scales with tokens, not expert
count. And at 8,192 tokens essentially every expert is active, so a count-based
min-max is degenerate: every rank already holds a full active set.

Mechanically it does not fit either: the fused kernel is a single CTA with
`tl.arange(0, ROUTE_BLOCK)` sized for decode's ~128 routes against prefill's
65,536, and it emits `BLOCK_M=16` metadata where prefill's Marlin call selects
`block_size_m=64`.

So the lever is real and now priced at **185-232 ms/chunk, 7-9% of prefill**,
but taking it needs a *token-weighted* min-max assignment — a different
objective and a different kernel, not an ungating.

## Also settled: the chunk-size lever is a bad trade

`plan_tiered_glm_runtime_buffers` is fully parametric in
`max_num_batched_tokens`, so the `!= 8192` guard hides no assumption. But the
Marlin arenas are `M * 393,216` bytes: **3.22 GB/rank at 8,192 and 6.44 GB at
16,384**. Against the measured 7.15 GiB free and a 5.59 GiB audit floor, +3.22 GB
fails the post-warmup audit outright. Paying for it in residency means moving
**160 experts/rank** (3.22 GB / 20.05 MB) from HBM to Grace, dropping hot
residency 59.8% -> 56.5%, which costs roughly 1.8 ms/step of decode
(~0.7 more active cold experts per layer x 35.6 us x 75 layers) to save ~123 ms
of per-chunk prefill fixed cost. Phase 32's conclusion was to *buy* HBM
residency; this spends it. Dropped.

## Files

| File | What |
| --- | --- |
| `job-ab.sh` | The four arms, one node |
| `capture_prefill.py` | One profiled prefill chunk per arm |
| `compare.py` | TTFT, protocol read from kernel names, collective cost |
| `*-ttft-r*.json` | Raw per-arm timing |
| `*-server.{out,err}` | Per-arm server logs |

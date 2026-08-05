# Prefill collectives: NCCL protocol and chunk size

Status: **In flight** (job `1245333`). Design and the two failed attempts are
recorded here; results follow.

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

Pending job `1245333`.

## Files

| File | What |
| --- | --- |
| `job-ab.sh` | The four arms, one node |
| `capture_prefill.py` | One profiled prefill chunk per arm |
| `compare.py` | TTFT, protocol read from kernel names, collective cost |
| `*-ttft-r*.json` | Raw per-arm timing |
| `*-server.{out,err}` | Per-arm server logs |

# MTP3 c=1 kernel and communication profile (2026-09-04)

Torch-profiler captures of **prefill** and **decode** for the configuration the
same-day speculator comparison recommends for Claude Code traffic: GLM-5.3
W4A16 tiered, MTP K=3, `max_num_seqs=1`, DCP 1, 400K context. The comparison
answered *which speculator*; this answers *where the time goes*, so the next
round of work can be kernel- and comms-level rather than configuration-level.

Job **1665068** on `jpbo-013-03`. Traces under
`/e/project1/profound/alint77/traces/mtp3-profile-1665068/{prefill,decode}`.

## Why both phases

At 96K context a request is roughly half prefill: TTFT is 23.17 s and the
decode of a ~1.4K-token answer is comparable. Prefill has never been profiled
at this context length on this fork -- the 2026-08-05 production capture used
16K prompts -- so half of every request is currently unmeasured. Both captures
therefore replay the **same** ~96K Claude-Code-shaped prompt, which also keeps
decode at the context length it actually serves; profiling decode off a short
prompt would understate attention.

Prefix caching is off (`--no-enable-prefix-caching`) so a repeated prompt
cannot skip the prefill under measurement.

## Baseline for distortion

The profiler perturbs what it measures, so the profiled decode step is compared
against this exact shape run **unprofiled**, not against the +4.3% figure from
the 2026-08-05 GLM-5.2 DCP4 campaign. From the four-arm comparison's `spmtp3`
arm, 60 requests:

| quantity | unprofiled value |
| --- | --- |
| decode step | 29.15 ms median (mean 29.20, range 27.6-32.6) |
| acceptance length | 2.713 |
| TTFT at 96K | 23.17 s median -> 4145 tok/s, ~1.93 s per 8192-token chunk |

If the profiled decode step lands within a few percent of 29.15 ms, decode
absolutes are quotable; otherwise only the structure is.

## Two harness bugs fixed before running

* **`wait_for_decode` watched the wrong counter.** `2026-08-05-prod-profile`
  polled `prompt_tokens_total`, which advances only when a prefill *chunk
  completes* (~1.9 s here). Two polls a quarter second apart therefore read as
  "stable" in the middle of a chunk, and two of that campaign's four captures
  hold the wrong phase -- a failure that harness documents about itself. This
  version gates on `generation_tokens_total`.
* **The prefill window opened on the clock.** The capture fired the request and
  armed the profiler in the same breath, so tokenizing a 441K-character prompt
  and admitting it were inside the window. It now gates on
  `num_requests_running`.

## Analysis

`analyze.py` keeps the launch-correlation attribution from
`2026-07-29-marlin-smem-monopoly/analyze_step_budget.py` -- a kernel belongs to
the step whose CPU launch produced it, not to the step whose annotation window
its GPU timestamps land in -- but not that file's `summarize`, which cannot
describe an eager phase (it averages per-layer and CUDA-graph statistics
unconditionally, so prefill's empty sequences raise from `fmean`) and whose
strict census asserts a GLM-5.2 DCP4 kernel structure this DCP1 capture cannot
satisfy. The census is reported here, never asserted.

It adds the **communication split**. The shared bucket map folds custom
all-reduce, NCCL all-gather and DCP NCCL into a single line, and `family()`
labels every NCCL kernel a vocabulary all-gather; the three have different
fixes, so comms kernels are reported individually by name. A top-N kernel table
comes with it, because prefill kernels missing from the family list would
otherwise be charged to glue.

Validated against `prod-profile-1240390/decode-c1`, which it reproduces to the
published figures: routed 24.06%, comms 19.09%, dense 18.85%, glue 15.61%,
GPU-empty 10.79%.

```
.venv/bin/python analyze.py \
  /e/project1/profound/alint77/traces/mtp3-profile-1665068 \
  --json budget.json
```

## Results

Full tables in `budget-1665068.txt`, `budget.json`, `streams-1665068.txt` and
`overlap-1665068.txt`. Captured 6 prefill chunks and 60 decode steps per rank.

### Distortion: opposite conclusions per phase

| phase | profiled | unprofiled | distortion |
| --- | --- | --- | --- |
| prefill chunk | 1902.6 ms | ~1930 ms (23.17 s / 12 chunks) | nil |
| decode step | 32.87 ms | <= 29.15 ms | **>= +12.8%** |

Prefill absolutes are quotable. Decode's are not -- the inherited +4.3% from
the GLM-5.2 DCP4 campaign badly understates it here, which is why it was
re-measured. Decode is reported as shares only. The decode bound is one-sided:
29.15 ms is a client-side SSE gap, which includes streaming overhead and so is
an upper bound on the unprofiled engine step.

### Budget

| bucket | prefill (ms/chunk, %) | decode (%) |
| --- | --- | --- |
| routed experts (W4 Marlin) | 783.1 / 39.7% | 36.1% |
| attention (FlashMLA + DSA + KV) | 652.5 / 33.0% | 7.3% |
| TP communication | 177.8 / 9.0% | 21.7% |
| dense/shared GEMMs | 153.6 / 7.8% | 17.2% |
| glue, elementwise, uncategorized | 144.0 / 7.3% | 8.7% |
| MoE routing, activation, sum | 54.0 / 2.7% | 4.6% |
| GPU-empty host/graph gaps | 9.7 / 0.5% | 4.6% |

Prefill is GPU-bound: busy is 99.5% of span, so there is no host-side slack to
recover and every gain must come from a kernel or from overlap.

### The comms line is two unrelated problems

Splitting it by kernel, which the shared bucket map cannot do, separates them.
The dispatch log explains why the two phases use different kernels at all:

```
SymmMemCommunicator: symmetric memory multicast operations are not supported.
Using ['CUSTOM', 'PYNCCL'] all-reduce backends (in dispatch order) for 'tp:0'
```

Decode's 4-token operand is small enough for CUSTOM; prefill's 8192-token
operand exceeds the custom all-reduce's size limit and falls through to PYNCCL.
That is the designed dispatch, not a fallback bug.

#### Decode: 89% of the "communication" is not communication

`cross_device_reduce_1stage`, 166x per step, 7.2 ms. It is sharply bimodal, and
the split is **structural, not jitter** -- across all 60 steps:

| ordinals | count | mean of per-ordinal medians | total |
| --- | --- | --- | --- |
| even | 83 | 65.2 us | 5.415 ms/step |
| odd | 83 | 5.9 us | 0.490 ms/step |

There are two all-reduces per layer. The **first of each pair absorbs the
skew**; by the second, the ranks are already in step. Every ordinal, the slow
ones included, has a floor of ~4.0 us across the 60 steps, which is the actual
cost of moving 40 KB over NVLink:

```
166 x 4.01 us floor          = 0.666 ms/step   <- the collective
sum of per-ordinal medians   = 5.905 ms/step
=> wait absorbed at barrier  = 5.238 ms/step   <- rank skew
```

So decode's "21.7% communication" is really **~16% of the step spent waiting on
rank divergence and ~2% moving data.** Per-rank totals are balanced (6.6, 7.2,
7.3, 7.5 ms), so no single rank is the laggard -- the divergence is per layer,
which points at routed-expert load imbalance rather than a slow GPU.

#### Prefill: bandwidth-bound, fully exposed, possibly the wrong protocol

`ncclDevKernel_AllReduce_Sum_bf16_RING_LL`, 160x per chunk, 177.7 ms.
`record_shapes` gives the operand as `[8192, 6144]` bf16 = 100.7 MB; a ring
moves 2*(3/4)*100.7 MB per GPU in a 992 us median = **152 GB/s algorithmic**.

The open question is the **`_LL` protocol**. LL is NCCL's low-latency
small-message protocol and carries flags inline, roughly doubling wire bytes;
for a 100 MB operand the tuner would normally pick Simple. No `NCCL_PROTO`,
`NCCL_ALGO` or any other `NCCL_*` variable is set anywhere in the launch path
(`jupiter-env.sh`, `run-server.sh`, `job-profile.sh` are all clean), so this is
NCCL's own choice. **Not yet established whether 152 GB/s is near this node's
achievable peak** -- that needs an `all_reduce_perf -b 100M -e 100M` on a
Booster node, and until it is run no claim is made either way.

Both collectives are **100% exposed**: across every step sampled, no other
kernel is ever resident while an all-reduce runs. Neither phase has any
communication/compute overlap.

### Hot/cold Marlin overlap: already optimal in decode, absent in prefill

Marlin launches in groups of four per layer (w13/w2 x hot/cold):

| phase | fork pattern | cumulative | union | saving |
| --- | --- | --- | --- | --- |
| decode | 75 of 77 groups fork 2+2 across two streams | 15.75 ms | 12.29 ms | 22.0% |
| prefill | **all 76 groups on one stream** | 775.9 ms | 775.9 ms | 0% |

The decode floor is `max(hot, cold)` **per layer**, not the max over the step --
hot carries far more experts than cold, so perfect overlap saves 22.8%, not 50%:

```
cumulative 15.752  measured union 12.287  floor max(hot,cold) 12.166
realised 22.0% of a possible 22.8%;  remaining headroom 0.121 ms/step
```

That is 0.4% of the step. **There is no decode overlap headroom left**, exactly
as `2026-07-29-marlin-smem-monopoly` concluded when its shared-memory fix landed
the union on `max(hot, cold)`. An earlier reading of this capture put the
headroom at ~6.8 ms by treating the three observed streams as collapsible to
one; that is wrong, and the per-layer measurement above is what settles it.

Prefill never forks at all -- the `tiered_overlap_max_tokens` gate keeps
large-M chunks on one stream, which is the designed behaviour and a question
that campaign explicitly parked.

## Ranked candidates

1. **Decode rank skew -- 5.24 ms/step, ~16% of the step.** The largest
   addressable item in decode by a wide margin, and it is a routed-expert
   load-balance problem, not a comms problem. The signature is specific enough
   to chase: even-ordinal all-reduces, uniform across ranks, ~4 us floor.
2. **Prefill exposed all-reduce -- 177.7 ms/chunk, 9.0%, ~2.1 s of a 23.2 s
   TTFT.** Two independent angles: the `_LL` protocol choice (test
   `NCCL_PROTO=Simple`/`LL128` against an `all_reduce_perf` ceiling), and the
   total absence of comm/compute overlap. Note `fuse_allreduce_rms` is **not**
   the route -- it is already known to fault with an illegal memory access on
   this stack (`HANDOFF.md:520`, `README.md:143`), which is why every launcher
   sets it false.
3. **Prefill attention -- 33.0%** (`flash_fwd_splitkv_mla_fp8_sparse` 550.7 ms,
   DSA indexer 97.6 ms, `topKPerRowPrefill` 34.2 ms per chunk). Understated
   here: see the sampling caveat below.
4. **Prefill Marlin never forks** -- 775.9 ms/chunk on one stream. Only worth
   revisiting because the smem monopoly that made forking useless has since been
   fixed; the gate predates that fix.
5. **Closed: decode hot/cold overlap.** 0.121 ms/step remains. Do not spend on it.

## Caveats

* **Prefill sampling.** The window opened on first schedule, so these are chunks
  1-6 of ~12. Marlin and the all-reduce are per-token and unaffected, but the
  DSA indexer scores against all preceding KV, so **attention's 33.0% is an
  underestimate** for the full prefill.
* **Decode workload.** The decode capture is a temperature-0 `ignore_eos`
  completion, which degenerates: AL 3.908 against the Claude-Code corpus's
  2.713. A step still verifies 4 tokens and the kernel structure is unchanged,
  so shares hold, but this is not CC-shaped output.
* **Prefill distortion.** The 1930 ms baseline is TTFT/12, which includes
  tokenization and the first decode step and averages all 12 chunks, against a
  profiled mean of early chunks. The distortion is small -- within that
  baseline's own uncertainty -- rather than provably nil.

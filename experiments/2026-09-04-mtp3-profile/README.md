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

### The comms line is one problem, in both phases: rank skew

Splitting by kernel separates the two phases' collectives. The dispatch log
explains why they differ at all:

```
SymmMemCommunicator: symmetric memory multicast operations are not supported.
Using ['CUSTOM', 'PYNCCL'] all-reduce backends (in dispatch order) for 'tp:0'
```

Decode's 4-token operand takes CUSTOM; prefill's 8192-token operand exceeds the
custom all-reduce's size limit and falls to PYNCCL. That is the designed
dispatch, not a fallback bug.

The decisive measurement is **cross-rank**: align the four ranks by step index
and, for each (step, ordinal), compare the minimum duration across ranks
against the mean. A collective that is genuinely slow is slow on all four ranks
at once; one that is waiting is fast on whichever rank arrives last.

| | decode (166 AR/step, 60 steps) | prefill (160 AR/chunk, 6 chunks) |
| --- | --- | --- |
| cross-rank min, p50 | 3.8 us | 486.9 us |
| cross-rank mean, p50 | 6.9 us | 905.8 us |
| sum of minima | 0.632 ms/step | 77.5 ms/chunk |
| sum of means | 7.174 ms/step | 177.7 ms/chunk |
| **skew** | **6.543 ms/step (91.2%)** | **100.2 ms/chunk (56.4%)** |
| all four ranks slow at once | 1 / 9960 = 0.0% | 0 / 960 = 0.0% |

**Neither collective is ever slow on all four ranks simultaneously.** Both are
arrival skew. In step terms that is **6.54 ms of a 32.9 ms decode step (19.9%)**
and **100.2 ms of a 1974 ms prefill chunk (5.1%)**.

This kills the earlier reading that prefill was bandwidth-bound. Its 992 us
median over an identical 100.7 MB operand spans min 461 / p90 2387 / max 3907 us
-- an 8.5x spread no bandwidth-bound collective has. At the 487 us cross-rank
minimum the ring achieves **~310 GB/s algorithmic, not the 152 GB/s the median
implied**, so the "already at NVLink peak, not a lever" conclusion was measuring
skew and is withdrawn. The `_LL` protocol question stays open but is now
secondary to the skew.

### Rank 1 arrives last, in both phases

In a spin-wait barrier the rank that *waits least* is the one that *arrives
last*. Counting which rank holds the minimum for each (step, ordinal), against
25% for chance:

| | r0 | r1 | r2 | r3 |
| --- | --- | --- | --- | --- |
| decode | 24.4% | **43.1%** | 18.0% | 14.5% |
| prefill | 12.0% | **58.8%** | 17.4% | 11.9% |

Rank 1 is the laggard in both phases, which is why its all-reduce *total* is the
lowest of the four (6.67 ms/step against 7.21 / 7.28 / 7.54). An earlier pass
read those balanced totals as "no single rank is the laggard" -- that is exactly
backwards for a spin-wait barrier.

Being consistent across both phases and every layer points away from routing
randomness and toward something rank-specific: NUMA binding, C2C bandwidth on
that socket, or a per-rank placement difference. This project has hit that
failure mode before (`numa-bind-uva`).

**Not established:** which of the two per-layer all-reduces carries the wait.
Even ordinals hold it (83 ordinals, 65.2 us mean-of-medians, 5.415 ms/step)
against the odd ones (5.9 us, 0.490 ms/step), and the slow run starts at ordinal
8, consistent with `first_k_dense_replace = 3` leaving the first layers with no
experts to skew. But the kernel immediately preceding an all-reduce is not its
producer under three concurrent streams, so the post-MoE / post-attention
mapping is **not** confirmed and no mechanism is claimed from it.

### Hot/cold Marlin overlap: already optimal in decode, absent in prefill

Marlin launches in groups of four per layer (w13/w2 x hot/cold):

| phase | fork pattern | cumulative | union | saving |
| --- | --- | --- | --- | --- |
| decode | 75 of 77 groups fork 2+2 across two streams | 15.75 ms | 12.29 ms | 22.0% |
| prefill | **all 76 groups on one stream** | 775.9 ms | 775.9 ms | 0% |

The decode floor is `max(hot, cold)` **per layer**, not the max over the step:

```
cumulative 15.752  measured union 12.287  floor max(hot,cold) 12.166
realised 22.0% of a possible 22.8%;  remaining headroom 0.121 ms/step
```

0.4% of the step. **No decode overlap headroom remains**, exactly as
`2026-07-29-marlin-smem-monopoly` concluded. An earlier reading of this capture
put it at ~6.8 ms by treating the three observed streams as collapsible to one;
the per-layer measurement above is what settles it. Prefill never forks --
the `tiered_overlap_max_tokens` gate keeps large-M chunks on one stream.

### Dense/shared GEMMs are 17.2% of decode, and they are bf16

The checkpoint quantizes routed experts only. Its `ignore` list is
`self_attn`, `shared_experts`, the dense MLP of the first three layers,
`lm_head`, embeddings and norms -- so everything outside `mlp.experts.N.*`
stays bf16:

| bucket | size | dtype |
| --- | --- | --- |
| routed experts | 384.75 GiB | 342.00 quantized + 42.75 bf16 scales |
| **self_attn** | **24.67 GiB** | **bf16** |
| **shared experts** | **5.34 GiB** | **bf16** |
| **dense MLP (layers 0-2)** | **1.27 GiB** | **bf16** |
| embed/lm_head, norms | 3.91 GiB | bf16 |

Those three bf16 blocks are read on **every** step: 31.28 GiB, 7.82 GiB per GPU
at TP4, which at ~4 TB/s is a **2.10 ms/step streaming floor against the 5.734
ms/step measured**. So the bucket is not weight-bound -- ~3.6 ms/step is
overhead in many small M=4 GEMMs (`nvjet_..._64x8_...` at 214x and 161x per
step). Quantizing `self_attn` would move the floor to ~0.9 ms, but the larger
share of this bucket is launch and tail inefficiency, not bytes.

## The prefill chunk, kernel by kernel

A complete per-kernel breakdown of one 8192-token chunk -- launch-order
sequence, roofline with arithmetic intensity, the communication latency and
bandwidth split, and the per-rank delta -- is in **[PREFILL.md](PREFILL.md)**.
Headline: prefill is single-stream, so shares are additive; sparse MLA launches
64 query heads for the 16 the rank owns and is **10.2% useful**; the cold
expert tier costs 2.1x hot for fewer experts; and 56% of all communication time
is arrival skew, of which 78.8 of 100.2 ms reproduces the per-layer
routed-expert imbalance to within 0.7%.

## Ranked candidates

1. **Rank-1 arrival skew -- 6.54 ms/step (19.9% of decode) and 100.2 ms/chunk
   (5.1% of prefill).** One rank, both phases, every layer. The largest single
   item in the profile and the only one that pays in both phases. Start with
   NUMA/C2C on rank 1, not with the router.
2. **Dense/shared bf16 GEMMs -- 5.73 ms/step (17.2% of decode)** against a
   2.10 ms streaming floor. Mostly small-GEMM overhead; quantizing `self_attn`
   is a second, quality-risky lever on top.
3. **Prefill attention -- 33.0%** (`flash_fwd_splitkv_mla_fp8_sparse` 550.7 ms,
   DSA indexer 97.6 ms, `topKPerRowPrefill` 34.2 ms per chunk). Understated;
   see caveats.
4. **No comm/compute overlap in either phase** -- every all-reduce is 100%
   exposed. Worth attacking only after the skew, since overlap would hide the
   77.5 ms of real prefill transfer but not the 100.2 ms of waiting.
   `fuse_allreduce_rms` is **not** the route: it faults with an illegal memory
   access on this stack (`HANDOFF.md`, `README.md`), which is why every launcher
   sets it false.
5. **Prefill Marlin never forks** -- 775.9 ms/chunk on one stream, gated by
   `tiered_overlap_max_tokens`, which predates the smem fix.
6. **Closed: decode hot/cold overlap.** 0.121 ms/step. Do not spend on it.

Still worth having but no longer gating anything: an `all_reduce_perf
-b 100M -e 100M` ceiling. The trace's own cross-rank minimum already bounds the
achievable bandwidth at ~310 GB/s.

## Caveats

* **Prefill sampling.** Chunks 1-6 of ~12. Marlin and the all-reduce are
  per-token and unaffected, but the DSA indexer scores against all preceding KV,
  so **attention's 33.0% is an underestimate** for the full prefill. Only 6
  aligned chunks back the prefill skew figure, against 60 decode steps.
* **Decode workload.** A temperature-0 `ignore_eos` completion, which
  degenerates: AL 3.908 against the Claude-Code corpus's 2.713. A step still
  verifies 4 tokens and the kernel structure is unchanged, so shares hold, but
  this is not CC-shaped output.
* **"GPU busy 99.5%" in prefill counts NCCL spin-wait as busy.** Kernel-resident
  is 99.5%; actual compute is nearer 94%.
* **Profiler distortion.** Small for prefill -- the 1930 ms baseline is TTFT/12
  and carries its own uncertainty, so "within noise" rather than provably nil.
  Decode is **>= +12.8%**, so decode is reported as shares only.
* **`NCCL_*` environment.** Checked in `jupiter-env.sh`, `run-server.sh` and the
  job script, all clean -- but not in the module-loaded environment or
  `/etc/nccl.conf`, so "nothing forces `_LL`" is not yet established.

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

Splitting it by kernel, which the shared bucket map cannot do, separates them:

* **Prefill is one `ncclDevKernel_AllReduce_Sum_bf16_RING_LL`, 160x per chunk,
  177.7 ms.** `record_shapes` gives the operand as `[8192, 6144]` bf16 =
  100.7 MB, and the ring moves 2*(3/4)*100.7 MB per GPU in a 992 us median =
  **~151 GB/s, at NVLink peak.** This kernel is bandwidth-bound and already
  near-optimal; tuning the collective is not a lever.
* **Decode is one `cross_device_reduce_1stage`, 166x per step, 8.07 ms.** It is
  bimodal: p50 6.4 us -- about right for 40 KB on NVLink -- but p90 163 us, and
  **the slowest 10% of calls carry 48% of the time.** That tail is not data
  movement, it is rank skew being absorbed at the barrier.

Both are **100% exposed**: across every step sampled, no other kernel is ever
resident while an all-reduce runs. There is no communication/compute overlap in
either phase.

### The tiered hot/cold overlap works in decode and is absent in prefill

Marlin launches split across CUDA streams:

| phase | streams | cumulative | union | overlap saves |
| --- | --- | --- | --- | --- |
| decode | 3 (106 / 100 / 100, balanced) | 15.6 ms | 12.0 ms | 3.6 ms (22.9%) |
| prefill | **1** (all 306 launches) | 775.9 ms | 775.9 ms | **0** |

Decode overlaps three balanced streams but recovers under a quarter of the
possible saving; perfect three-way overlap would be ~5.2 ms against the 12.0 ms
actually spent. Prefill serialises everything on one stream.

### Ranked candidates

1. **Decode Marlin overlap** -- ~6.8 ms/step of unrealised overlap, about 20%
   of the step. Three balanced streams already exist, so the scheduling is
   there and only the overlap is missing.
2. **Prefill all-reduce overlap** -- 177.7 ms/chunk fully exposed, ~2.1 s of a
   23.2 s TTFT. `fuse_allreduce_rms` is explicitly `false` in this launcher's
   compilation config and was never A/B'd on this fork.
3. **Decode all-reduce tail** -- ~3.9 ms/step, 12% of the step, in 17 calls.
   This is EP rank skew, so it is a load-balance problem, not a comms problem.
4. **Prefill Marlin single-stream** -- the hot/cold overlap decode benefits
   from does not exist in prefill at all, against 783 ms/chunk.
5. **Prefill attention, 33%** -- `flash_fwd_splitkv_mla_fp8_sparse` 550.7 ms,
   DSA indexer 97.6 ms, `topKPerRowPrefill` 34.2 ms per chunk.

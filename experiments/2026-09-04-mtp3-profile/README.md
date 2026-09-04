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

Pending -- job 1665068 is capturing.

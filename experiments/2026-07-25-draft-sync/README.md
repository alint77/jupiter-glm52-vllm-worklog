# Removing the per-draft-step host synchronization

P2 from the [critical-path review](../2026-07-25-c4-mtp3-critical-path/README.md),
first half: the MTP draft tail, measured at 1.57 ms/step of wall with the GPU
97% idle.

## Diagnosis

The review established *that* the host serializes between draft graphs. The
trace's host-side API timeline names the mechanism exactly. Within one steady
step:

| host t | call | duration | effect |
| ---: | --- | ---: | --- |
| 2.922 ms | `cudaGraphLaunch` (target) | 3776 us | GPU 2.952 → 58.810 |
| 8.598 ms | `cudaGraphLaunch` (draft 1) | 134 us | GPU 59.165 → 60.067 |
| **9.400 ms** | **`cudaStreamSynchronize`** | **50,692 us** | host blocked |
| 60.855 ms | `cudaGraphLaunch` (draft 2) | 106 us | GPU 60.894 → 61.589 |
| **61.530 ms** | **`cudaStreamSynchronize`** | **76 us** | host blocked |
| 62.299 ms | `cudaGraphLaunch` (draft 3) | 97 us | GPU 62.336 → 62.859 |

Each `cudaStreamSynchronize` sits inside an `aten::to` → `aten::_to_copy` →
`aten::copy_` on a **0-dimensional int tensor** — a scalar copy with a device
change. The surrounding CPU ops (`aten::to` on a `[4]` int tensor, then
`aten::floor_divide` on `[4]`) identify the caller as `get_dcp_local_seq_lens`
in `vllm/v1/attention/backends/utils.py`, reached once per draft step through
`_build_draft_attn_metadata` → `prepare_dcp_local_seq_lens`.

The offending line, on the `dcp_rank is not None` branch:

```python
rank_offsets = torch.tensor(dcp_rank, dtype=torch.int32, device=seq_lens.device)
```

`dcp_rank` is a Python int. Materializing it as a 0-d device tensor is a
**pageable** host-to-device copy, and a pageable H2D copy blocks the host until
the stream drains. `rank_offsets` is then used only as
`rank_offsets * cp_kv_cache_interleave_size` inside a `torch.clip` — a pure
scalar multiply that a Python int performs identically.

This is why the first sync costs 50.7 ms: it waits out the entire target graph.
That wall time is not itself lost (the GPU is busy), but the pattern forces the
host to re-enter the draft loop *after* each graph completes instead of running
ahead, which is what produces the 827 us and 747 us gaps with ~25 us of GPU work
in them.

The function is upstream code, not part of this branch, so the fix applies to
any DCP deployment; the tiered MoE work merely made it visible by putting it
inside a speculative draft loop.

## Fix

```python
rank_offsets = dcp_rank   # plain Python scalar; no device transfer
```

Equivalence checked exhaustively offline over `dcp_size ∈ {2,4,8}`,
`cp_kv_cache_interleave_size ∈ {1,4,64}`, on CPU and CUDA: the per-rank result
equals the corresponding column of the all-rank result, and the shards sum back
to `seq_lens`.

Unit tests: `tests/v1/worker/test_cp_utils.py`,
`tests/v1/attention/test_indexer_dcp_localize.py` (37 passed) and
`tests/v1/worker/test_gpu_autoregressive_speculator.py` (3 passed).

A scan for the same anti-pattern across `vllm/v1/attention/backends/` and
`vllm/v1/worker/gpu/` found no other instance.

## Expected effect

Removing the sync should let the host enqueue all three draft graphs while the
GPU is still executing the target graph, as it already does for the target and
the first draft. The upper bound is the 1.57 ms/step of inter-draft gaps, about
2.5% of the 62.43 ms step. It does not address the 2.73 ms eager prologue, which
is the second half of P2 and needs input-prep graph capture or async scheduling.

This is a modest, well-understood gain rather than a large speculative one —
recorded as such.

## Jobs

| Job | Config | State |
| --- | --- | --- |
| 1040935 | MTP3, c4, DCP4, 7 GB reserve | submitted |

Gates: the semantic smoke completion, the exact-400K golden SHA
`d594e4d4268600a0d9e2d51355913907c638ef0c61615547598615384dcfc528`, and the
24-prompt realistic suite at concurrency four, matched against the
[c4 MTP-depth sweep](../2026-07-23-c4-mtp-depth/README.md) MTP3 row
(179.47 tok/s warmed output, 20.79 ms mean TPOT, 228.33 tok/s effective
aggregate).

## Results

Job 1040935, Booster. Both correctness gates pass: the semantic completion is
` Paris. Distance from Paris to Lyon is` and the exact-400K continuation SHA is
`d594e4d4268600a0d9e2d51355913907c638ef0c61615547598615384dcfc528`, reproduced
by hand. Output is byte-identical, as it must be for a host-side scalar change.

### Mechanism: the host now runs a step and a half ahead

Measured GPU-side gaps between consecutive CUDA-graph replays:

| Inter-graph gap | baseline | draft-sync | change |
| --- | ---: | ---: | ---: |
| target → draft 1 | 357.7 us | 294.0 us | −64 us |
| draft → draft (x2) | 786.1 us | **70.3 us** | **−716 us each** |
| draft 3 → next target | 2,616.8 us | **172.8 us** | **−2,444 us** |
| **total per step** | **4,546.7 us** | **607.4 us** | **−3,939 us** |

| | baseline | draft-sync |
| --- | ---: | ---: |
| host lead (gpu_start − host_launch), median | 37.5 us | **104.9 ms** |

The host went from launching each graph just-in-time to running a full step and
a half ahead. That is the predicted mechanism, and it means **this one line also
fixed P2's second half**: the 2.73 ms eager prologue is no longer on the
critical path, because prep for step *n*+1 now overlaps step *n*'s GPU work.
The "draft 3 → next target" gap, which contains the prologue, collapsed from
2.62 ms to 0.17 ms.

Recovered 3.94 ms/step against a 62.43 ms step — 6.3%, versus the 1.57 ms
(2.5%) predicted from the draft tail alone.

### Throughput

| Case | metric | baseline | draft-sync | change |
| --- | --- | ---: | ---: | ---: |
| Realistic c4, warmed | output tok/s | 179.47 | **185.05** | **+3.1%** |
| | median ITL | 50.55 ms | 48.19 ms | −4.7% |
| Realistic c4, first use | output tok/s | 107.08 | **112.44** | **+5.0%** |
| | mean TPOT | 21.69 ms | 20.76 ms | −4.3% |
| 4K c4 r1 | output tok/s | 94.00 | **99.13** | **+5.5%** |
| | median ITL | 61.03 ms | 56.13 ms | −8.0% |
| 4K c4 r2 | output tok/s | 103.08 | **110.25** | **+7.0%** |
| | mean TPOT | 23.30 ms | 21.24 ms | −8.8% |
| 396K c4 | median ITL | 60.10 ms | **58.13 ms** | **−3.3%** |
| Exact 400K c1 | median ITL | 38.33 ms | 35.91 ms | −6.3% |

Two caveats on reading these.

**396K c4 acceptance is not comparable run to run.** This run reports 2.78
against the sweep's 2.66. Acceptance at concurrency four is not bit-deterministic
because batch composition varies with arrival timing and changes kernel
reduction shapes; the exact-400K *c1* SHA gate is the determinism check, and it
passes. Holding acceptance at the baseline value the 396K effective aggregate is
177.36 → 183.39 tok/s (+3.4%); using each run's own acceptance it reads 191.3
(+7.9%). The median-ITL delta of −3.3% is the clean attributable number.

**`mean_tpot_ms` on exact-400K c1 reads +6.65%** while both mean and median ITL
improve ~6%. With one request and a ~110 s prefill dominating, TPOT is not a
decode signal there.

## Status

P2 is complete, both halves, from one line. The remaining GPU idle is 0.61 ms of
inter-graph gap plus the ~3.8 ms of intra-target-graph node scheduling, which is
P4 (graph node count) territory.

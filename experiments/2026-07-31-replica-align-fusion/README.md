# Fuse replica assignment with dual-tier Marlin alignment

Status: **plan only**.

## Goal

Remove the 75 standalone replica-scheduler kernels from each c4 target graph
without changing the greedy policy or the hot/cold Marlin execution.

The narrow design is one tiered alignment operation after grouped top-k and
before the hot/cold streams split. It will:

1. count the current logical routes once;
2. run the existing deterministic replica assignment;
3. build the hot and cold expert maps;
4. produce the padded/sorted routing metadata for both Marlin tiers.

Both Marlin calls then consume that prepared metadata and skip their current
per-tier `moe_align_block_size` calls.

## Why this boundary

The current per-layer device order is:

```text
grouped top-k
  -> replica scheduler
  -> hot align + sort    \
                         -> hot/cold Marlin overlap
  -> cold align + sort   /
```

The paired c4 trace measures, per rank-step:

| Work | Calls | Cumulative device time |
| --- | ---: | ---: |
| Replica scheduler | 75 | 0.699 ms |
| Align kernels | 150 | 0.472 ms |
| Alignment sort kernels | 150 | 0.359 ms |
| Total scheduling/alignment | 375 | 1.530 ms |

`moe_align_block_size` already scans every selected route and builds expert
counts. The scheduler performs a separate scan immediately before it. A common
dual-tier alignment can reuse those counts and reduce 375 graph nodes to at
most 150, ideally 75.

The grouped-top-k router is not the first target. Its CUDA implementation uses
one block per token, while assignment needs a globally complete route set.
Fusing there would require a cooperative-grid barrier or a global completion
protocol in a generic router kernel. That is more invasive and easier to get
wrong than a tiered-specific alignment operation.

## Performance bound

The existing greedy scheduler already raises c4 throughput from 148.14 to
158.12 output tok/s. Fusion cannot recover more than the scheduler and
alignment portion of the graph, so the expected additional end-to-end gain is
small: roughly 1–2%, not another large step.

This experiment is worthwhile only if it removes most of the 0.699 ms
scheduler cost without harming Marlin overlap. It should stop early if the
extra code produces less than a measurable 0.5% serving gain.

## Proposed design

### One shared operation before stream split

Add a tiered-only alignment path with fixed-shape inputs and outputs.

Inputs:

- logical `topk_ids`;
- primary and secondary rank tables;
- primary HBM residency;
- primary hot/cold and physical cold maps;
- EP rank/size and the existing HBM/Grace cost constants;
- independently selected hot and cold Marlin block sizes.

Outputs:

- writable hot and cold global-to-local maps;
- hot and cold `sorted_token_ids`;
- hot and cold physical `expert_ids`;
- hot and cold `num_tokens_post_padded`;
- an optional selected-rank table retained for tests and diagnostics.

The operation runs before `aux_stream()` is entered. When it completes, the
hot and cold Marlin branches can launch on their existing streams with no
cross-stream map race.

### Kernel phases

For the decode shapes, one cooperative thread block should be sufficient:

1. Count valid routes for all 256 logical experts.
2. Account fixed non-replicated tasks by rank and tier.
3. order active replicated tasks with the exact current key:
   residence cost, route count, then expert ID;
4. choose the primary or secondary with the exact current
   `(max rank time, sum rank time, rank ID)` tie-break;
5. write distinct hot and cold maps;
6. compute independent padded prefix sums for the two tiers;
7. fill both sorted-token and physical-expert outputs.

If filling both sorted buffers in the same block is slower, permit one second
fixed kernel. Do not reintroduce one scheduler plus two ordinary alignment
pipelines.

### Marlin plumbing

Factor Marlin's block-size choice from `fused_marlin_moe` and allow the tiered
path to pass pre-aligned routing metadata. The ordinary Marlin path and all
non-tiered callers keep the existing behavior.

The shared operation remains upstream of the two Marlin streams, as the
current standalone scheduler is today. Hot/cold GEMM overlap must therefore be
unchanged after alignment finishes.

### Prefill and fallback

For shapes above the decode scheduling limit, the same operation writes the
primary maps and constructs the two ordinary primary-tier alignments. It does
not run greedy assignment.

Keep the current standalone scheduler as a control and fallback until the new
path passes. The fused path must fail closed for unsupported expert counts,
EP layouts, quantization backends, LoRA, or non-Marlin execution.

## Experimental phases

### Phase 0 — freeze the baseline

Use the existing c4 traces to record:

- scheduler, align, and sort kernel durations and counts;
- the gap from grouped top-k through first Marlin launch;
- hot/cold first-Marlin start skew;
- full graph cycle and span;
- custom all-reduce residency and layer rank skew.

Re-run the extractor on the same traces after any analysis-script change. The
baseline is 75 scheduler, 150 align, and 150 sort launches per rank-step.

### Phase 1 — isolated dual-alignment prototype

Build the smallest fixed-shape CUDA prototype outside the model path. Test:

- c1/MTP3: 32 routes;
- c4/MTP3: 128 routes;
- the saved 985-copy placement;
- route sets from the real Claude/agentic captures;
- random and adversarial route sets only as supplementary coverage.

Compare current:

```text
scheduler + hot align/sort + cold align/sort
```

against:

```text
dual-tier assignment/alignment
```

Measure graph replay time, individual kernel duration, node count, and output
metadata. Do not start model integration unless the prototype is faster.

### Phase 2 — correctness-first runtime integration

Add a temporary opt-in fused assignment mode so the current `greedy` path
remains an in-process control.

Required checks:

- selected-rank tables exactly match the current GPU scheduler;
- hot/cold maps exactly match for every rank;
- every valid route occurs in exactly one rank and one tier;
- every selected rank actually stores that expert;
- padded counts and expert blocks satisfy Marlin's alignment contract;
- prefill restores primary ownership;
- CUDA graph capture and replay are stable;
- disabling fusion is bitwise identical to the current path.

Compare fused Marlin output against current greedy execution using the same
inputs and tolerances already used by the launch-policy tests.

### Phase 3 — isolated production-shape measurement

On GH200, measure at least 1,000 graph replays for c1 and c4. Report:

- total assignment/alignment span;
- kernel-node count;
- hot and cold alignment completion times;
- first-Marlin launch times on both streams;
- graph memory delta.

Gate:

- zero standalone scheduler kernels;
- no more than two dual-alignment kernels per routed layer;
- scheduling/alignment cumulative device time below **0.90 ms/c4 rank-step**;
- no increase in the grouped-top-k-to-first-Marlin critical span;
- no loss of hot/cold Marlin overlap.

### Phase 4 — parallel end-to-end A/B

Submit these arms concurrently on separate Booster nodes:

1. current greedy, c1;
2. fused greedy, c1;
3. current greedy, c4/DCP4;
4. fused greedy, c4/DCP4;
5. current greedy c4 profile;
6. fused greedy c4 profile.

Use the established AutoRound W4G64, MTP3, 985-copy profile and realistic
16-prompt Python/PyTorch/CUDA/ML/math/email suite. Run two unprofiled repeats
with 256 output tokens. Monitor per-rank VRAM and MTP acceptance.

Primary comparisons:

- output tok/s and TPOT;
- MTP acceptance and request failures;
- scheduler/alignment node census;
- graph cycle/span;
- layer rank skew and custom all-reduce residency;
- routed Marlin cumulative time and overlap;
- peak HBM headroom.

## Success and stop criteria

Accept the fused path only if:

- all exactly-once and graph-replay checks pass;
- c4 throughput improves by at least **1.0%** over current greedy;
- c1 does not regress by more than **0.5%**;
- rank-skew and custom all-reduce improvements remain within 5% of current
  greedy;
- peak HBM use grows by no more than 128 MiB/rank;
- no generic router, Marlin, or non-tiered behavior changes.

Stop and retain the current scheduler if:

- the isolated dual alignment is not faster;
- the common alignment delays the cold Marlin enough to reduce overlap;
- numerical differences exceed the current Marlin tolerance;
- the implementation requires a new collective, CPU synchronization, dynamic
  allocation, or a general routing framework;
- the measured serving gain is below 0.5%.

## Expected implementation surface

Keep the experiment local to:

- `csrc/libtorch_stable/moe/moe_align_sum_kernels.cu`;
- the stable MoE binding and `_custom_ops.py`;
- `moe_align_block_size.py`;
- Marlin's optional pre-aligned metadata path;
- `tiered_moe_execution.py` and the tiered scheduler/reference helpers;
- existing tiered-MoE and Marlin tests.

Do not modify generic grouped-top-k routing for this experiment.

## Follow-up boundary

If fusion passes, the next separate experiment is cost-model calibration using
the observed 3.972 ms increase in cumulative routed Marlin residency. Do not
mix that policy change into this A/B: this experiment changes execution
mechanics only, so any performance delta is attributable to fusion.

# MTP prompt mix and decode profile

## Prompt mix

Six fixed categories use three prompts each, temperature zero, concurrency one,
and a maximum of 256 output tokens. The normal GLM chat template is applied.
All 18 requests completed and produced 256 tokens.

| Category | TTFT | TPOT | Decode | Acceptance | Tokens/step |
| --- | ---: | ---: | ---: | ---: | ---: |
| Python | 254 ms | 11.027 ms | 90.69 tok/s | 64.01% | 2.92 |
| PyTorch | 267 ms | 11.757 ms | 85.05 tok/s | 58.21% | 2.75 |
| CUDA C++ | 249 ms | 10.390 ms | 96.25 tok/s | 71.56% | 3.15 |
| Math | 267 ms | 9.738 ms | 102.69 tok/s | 77.63% | 3.33 |
| Email | 264 ms | 10.223 ms | 97.82 tok/s | 70.30% | 3.11 |
| Explanation | 248 ms | 10.790 ms | 92.68 tok/s | 66.15% | 2.98 |

Across 4,608 output tokens, weighted draft acceptance is 67.61%, effective
acceptance length is 3.03 tokens per target step, and weighted TPOT is 10.654
ms (93.86 tok/s). Math is the easiest domain for this MTP head and PyTorch is
the hardest in this small deterministic sample. Acceptance explains most of
the category-level decode spread (Pearson `r=0.990`).

The first Python pass encountered a one-time short-prefill compile. The table
uses the immediate warm rerun; its acceptance changed from 63.02% to 64.01%
because the benchmark dataset shuffles request order, while steady TPOT stayed
effectively unchanged (11.079 versus 11.027 ms).

## Profile method

The profiler-enabled MTP3 server uses the qualified TP4/EP4, reserve-10,
size-4 CUDA-graph configuration. A 399,744-input/256-output request skips its
50 chunked-prefill iterations and records eight decode iterations on all four
ranks. Tensor shapes are enabled; Python stacks and memory profiling are
disabled to bound overhead. Profiled latency is not used as a performance
result.

The profiled request accepted 63.98% of drafts (2.92 tokens/target step). Its
19.13 ms TPOT includes profiler overhead and is not compared with the qualified
9.245 ms result.

Average GPU kernel activity per target step was:

| Category | Time/step | Evidence |
| --- | ---: | --- |
| Routed W4 MoE | 8.48 ms | 300 Marlin calls/step; Phase 8 was 8.44 ms |
| Custom all-reduce | 4.99 ms | 166 calls/step |
| Dense W4 GEMM | 4.65 ms | 312 Machete calls/step |
| Sparse MLA split + combine | 1.93 ms | 81 split calls/step |
| DSA logits + top-k | 1.61 ms | 24 calls/step |
| Vocabulary projection | 0.59 ms | one batch-4 plus three batch-1 calls/step |
| MTP FP8 dense/MoE/scale core | about 0.55 ms | 24 FP8 MTP forwards |

These categories overlap across streams and are not additive wall time. The
important result is that routed target MoE time per target step is unchanged
while each step now verifies four tokens. The new FP8 MTP core is small; it is
not the first component to optimize.

Rank 2 spent 72.42 ms in routed Marlin across the trace versus 63.22 ms on rank
0, a 14.6% spread. Custom-all-reduce time moves in the opposite direction
(35.61 versus 44.04 ms, rank correlation `r=-0.988`), showing that much of the
apparent collective cost is ranks waiting for unequal expert work. The current
ownership/heat profile was captured without MTP verification traffic.

The shape trace also finds two smaller MTP inefficiencies:

- `index_share_for_mtp_iteration` is ineffective inside the captured draft
  graphs. There are 24 indexer calls per target step: 21 target calls plus all
  three MTP calls. The 16 batch-1 top-k calls over eight steps are precisely
  draft passes two and three, which should have reused pass one's indices.
- Every step performs four `6144 x 38720` per-rank vocabulary projections and
  four full-vocabulary all-gathers. The target batch-4 GEMM costs 145 us, while
  each of the three serial batch-1 draft GEMMs costs 147 us.

## Roofline analysis

The [complete forward-pass roofline](roofline-analysis.md) inventories all 91
grouped CUDA kernel/copy names and models their FLOPs, logical traffic, and
applicable GH200 roof. It identifies two additional first-order losses: the
size-4 MTP verification path serializes hot-HBM and cold-Grace expert calls
(2.18 ms/step ideal kernel-overlap bound), and sparse MLA pads 16 real local
heads to 64 (75% wasted head work). The raw tables and plots are in
[`roofline-kernels.csv`](roofline-kernels.csv),
[`roofline-summary.json`](roofline-summary.json), and
[`roofline.svg`](roofline.svg).

## Optimization order

1. Capture routing under MTP and rebuild owner/hot placement for verification
   traffic. This is the highest-confidence opportunity because imbalance is
   directly visible across ranks.
2. Benchmark sequence-parallel MoE only for uniform size-4 verification. The
   old batch-1 path was bad because it padded one token to TP4; MTP verification
   already supplies exactly four tokens and may benefit from one token/rank.
3. After balancing, benchmark the all-reduce backends for `[4, 6144]` and retry
   `fuse_allreduce_rms`. Custom all-reduce is the largest synchronized kernel
   category, but its current duration includes expert-load wait.
4. Capture separate indexer and index-reuse draft graphs, or make the skip a
   graph-visible runtime input. Removing passes two and three saves two DSA
   index/top-k paths per target step.
5. Add `get_top_tokens` to DeepSeek MTP and enable local-argmax reduction for
   greedy drafting. This removes three full-vocabulary gathers and their logits
   materialization, though it does not remove the three local head GEMMs.
6. Sweep MTP2 versus MTP3, especially at 400K and for PyTorch prompts. Third
   position acceptance is only 35.6% at 400K and 38.2% for PyTorch, versus
   63.2% for math, so a fixed depth is unlikely to be optimal for every case.

Startup warmup should also cover padded draft preparation, rejection sampling,
and exact chunk metadata: five MTP Triton kernels and one long-prefill metadata
kernel JIT-compiled on the first requests. This affects cold latency, not the
steady decode profile.

## Source branch

The cumulative tiered-Grace and MTP vLLM implementation is pushed to
`alint77/vllm`, branch `tiered-moe-grace-mtp`, at commit `a66535e59`.
Ruff, the complete pre-commit hook set, and 85 focused tests pass.

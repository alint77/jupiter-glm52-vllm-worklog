# Marlin MoE launch tuning for small-M verify batches

P1 from the [critical-path review](../2026-07-25-c4-mtp3-critical-path/README.md).

## Why

Routed W4 Marlin is 20.15 ms of the 62.43 ms MTP3 c4 critical path — the single
largest item — and runs at only 29–38% of HBM peak. The trace shows why this is
worth probing before writing any new kernel: **all 300 Marlin MoE launches per
step use an identical geometry**, grid `(396,1,1)` = 3 blocks/SM × 132 SMs,
76,458 B shared memory, *regardless of whether the verify batch is 8, 12 or 16
tokens*. That is exactly one full occupancy wave, shared-memory limited.

Reading `csrc/libtorch_stable/moe/marlin_moe_wna16/ops.cu`, the block count is
`blocks = sms * exec_cfg.blocks_per_sm`, and `blocks_per_sm` comes from a
heuristic (`determine_exec_config`) driven by register and shared-memory budgets
with a cap of 4 for `thread_m_blocks == 1`. The op exposes `blocks_per_sm`,
`thread_n` and `thread_k` as parameters defaulting to `-1` (auto), and
`marlin_moe.py` never passes them. The workspace is already allocated for up to
4 blocks/SM, so the whole range is sweepable without touching the kernel.

Two hypotheses:

1. **Single tier.** The auto heuristic is tuned for prefill-shaped GEMMs. At
   M ≤ 32 a different `blocks_per_sm` / thread config may do better. Per-expert
   cost scales both the Marlin span *and* the all-reduce skew wait (the skew is
   an order statistic multiplying that cost), so a 25% kernel win is worth
   ~6.5 ms of Marlin plus ~1.1 ms of skew — about 12% of the step.
2. **Two tiers.** Hot and cold each request a full wave, so they cannot
   co-reside; the second tier's blocks only start as the first tier's retire.
   This is the measured cause of the 3.6× dilation of cold `w2` and the
   unphysical cold w13/w2 ratio of 0.56 against a 2.0 byte ratio. Splitting the
   SM budget — e.g. cold at 1 block/SM, hot at 2 — may let them genuinely
   overlap HBM and C2C traffic.

## Method

`benchmarks/kernels/benchmark_moe_wna16_marlin_decode.py`, at GLM-5.2 geometry
(H=6144, I=2048, group 128, uint4b8, top-8, 64 local experts under EP4).

- `single`: sweeps `(blocks_per_sm, thread_n, thread_k)` for `w13` and `w2`,
  over M ∈ {8,12,16,32} and 5/16/22 activated experts, reporting achieved
  bandwidth against the 3.5 TB/s roof.
- `overlap`: hot tier in HBM, cold tier in pinned Grace memory reached through
  `GraceAllocation.allocate_pinned` — the same pinned-UVA alias the tiered
  loader uses — with NUMA page auditing, sweeping the SM split across the two
  streams and reporting the union span against hot/cold solo times.

Activated-expert counts of 19 hot / 3 cold match the trace-derived operating
point. The job runs a tiny smoke configuration first so API errors surface
before the real sweep.

Run on a Booster node with `numactl` binding to the Grace node paired with
GPU 0. Login-node C2C and clock behaviour do not transfer, so no measurement
here is taken on the login node.

```bash
sbatch --job-name=marlin-decode-tune \
  agent_space/experiments/2026-07-25-marlin-decode-tuning/job.sh
```

## Jobs

| Job | Purpose | State |
| --- | --- | --- |
| 1040904 | smoke + full sweep | complete |

Booster node `jpbo-074-26`, 132 SMs, GH200 480GB. Grace page auditing reported
100% of sampled cold pages on the paired NUMA node, so the C2C path is real.

## Result: both hypotheses refuted

**Launch configuration does nothing.** Mean time relative to the auto heuristic
across all 24 single-tier points:

| Config | mean relative time |
| --- | ---: |
| `blocks_per_sm=3` | 0.9999x |
| auto (`-1`) | 1.0000x |
| `blocks_per_sm=4` | 1.0012x |
| `blocks_per_sm=1` | 1.0029x |
| `blocks_per_sm=2` | 1.0037x |
| `thread_n=128, thread_k=128` | 1.1331x |
| `blocks_per_sm=2, 128, 128` | 1.2673x |

Every `blocks_per_sm` value lands within 0.4% of auto; explicit thread configs
are 13–27% *worse*. At the trace operating point (M=16, 16–22 activated experts)
the best config beats auto by 0.0–1.8%. The apparent 12–35% wins all sit at
5 activated experts or M=32, neither of which is the operating point. The
heuristic in `determine_exec_config` is already choosing well.

**Splitting the SM budget across tiers does nothing.** Best union span versus
the auto/auto pair: +0.0% at M=8, 0.3% at M=12, 0.4% at M=16, 0.3% at M=32.
The full 4×4 sweep of (hot, cold) block counts is flat. Whatever prevents the
two tiers from overlapping, it is not the threadblock budget.

**The hot kernel is not the problem.** Isolated at realistic shapes it reaches
**53–59% of HBM peak**, or 2.18–2.25 TB/s:

| activated | w13 | w2 | layer total |
| ---: | ---: | ---: | ---: |
| 16 | 103.8 us (57%) | 50.7 us (58%) | 154.5 us |
| 22 | 139.1 us (59%) | 77.3 us (53%) | 216.4 us |

The critical-path review's 29–38% estimate was too pessimistic: it divided the
*in-situ* hot time by assumed byte counts and so charged cross-stream contention
to the kernel. In-situ hot is 320.1 us/layer against 216.4 us isolated at 22
activated experts — **1.48x**. Roughly a third of the 24.5 ms hot chain, about
7.8 ms/step, is contention with the cold stream rather than kernel inefficiency.
The SM-split result says that contention is not recoverable through the launch
configuration.

## Unplanned result: the cold tier degrades with batch size, the hot tier does not

Weight bytes are held constant (19 hot experts, 3 cold) while token count varies:

| M | hot | hot GB/s | cold | cold GB/s | hot vs M=8 | cold vs M=8 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 113.0 us | 2,181 | 195.7 us | 199 | 1.00x | 1.00x |
| 12 | 109.4 us | 2,253 | 230.5 us | 169 | 0.97x | 1.18x |
| 16 | 110.5 us | 2,230 | 326.7 us | 119 | 0.98x | **1.67x** |
| 32 | 142.3 us | 1,733 | 565.9 us | 69 | 1.26x | **2.89x** |

The hot tier is essentially flat to M=16 — re-visits to the same expert weights
hit L2. The cold tier grows 2.89x for the same bytes, and its effective C2C
bandwidth collapses from 199 to 69 GB/s, because C2C reads are not cached and
Marlin re-streams each expert tile per token block.

This matters for the deployment target. Raising concurrency multiplies exactly
the traffic the cold tier handles worst, so the c=4 → c=8 extrapolation in the
critical-path review (drawn from MTP depth, which varies positions rather than
sequences) is optimistic wherever cold residency is non-trivial. It also
reinforces the review's Finding 3: keep cold residency small, and spend HBM on
KV rather than on either tier.

## Open discrepancy

Isolated cold bandwidth here (199 GB/s at M=8, falling to 69) is well under the
421 GB/s C2C figure, and disagrees with the
[2026-07-17 Marlin/UVA probe](../2026-07-17-marlin-uva/README.md), which
measured full-footprint Grace-UVA execution within 2% of HBM. The critical-path
review's Finding 2 derived "2.85 activated cold experts per layer" by *assuming*
cold w13 runs at the C2C roof; that assumption is now doubtful, so the absolute
number should not be relied on. The qualitative conclusion — cold is a minority
of the routed work and the hot tier owns the critical path — is unaffected,
since it rests on the measured w13/w2 ratios and the layer-span totals.

Resolving this is the prerequisite for any further tiering work: either the
earlier probe used a shape that streams differently, or this benchmark's cold
path differs from the production one. Compare against
`agent_space/benchmarks/marlin_grace_uva.py` before trusting either number.

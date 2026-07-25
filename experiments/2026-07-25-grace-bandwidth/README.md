# Resolving the Grace C2C bandwidth discrepancy

Three measurements of the cold tier disagreed by more than 5x:

| Source | Claim |
| --- | --- |
| [Critical-path review](../2026-07-25-c4-mtp3-critical-path/README.md) Finding 2 | cold w13 at the 421 GB/s C2C roof (assumed, to derive expert counts) |
| [2026-07-17 Marlin/UVA probe](../2026-07-17-marlin-uva/README.md) | Grace UVA within 2–4% of HBM |
| [Marlin decode tuning](../2026-07-25-marlin-decode-tuning/README.md) | 69–199 GB/s, collapsing with batch size |

Job 1040910 on Booster node `jpbo-036-21`, Grace pages 100% NUMA-local.
(Job 1040908 died first: `moe_block_size` 32 raises an async
`AcceleratorError` for this shape, which poisons the CUDA context and cannot be
caught. Block sizes other than 16 are not usable here — worth revisiting
separately, but at production routing every expert gets one block anyway.)

## Answer: there is no anomaly. C2C runs at 88–95% of its roof

The decode-tuning benchmark was wrong, and the bug was mine. Its `routing()`
handed the cold tier's three experts the *entire* routing — 43 assignments each,
so three 16-token blocks per expert. Production splits **one** routing across
both tiers via per-tier `expert_map` (`moe_align_block_size(..., expert_map,
ignore_invalid_experts=True)`), so a cold expert receives its true minority
share and gets **one** block.

That matters because Marlin re-reads an expert's full weight tile **once per
token block**. HBM absorbs the repeats in L2; C2C has no such cache. Dividing
*logical* weight bytes by a time that streamed them three times produces an
apparent bandwidth collapse that is pure accounting.

Counting physical (re-streamed) bytes instead, with production routing:

| cold share | blocks | physical | time | achieved | % of 421 GB/s roof |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.05 | 3 | 38.9 MB | 104.3 us | 372.8 GB/s | 88.6% |
| 0.13 | 3 | 38.9 MB | 104.4 us | 373.0 GB/s | **88.6%** |
| 0.25 | 3 | 38.9 MB | 104.9 us | 370.4 GB/s | 88.0% |
| 0.50 | **6** | 77.9 MB | **194.5 us** | 400.4 GB/s | **95.1%** |

The 0.50 row is the control: doubling the routing mass doubles the blocks *and*
doubles the time, while achieved bandwidth stays at the roof. The model is
confirmed and the C2C path is healthy.

**Cold does not degrade with batch size.** At production share, cold w13 is flat:
103.6 us at M=8, 104.4 at M=16, 105.1 at M=32. The 2.89x growth reported in the
decode-tuning experiment was the same artifact and is withdrawn, along with the
concurrency-scaling warning drawn from it.

The hot tier is steady at 61–62% of the HBM roof (2.14–2.18 TB/s) across the
same range.

## Why the 2026-07-17 probe said "within 2–4% of HBM"

Re-run here for comparison, unchanged (M=1, dense Marlin, 8 experts,
6,144 x 4,096):

| Source | us/expert | achieved | % of its own roof |
| --- | ---: | ---: | ---: |
| HBM | 35.50 | 365.5 GB/s | **10.4%** |
| pinned Grace UVA | 43.29 | 299.7 GB/s | 71.2% |

At M=1 the dense kernel drives HBM at a tenth of its capability. Grace looked
competitive because **HBM was not being used**, not because C2C approaches HBM
bandwidth. The original ratio was real; the inference from it was not. Any
future tiering claim must report achieved bandwidth against each path's own
roof, not a ratio between two latency-bound measurements.

## Consequence: Finding 2's assumption is restored

Finding 2 derived "2.85 activated cold experts per layer" by assuming cold w13
runs at the C2C roof. That assumption is now measured at 88–95%, so the figure
stands (scaled by ~1/0.9 it becomes ~3.2). The retraction issued in
[the decode-tuning report](../2026-07-25-marlin-decode-tuning/README.md) is
itself withdrawn.

## New finding: the two tiers barely overlap, and that is the largest lever found

Measured directly at production share, M=16, w13, hot in HBM and cold in Grace:

| Quantity | us |
| --- | ---: |
| hot alone | 115.3 |
| cold alone | 104.4 |
| **union (both streams)** | **194.0** |
| serial (hot + cold) | 219.7 |
| ideal (max) | 115.3 |

Overlap saves 25.7 us of a possible 104.4 — it captures **24.6% of the
available overlap**, and the union sits **68% above ideal**. At cold share 0.50
it is 41% above ideal. The two tiers are running much closer to serial than to
concurrent.

This reframes the routed MoE budget. Per layer at M=16, isolated: hot
w13 115.4 + w2 60.9 = 176.3 us; ideal overlapped layer is therefore 176.3 us,
or **13.2 ms** across 75 layers. Serial would be 25.1 ms. The trace's measured
layer-span total is **25.7 ms** — matching the *serial* estimate, not the ideal.

So the "39–40% overlap saving" reported both by the critical-path review and by
the earlier SOL analysis is an artifact: both compared the layer span against a
sum of *dilated* kernel durations, which double-counts the dilation. Against
isolated costs there is almost no overlap benefit.

**Roughly 12 ms/step — about 19% of the 62.43 ms step — is available if the hot
and cold tiers actually overlapped.** That is now the largest single lever
identified anywhere in this analysis, and unlike the three refuted hypotheses it
rests on direct measurement of both the isolated and combined cases.

`blocks_per_sm` does not unlock it (swept, 0.0–0.4%). The leading hypothesis is
that both kernels launch one full occupancy wave and each block holds its SM
while stalled on memory, so a C2C-bound cold block occupies an SM without
issuing work. Next probes, cheapest first:

1. Launch cold with a genuinely small grid (a fraction of one wave) rather than
   `sms * blocks_per_sm`, which never goes below 132 blocks.
2. Give the cold stream a lower priority (`torch.cuda.Stream(priority=...)`) so
   the scheduler backfills hot blocks.
3. Check whether the two kernels are co-resident at all, with a profile of this
   isolated two-stream case — if CUDA is serialising them outright, the fix is
   different from any occupancy tuning.

## Reproduce

```bash
sbatch --job-name=grace-bw agent_space/experiments/2026-07-25-grace-bandwidth/job.sh
```

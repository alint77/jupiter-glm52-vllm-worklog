# JUPITER GLM-5.2 vLLM worklog

Experimental worklog for serving the 361.06 GiB
`lowbitcoffee/GLM-5.2-W4A16` checkpoint on one JUPITER Booster node. The goal
is to improve batch-one decode at up to 400K context by keeping hot MoE experts
in HBM and executing colder experts from coherent Grace memory through CUDA
UVA. Native vLLM CPU offload decodes this model at 37.57 tok/s. The project
minimum of 100 tok/s was passed at Phase 9; the sections below record where the
qualified path stands now.

## Current state

Last indexed 2026-09-02, covering source commit `567bdc9721` on
`dflash2-backport`, through Phase 51. **Phases 48-50 are written up in
[the DFlash2 upstream audit](experiments/2026-08-30-dflash2-upstream-audit/README.md)
and [the tiered-off control](experiments/2026-09-02-dflash2-tiered-off/README.md)
rather than here.**

**Phase 51 overturns Phase 48's verdict.** Stock upstream sglang, on this
hardware with incoai's own published checkpoint pair, reaches **5.71** GSM8K
acceptance where this fork reaches **3.995** -- 96% of the card's 5.94 against
our 67%. MTP7 agrees across the two engines to 0.29% (4.9348 fork / 4.9489
sglang), so the divergence is DFlash2-specific: **there is a defect in the
fork's DFlash2 port worth ~43% acceptance, and a working reference to diff
against.** Phase 48's "there are no fork defects left that we can find" is
withdrawn. See
[the sglang control](experiments/2026-09-02-sglang-dflash2-control/README.md). Phases 40 and 41 remain reserved for the
two 2026-08-28 experiment directories that are on disk but not yet written up:
`2026-08-28-mtp-acceptance-zero` and `2026-08-28-decode-comms`. The qualified
production branch is unchanged at `cec73c66b3` (`known-good-1238882`).

**The served model is now GLM-5.3**, on the `JANGQ-AI/GLM-5.3-W4A16` int4
group-32 checkpoint, with the hot-expert ranking re-derived for 5.3 (44) and
2496 hot slots per rank (46). GLM-5.2 AutoRound W4G64 remains the reference
target for quality comparisons; 5.3 scores about 4pp lower on GSM8K for reasons
Phase 43 traces to post-training rather than to anything in this stack.

| | |
| --- | --- |
| Qualified default | MTP3, TP4/EP4, hot/cold Marlin overlap under the tight shared-memory launch policy, exact replica assignment, no sequence parallelism, no local draft argmax |
| Interactive | c1/DCP1, V1 model runner, HBM main KV cache, 400K context |
| Agent swarm | c4/DCP4, V2 model runner, per-sequence KV, 11 GB planned HBM reserve |
| Claude Code host | GLM-5.3 W4A16 g32, c4/DCP4/MTP3, 2496 hot slots, `gpu-memory-utilization` 0.94, KV pinned at 21,689,598,771 bytes, 6 GB tiered reserve (46) |
| Routing capture | forced to c1/DCP1 and the V1 runner; routed-expert return exists on neither the V2 runner nor DCP (47) |
| Correctness gates | greedy exact-text smoke; invariant, census and tolerance checks for kernel changes; the exact-400K golden SHA, which must be re-established after any Marlin grid change |

Headline measurements. Each is against its own matched control, and the harness
matters more than the number: do not compare rows.

| Harness | Measurement | Phase |
| --- | --- | --- |
| Batch-one 4K/256, native CPU offload | 37.06 tok/s decode | baseline |
| Batch-one exact 399,744+256, native CPU offload | 37.57 tok/s decode | baseline |
| Batch-one exact 399,744+256, tiered MTP3 + overlap | 123.65-127.67 tok/s decode | 12, 13 |
| Warmed 24-prompt realistic suite, c1/q4 | 108.08 tok/s decode, 98.03 end to end | 17 |
| 18-prompt mixed-domain suite, c4 | 18.687 ms TPOT, 181.60 tok/s aggregate | 27 |
| No-MTP acceptance-free step time, c1/c2/c4 | shared-memory fix worth 5.3-6.7% | 26 |
| Acceptance-free batch 4 / MTP3 batch 16, same node | replica assignment worth 5.96% / 6.50% of step time | 31 |
| 16K-in / 512-out PyTorch coding, c1 | 101.3 tok/s decode, 50.16 end to end | 33 |
| 16K-in / 512-out PyTorch coding, c4 | 265.2 tok/s decode aggregate, 70.98 end to end | 33 |
| Real-code decode suite, c4/DCP4/MTP3, GLM-5.3 NVFP4 | 194.2 +/- 4.6 tok/s aggregate, 20.60 ms TPOT | 46 |
| Real-code decode suite, c4/DCP4/MTP3, GLM-5.3 W4A16 g32 | **213.1 +/- 1.4 tok/s aggregate, 18.77 ms TPOT** | 46 |
| Same suite, three W4A16 *rankings* (mislabelled a hot-slot sweep) | 213.1 / 214.8 / 205.2 tok/s; residency was identical in all three, see 46 | 46 |

The Phase 27 aggregate is the most recent full-suite number, but it predates
both the replica work and its own retracted per-domain table; no post-replica
measurement on that suite exists yet.

Lever status, so that settled questions are not reopened:

| Lever | Status |
| --- | --- |
| Marlin shared-memory monopoly | Fixed (26); worth 5.3-8.1% of step time |
| Exact replica assignment | Shipped (31); worth 5-6.5% of step time at c4 |
| Cold-tier kernel, tile, split-K and occupancy tuning | Retired (26, 32): the cold tier runs at the C2C roof |
| Prefill Marlin tile | Shipped (35) behind `VLLM_TIERED_MOE_PREFILL_TILE`, default on; ~2.2% of c4 wall clock, **kernel-measured only** |
| Prefill shared-memory policy and CTAs/SM | Refuted (35): worth 0.3%; the grid oversubscribes the machine eightfold |
| Prefill collective bandwidth, NCCL protocol, chunk size | Refuted (36): the median all-reduce is already the hardware optimum |
| Ungating replica assignment for prefill | Refuted (36): the objective is count-based and degenerate at 8,192 tokens |
| KV cache on Grace over C2C | Refuted (38, 39): +8.5 ms/step at c4 against a ~9.4 ms residency gain, and the kernel is occupancy-bound |
| Per-layer hot/cold rebalance | Refuted (32): every layer is already cold-bound |
| Green contexts for SM isolation | Refuted (30): 9-40% worse than the production fork/join |
| `blocks_per_sm` and SM-budget partitioning | Refuted (21, 23, 26) |
| Target sequence parallelism | Refuted (13) |
| Wider or deeper speculation: MTP6, DSpark, DFlash1 | Refuted (11, 28): throughput is inverse to verify-batch width |
| **DFlash2 port correctness** | **Open defect (51)**: upstream sglang gets 5.7236/5.7063 (two draft-attention backends, 0.30% apart) where this fork gets 3.9951, same checkpoint pair, same hardware, same protocol. MTP7 agrees across stacks to 0.29%, so it is DFlash2-specific. Phase 49f localised the fork's deficit to position-0 candidate recall; sglang is the reference for the diff |
| DFlash2 vs MTP3 throughput | **Answered (51)** on sglang: DFlash2 73.8 tok/s vs MTP3 50.9, +45% at c1 -- but step times are identical to MTP7's (77.5 ms), so the win is *all* acceptance and the draft is not cheaper here. At the fork's 3.995 acceptance the same step time puts DFlash2 *below* MTP3 |
| DFlash2 at verify width 8 | Refuted (42) **against a GLM-5.2 target**: 2.68 acceptance against 3.65 break-even, 26.4% slower than MTP3, because a 5.3-trained drafter does not transfer to a 5.2 target. Width 8 itself is survivable at 32.85 ms. **Reopened (43)** now that the served model is 5.3: three missing upstream fixes took acceptance 2.815 -> 4.029, a 32% gap to the card remains, and the matched throughput comparison has never been run |
| Capturing the draft path in CUDA graphs | Refuted (26): the drafts are already graphed |
| Graph-node fusion | Priced (24) at about 1.2%; not started |
| Grace-to-HBM cold-weight staging | Open (30), but its control predates the shared-memory fix and it must be re-measured before it means anything |
| Buying more HBM residency | **Still open (32).** Phase 46 appeared to answer it and did not: `VLLM_TIERED_MOE_PROFILE_CAP` defaults off, so a profile's slot count never bound residency and all three arms ran identical. Needs that flag set, or a lever that moves `available_hbm`. The KV-cache route is still closed (38) |
| Hot-slot count as a tuning knob | **Untested (46, corrected 2026-08-30)**: the profile slot count is not a knob at all unless `VLLM_TIERED_MOE_PROFILE_CAP=1`; the sweep that appeared to test it varied only the ranking |
| Prefix caching under MTP | **Open defect (46)**: first-pass GSM8K drops to 56-59% against 91% without prefix caching or without MTP. Pre-existing, both checkpoints. The losslessness diff test is blocked because greedy decoding is not reproducible on this server, which is a second defect |
| Checkpoint format at fixed placement | Settled (46): W4A16 int4 g32 beats NVFP4 g16 by 9.77% +/- 3.32% on decode, TTFT unchanged; mechanism unattributed |
| Per-layer EP skew in prefill | **Open (36)**: priced at 185-232 ms/chunk, 7-9% of prefill; needs a token-weighted min-max and a new kernel |
| Symmetric memory under DCP | Guard kept (37): no hang in 3 runs, but the arm that originally hung never exercised the path |
| The 1.8 ms/step sampling and logits boundary | Open (26); host-side, so graph capture alone will not recover it |
| Prefill/decode co-scheduling | **Top open lever (33, 34)**: a mixed step falls out of the CUDA graph, switches to the un-graphed two-stage all-reduce, and spends 61.5% of itself in communication |
| DCP4 at concurrency 1 | Open (34): 2.4 ms/step, 8.5% of the c1 step, for capacity only c4 uses |
| Dense GEMMs and elementwise glue | Open (34): 10.5 ms/step at c1, 37% of the step, batch-one fixed cost |
| The sampling/logits boundary | Largely closed (34) by the V2 runner: between-graph idle 2.0 -> 0.642 ms |

## Platform

- One Booster node with four NVIDIA GH200 Superchips
- Four 96 GiB Hopper HBM devices connected by NVLink
- Four 72-core Grace CPUs: 288 Arm cores total
- About 857 GiB of NUMA-visible Grace + HBM memory
- ExaSTORE/GPFS project storage; Booster nodes have no external internet
- vLLM commit `d08eebad162bbd1f2e99cca550313daaa81c7654`
- PyTorch 2.11, CUDA 13, TP=4, EP=4

Each rank owns 64 of the model's 256 routed experts. The baseline offloads
40.33 GiB of expert parameters per rank through vLLM's existing UVA offloader,
uses `fp8_ds_mla` KV cache, Inductor mode 3, and full/piecewise CUDA graphs.

## Baseline

Batch one, random input, deterministic decoding, 256 output tokens:

| Input   | TTFT      | Approx. prefill | Decode      | Total    |
| ------- | --------- | --------------- | ----------- | -------- |
| 4,096   | 0.990 s   | 4,139 tok/s     | 37.06 tok/s | 7.87 s   |
| 32,768  | 7.652 s   | 4,282 tok/s     | 37.06 tok/s | 14.53 s  |
| 399,744 | 120.853 s | 3,308 tok/s     | 37.57 tok/s | 127.64 s |

At idle, the working configuration uses about 90.7 GiB HBM per GPU, leaves
6.58 GiB free, and provides a 574,336-token KV capacity. Detailed settings and
result links are in [the baseline report](baseline/baseline-summary.md).

## How we got here

1. Downloaded and checksum-pinned the eight-shard model on the internet-facing
  login node, then used the local immutable checkpoint offline on Booster.
2. Built an editable vLLM environment with `uv` after loading the JUPITER 2026,
  GCC 14.3, CUDA 13, CMake, NCCL, ccache, and Ninja modules.
3. Enabled TP4/EP4, per-rank expert filtering, NUMA binding, 40 GiB/rank UVA
  expert offload, FP8 MLA KV cache, Inductor, autotuning, and CUDA graphs.
4. Moved compiler caches from the quota-limited home directory to scratch.
5. Disabled `fuse_allreduce_rms`; that fused warmup path produced an illegal
  memory access on this stack.
6. Reduced `gpu-memory-utilization` from 0.94 to 0.90. The former left only
  about 2.95 GiB free and failed at an 8K chunk during 32K prefill; 0.90 leaves
   enough transient HBM while preserving more than 400K KV capacity.
7. Repeated warmed 4K and 32K measurements, then completed the 399,744 + 256
  full-context case.

ExaFlash staging was investigated but not used; these results load directly
from ExaSTORE.

## Implementation status

Phase 0 development began on 2026-07-17 on branch
`tiered-moe-grace-view`. The first slice adds a capability-gated CUDA alias for
ordinary pageable Grace allocations, without pinning, registration, or a copy.
Correctness tests cover address identity, bidirectional visibility, ownership,
and invalid storage. Runtime qualification passes for PyTorch CUDA kernels, but
the driver migrates the pages from Grace node 0 to GPU-HBM node 4. Preferred-host
advice, read-mostly advice, and `mlock` did not preserve the physical LPDDR tier.
The direct pageable path therefore remains gated off while the destination-aware
pinned-UVA and CPU contingencies are evaluated.

The first pinned-UVA contingency allocator and GLM-shaped Marlin probe are now
complete. Final pinned backing stays on the local Grace NUMA node, produces
bit-exact Marlin output, and measured about 4.5% slower than HBM for the combined
gate/up plus down matrix sequence at batch one. See the
[Marlin/UVA experiment](experiments/2026-07-17-marlin-uva/README.md).

Phase 1 now has a header-only fail-closed manifest and deterministic EP4 expert
planner. It inventories all 175,527 tensors in about 1.5 seconds and separates
stored checkpoint bytes from the final fused-Marlin layout. Exact tracing of
non-routed tensors through TP4 sharding, dropped indexer copies, and runtime
fusion produces 4,181,609,280 resident bytes per rank. With the measured
machine capacities, the current deterministic plan places 3,097 experts in HBM
and 1,703 in pinned Grace memory per rank when retaining the baseline cache
allocation. Native 400K cache sizing now replaces that baseline input: the
host-main-cache scenario keeps 4,477 of 4,800 local layer-expert slots hot,
while the HBM-cache fallback keeps 3,425 hot. These final counts include exact
two-tier Marlin workspaces, maps, remap buffers, and one-expert conversion
scratch, plus conservative upper bounds for rounded baseline runtime metrics.
Both enforce the v2 plan's 5 GB HBM and 8 GB Grace reserves. Details are in the
[tier-plan experiment](experiments/2026-07-17-tier-plan/README.md).

The next loader prerequisite is also in place: safetensors iteration accepts a
fail-closed per-layer ownership map and skips remote packed weights, scales,
and metadata before payload materialization. Under linear EP4, this reduces the
planned checkpoint stream to 107,382,098,688 bytes per rank. It is tested as an
iterator primitive but is not yet wired into the tiered destination loader.

All dedicated v2 flags now flow through `EngineArgs` into a hashed
`TieredMoEConfig`. Cross-config validation enforces the pinned TP4/EP4, 400K,
batch-one, FP8 MLA, NUMA, reserve, and no-generic-offload contract. The real
`vllm serve --tiered-moe-plan-only` path now builds this engine config, validates
and hashes a checked-in GH200 machine profile, prints both complete physical
plans, and exits before sockets, workers, GPU allocation, or tensor payload
reads. Destination-loader wiring is the next implementation slice.

Phase 2 has started. The real `DefaultModelLoader` now installs the planner's
strict layer-aware EP4 ownership map and forwards it to safetensors. A compact
final-destination layout allocates component-major Marlin tensors from one HBM
buffer and one pinned-Grace buffer per layer, so weights, scales, and shape
metadata cannot split across tiers. A 60-hot/4-cold layer smoke test allocated
the exact 1,245,708,800 bytes in 0.683 seconds; all 256 sampled cold pages were
on the paired Grace NUMA node. A bounded production stager now rejects
interleaved or incomplete bundles, converts one real 19,464,240-byte checkpoint
expert with native Marlin, and commits the 19,464,200-byte result to Grace with
44,662,784 bytes peak HBM. Wiring this path into model parameter creation is
next. Worker startup now resolves the selected cache/expert scenario before
`initialize_model`, retains the exact rank plan in `DefaultModelLoader`, and
exposes it through a scoped construction context. The real rank-0 resolver
matches plan-only. After physical-capacity and runtime reconciliation, the
selected host-cache plan uses 4,330 hot and 470 cold slots across 75 layers.

The destination loader now completes the entire model: four split-shard experts
are deferred without breaking the one-expert staging bound, all 4,800 local
layer-expert slots stream directly into compact storage in about 30 seconds
with warm GPFS cache, and complete model loading takes 41-43 seconds with an
89.2 GiB per-rank model-memory delta. The first tiered execution slice also
completes one native prepare, hot Marlin, cold UVA Marlin, one join, and one
native finalize on all four ranks. The first host-UVA MLA cache slice now
creates all 78 main-cache tensors in paired Grace memory while retaining all
21 indexer tensors in HBM. A 400K server starts with 100% sampled NUMA locality
and 4.64 GiB observed free HBM per rank, and a deterministic eight-token
request completes successfully. A post-warmup audit fails closed below the v2
runtime reserve. See the
[storage results](experiments/2026-07-17-tiered-storage/README.md),
[dispatch trace](experiments/2026-07-17-tiered-dispatch/README.md), and
[host-UVA cache result](experiments/2026-07-17-host-uva-kv/README.md).

The compiled tracer bullet now works with full/piecewise CUDA graphs after
disabling the stack's failing FlashInfer all-reduce/RMSNorm fusion. A cache-tier
A/B showed that host-UVA and HBM main caches both decode at only about 4 tok/s
without graphs; cache placement is not the dominant short-context cost. With
the exact 400K main cache in HBM, graphs reduce TPOT from 245.60 ms to 29.03 ms,
or 34.45 decode tok/s, while preserving deterministic output and a 4.27 GiB
post-warmup physical reserve. A production-shaped Marlin probe also measured
native full-footprint Grace-UVA execution within 2% of HBM and isolated the
fixed two-tier call cost. See the
[compiled host-cache result](experiments/2026-07-17-compiled-host-uva-kv/README.md)
and [compiled HBM-cache result](experiments/2026-07-17-compiled-hbm-kv/README.md).

The hot and cold Marlin branches now use independent views from one workspace
allocation and overlap only for the captured one- and two-token decode shapes.
Large prefill remains serial to avoid an unnecessary 896 MiB warmup peak. Both
graph modes capture, deterministic output is unchanged, and two 4K/256 runs
measure 27.54-27.56 ms TPOT, or 36.28-36.31 decode tok/s. This is a repeatable
5.4% improvement and leaves about 2.0% to the 37.06 tok/s native baseline. See
the [stream-overlap result](experiments/2026-07-17-tiered-stream-overlap/README.md).

Long-context qualification exposed two capacity margins. Raising the planned
HBM reserve from 5 to 7 GB prevents FlashMLA's 2 GiB request-time allocation
from exhausting HBM. The cache planner now also budgets vLLM's permanently
reserved null block. With 3,176 hot and 1,624 cold expert slots per rank, the
32K and exact 399,744 + 256 cases complete at 36.24 and 36.57 decode tok/s. The
full-context TTFT is 118.385 seconds, 2.47 seconds faster than native. See the
[long-context result](experiments/2026-07-17-tiered-long-context/README.md).

The standalone Phase 5 full-footprint gate rejects the host-UVA main cache:
random/sorted graph replay is about 3.0 ms p95 versus the plan's 0.5 ms limit.
A full graphed server retry nevertheless shows why both complete plans matter.
Moving the 19.06 GiB cache to local Grace memory keeps 1,052 more expert slots
in HBM and improves decode from 36.29 to 40.46 tok/s at 4K and from 36.57 to
41.54 tok/s at exact 400K. The cost is a 75% higher cold 4K TTFT and 122%
higher exact-400K TTFT. AUTO remains fail-closed on HBM under the v2 gate, but
host-UVA is retained as a measured decode alternative for trace analysis. See
the [cache gate and production retry](experiments/2026-07-17-host-uva-cache-gate/README.md).

Phase 6 now captures exact request-bound
top-8 routes, validates a fingerprinted arbitrary per-layer EP4 owner map, and
loads it in the full graphed 400K server. A six-request train/two-request
held-out split reduces held-out cold routing from 31.24% to 2.32% in offline
replay. Two matched 4K/256 runs reduce mean TPOT from 27.553 to 25.931 ms,
raising decode from 36.29 to 38.57 tok/s on average. A bounded tail-aware swap
pass and exact request replay then reduce held-out TPOT from 27.099 to 24.209
ms. A cold-critical latency model fitted only on six training requests predicts
both placements on two held-out requests with 2.27% worst error, passing the
v2 20% gate. A bounded sidecar then captured 32 request-bound real DSA rows
across all 21 full-indexer layers. Full-footprint replay measured the real
pattern at 1.484 ms HBM p95 and 2.508 ms host-UVA p95 with exact output across
tiers. Host-UVA still misses the 0.5 ms gate by 5.0x, so AUTO remains on HBM.
See the [trace-placement result](experiments/2026-07-17-trace-placement/README.md)
and [real DSA trace result](experiments/2026-07-17-dsa-index-trace/README.md).

Phase 7 now completes the collective and populated-400K tuning pass. The exact
12 KiB TP4 reduction takes 4.06 us through vLLM's already-selected custom
one-stage backend. A trace-profiled 10 GB-reserve configuration completes two
399,744-input/256-output runs at a mean 107.009 seconds TTFT and 23.765 ms
TPOT, or 42.08 decode tok/s, while retaining at least 4,295 MiB free HBM per
GPU. This is 12.0% faster in decode than the native CPU-offload baseline and
clears the v2 observed-memory gate, but remains below the 100 tok/s project
minimum. See the
[end-to-end tuning result](experiments/2026-07-17-end-to-end-tuning/README.md).

Phase 8 profiles eight exact-400K decode steps on every rank and corrects the
Phase 7 collective model. The 75 routed layers were automatically using
sequence-parallel MoE, producing 150 NCCL reduce-scatter/all-gather pairs per
token and 6.5-7.3 ms of overlapping NCCL activity per rank. Tiered DeepSeek
layers now retain mirrored batch-one hidden states, execute only locally owned
experts, and combine partial results with the existing late custom all-reduce.
Two exact 399,744-input/256-output runs measure a mean 108.648 seconds TTFT and
18.129 ms TPOT, or 55.16 decode tok/s, with only 0.35% TPOT spread and at least
5,409 MiB free HBM. This raises decode 31.09% over Phase 7 and 46.82% over the
native CPU-offload baseline while preserving deterministic smoke output. See
the [decode critical-path result](experiments/2026-07-18-decode-profile/README.md).

Phase 9 grafts the official FP8 MTP layer onto the pinned W4A16 target, loads
only 64 of 256 draft experts per EP rank, and extends physical planning to the
draft weights and caches. MTP3 with size-4 CUDA graphs preserves the exact
eight-token baseline output. It reaches 103.42 tok/s at 4K with 78.51% draft
acceptance and 108.17 tok/s at exact 400K with 60.74% acceptance. The latter is
a 96.1% decode improvement over Phase 8, with a 5.8% TTFT cost and at least
3,311 MiB free HBM after the maximum-length request. See the
[MTP graft result](experiments/2026-07-18-mtp-graft/README.md).

Phase 10 measures 18 deterministic prompts across Python, PyTorch, CUDA C++,
math, email, and technical explanation. Weighted acceptance is 67.61% and
decode is 93.86 tok/s; category acceptance ranges from 58.21% for PyTorch to
77.63% for math and correlates with decode rate at `r=0.990`. An eight-step
exact-400K profile shows that routed W4 work is unchanged per target step, but
rank imbalance is exposed in custom-all-reduce wait. It also finds that CUDA
graphs defeat MTP index sharing and that three serial draft heads perform full
vocabulary gathers. The next measured priorities are MTP-aware placement,
size-4-only sequence parallelism/all-reduce tuning, graph-safe index reuse,
local draft argmax, and an MTP2/MTP3 sweep. See the
[MTP prompt and profile result](experiments/2026-07-18-mtp-prompt-profile/README.md).

The analytical forward-pass roofline shows that the grafted MTP FP8 block is
small; target MoE, its synchronization boundaries, and dense W4 kernels
dominate. Size-4 verification currently serializes hot-HBM and cold-Grace
experts, with a 2.18 ms/step ideal kernel-overlap bound, while sparse MLA pads
16 local heads to 64. See the
[forward-pass roofline](experiments/2026-07-18-mtp-prompt-profile/roofline-analysis.md).

Phase 11 tests MTP6 and stops the deeper sweep. At 4K, MTP6 reaches 107.26
tok/s, but at exact 400K it falls from MTP3's 108.17 to 84.47 tok/s. On the
matched exact request it drafts 98% more tokens while accepting fewer, and the
fourth through sixth draft positions accept only 11.0%, 7.7%, and 4.4%.
Greedy output remains byte-identical. MTP7 and MTP8 were skipped; MTP3 remains
the fixed-depth default. See the
[MTP depth result](experiments/2026-07-18-mtp-depth-sweep/README.md).

Phase 12 extends hot/cold expert stream overlap through size-4 MTP verification
and adds the DeepSeek/GLM local draft-argmax path. Two exact-400K overlap runs
average 127.67 tok/s with MTP3, 18.0% above the serial MTP3 control. MTP2 is
slower at roughly 110-114 tok/s. Traces show that overlap removes 15.7% of the
MTP3 routed span and that local argmax shrinks each draft vocabulary gather
from a 38,720-element shard to one value/index pair. Local argmax has no stable
batch-one throughput benefit, so it remains opt-in and MTP3 overlap remains the
default. See the
[MTP fast-path result](experiments/2026-07-18-mtp-fastpath/README.md).

Phase 13 tests target-only sequence-parallel MoE at MTP3's four-token
verification size. The exact physical plan keeps HBM usage constant by moving
57 more routed experts per rank to Grace. Two exact-400K runs average 112.98
tok/s versus a fresh 123.65 tok/s no-SP control, an 8.63% regression. The
trace shows unchanged routed-kernel span but 150 reduce-scatters and 150 extra
all-gathers per step, adding 7.26 ms of NCCL activity. The experiment was
reverted; tiered MTP retains the non-SP target path. See the
[sequence-parallel follow-up](experiments/2026-07-18-mtp-fastpath/README.md#sequence-parallel-follow-up).

Phase 14 re-analyzes the fresh no-SP control trace with GPU-annotation-aligned
phase segmentation, per-kernel occupancy/duration statistics, and solo-time
attribution. The target verify is 93% of the 28.6 ms in-profile step and the
GPU never idles for 50 us anywhere; the three drafts cost 2.1 ms. The
all-reduce bar is a desynchronization tail (p50 6.5 us at the isolated floor,
p99 162 us), ~20% of routed Marlin launches are near-empty, the DSA top-k uses
32 of 132 SMs, and step speed-of-light is ~6-6.5 ms (~3.5x away). TP2xPP2 was
analyzed and rejected: batch-one pipeline stages serialize and halve aggregate
HBM utilization. DCP4 was identified as the correct KV-dedup vehicle: the MLA
latent cache is replicated across TP, and 16 local heads x 4 folds to exactly
the FlashMLA 64-head shape. See the
[SOL re-analysis](experiments/2026-07-18-sol-reanalysis/README.md).

Phase 15 ports DCP to the FlashMLA sparse backend to unlock concurrency-4 at
400K (c=4 x 400K is 76 GiB/rank of replicated KV without DCP; the same
19 GiB as today with DCP4). The port adds DCP index filtering, base-e LSE
return, and head-fold-aware metadata sizing to the fp8_ds_mla mixed-batch
path; an lse=+inf sentinel for all-filtered rows had to be masked to -inf to
keep the cross-rank combine finite. A new kernel-level unit test matches the
real FP8 kernel under simulated DCP4 sharding against a full-index reference
on the login-node GH200. The tiered contract admits DCP1/DCP4, the KV planner
shards blocks (20.47 GB -> 5.19 GB per rank), and the residency planner
promotes cold experts when the budget grows (hot 2,870 -> 3,713 of 4,800).
Both DCP1 controls reproduced the exact-400K SHA at 129.1-136.4 tok/s. The
full-graph capture crash was bisected through eight rounds to the
graph-memory profiling pass (temporary pool + minimal stand-in KV cache);
the identical graph captures cleanly in the real capture path, so profiling
now skips FULL graphs under DCP. DCP4 then qualified losslessly at c=1
(76-82 tok/s: a later trace measured ~271 DCP collectives/step, costing
~12 ms against ~5 ms
of wins) and concurrency 4 was enabled: per-sequence KV provisioning,
config-derived overlap gating, and deterministic residency
promotion/demotion when the HBM budget moves. Two steady-state runs measure
**c=4 x 400K at 173-179 tok/s aggregate (43-45 tok/s per agent)** with the
c=1 golden SHA reproduced on the same server - 1.35x the best
single-request aggregate, +36% effective per-agent versus queueing on the
DCP1 config. Prefills serialize (~110 s each), so cold simultaneous
400K-agent starts ladder their TTFTs; cross-turn prefix caching is the
mitigation. Trace-guided A2A+NVLS reached 190.15 tok/s once, but a replica
stalled in collective NCCL window registration. The stable default remains
`ag_rs`; commit `990b1d378` adds the minimal DCP safety gate needed before a
future NVLS requalification. See the
[DCP port](experiments/2026-07-18-dcp-port/README.md).

Phase 16 removes the main host-launch bottleneck by qualifying the V2 GPU
runner's complete MTP CUDA graphs. A missing DCP draft-length refresh caused
mixed c4 batches to deadlock; after the minimal fix, the deterministic 400K c1
SHA passes and matched 4K c4 median ITL falls from 64.46 to 57.37 ms. Combined
with acceptance improving from 3.006 to 3.230 tokens, effective aggregate
throughput rises from 186.52 to 225.23 tok/s (+20.8%). See the
[V2 MTP full-graph follow-up](experiments/2026-07-18-dcp-port/v2-mtp-full-graph.md).

Phase 17 makes the 24-prompt Python/PyTorch/ML/math suite the primary c1/q4
performance regression gate. With prefix caching disabled, one excluded full
warmup, and two measured repetitions, pre-route-capture and current source are
identical at 108.09 and 108.08 decode tok/s. The requested exact `44 / 0 / 31`
layer layout reaches 95.88 tok/s, 2.1% below `44 / 1 / 30` and 11.3% below the
108.08 tok/s per-expert frequency layout. Exact-400K synthetic runs remain
memory/DCP/correctness stress tests, not the headline performance baseline. See
the [realistic placement qualification](experiments/2026-07-19-c1q4-placement/README.md#warmed-realistic-prompt-qualification).

Phase 18 qualifies the V2 model runner at c1/q4. The earlier apparent startup
hang was active first-use Inductor autotuning after the logged compile phase;
reusing completed VLLM, FlashInfer, and TRT-LLM caches lets every graph mode
start successfully. Full V2 graphs are required, but on the warmed realistic
suite V2 reaches 106.08 decode tok/s versus V1's 108.08 (-1.85%), and 95.44
versus 98.03 end-to-end tok/s (-2.65%). V2 remains the c4/DCP4 choice, while
V1 remains the c1/q4 default. See the
[V2 c1/q4 qualification](experiments/2026-07-19-v2-c1/README.md).

Phase 19 deploys the qualified c1/q4 V1 configuration as a four-hour,
authenticated Booster API for Claude Code on the login node. A single wrapper
submits or reuses the Slurm job, discovers its allocated hostname, verifies the
direct internal connection, exports all Anthropic model variables, and launches
Claude. Native messages, token counting, automatic tool calls, and a complete
two-turn Claude Code `Bash(pwd)` loop pass. A metrics monitor reports rolling
computed-prefill and decode throughput. Its load test exposed and fixed a
quantized chunked-context dtype dereference; repeated shared-prefix Claude
requests now pass at roughly 114 decode tok/s. See the
[Claude Code deployment](experiments/2026-07-22-claude-local/README.md).

Phase 20 re-analyzes the c4 MTP-depth sweep traces from the raw Perfetto data,
segmenting steps by CUDA-graph correlation ID rather than by profiler
annotations, and identifying MoE tiers from `apply_tiered` stream semantics. Of
the 62.43 ms MTP3 c4 step, routed Marlin is 20.15 ms of critical path, the TP
all-reduce 9.27 ms, GPU idle 7.95 ms, and all 400K-context-specific work only
7.6 ms. Classifying the 166 reductions shows post-attention ones sit at the
5.0 us payload floor while post-MoE ones carry 7.82 ms of skew wait, matching an
independent per-layer max-minus-mean of 7.76 ms. Reading the uncontended cold
`w13` launch as the true rate puts the Grace tier at the 421 GB/s C2C roof with
2.85 activated experts per layer and a ~10.4 ms solo cost hiding inside a
24.5 ms hot window. Offline replay of the captured routing traces then refuted
the two placement hypotheses that framing suggests, before any cluster time was
spent: only 0.61 ms of the skew is expectation the owner map can reach (86% is
an irreducible order statistic, and an earlier 77%-persistent reading was
five-sample bias), and the hot-slot frontier is monotone in the opposite
direction, with 24.8% of layer/rank cells already cold-bound. What remains is
that per-expert kernel cost multiplies both the Marlin span and the skew, that
the hot-slot frontier is flat past ~3,000 slots so HBM should buy KV, and that
4.3 ms/step is host round-trip with the GPU ~95% idle while the host sits 50 ms
ahead. The routed MoE is only 35% fixed weight streaming (9.02 ms +
1.057 ms/token), which is the real reason c4 yields 1.35x rather than 4x. See
the
[critical-path review](experiments/2026-07-25-c4-mtp3-critical-path/README.md)
and the [c4 MTP-depth sweep](experiments/2026-07-23-c4-mtp-depth/README.md) it
re-analyzes.

Phase 21 resolves a 5x disagreement about Grace C2C bandwidth and, in doing so,
finds the largest remaining lever. Sweeping the Marlin MoE launch configuration
is a dead end: every `blocks_per_sm` value lands within 0.4% of the auto
heuristic and splitting the SM budget across tiers gains 0.0-0.4%. A first
isolated probe appeared to show the Grace tier collapsing to 69 GB/s, but that
benchmark handed the cold tier's experts the whole routing and so timed
re-streamed blocks against logical bytes; production splits one routing across
tiers by `expert_map`, leaving one block per cold expert. With faithful routing
the cold path holds 88-95% of the 421 GB/s roof, flat across M=8/16/32, and
doubling the routing mass doubles blocks and time together at 95% of roof. The
2026-07-17 probe's "within 2-4% of HBM" turns out to have been measured where
HBM itself ran at 10.4% of its own roof, so it never spoke to bandwidth. The new
result is that the two tiers barely overlap: hot 115.3 us and cold 104.4 us
combine to a 194.0 us union against a 115.3 us ideal, capturing only ~25% of the
available overlap, and the trace's 25.7 ms layer span matches a serial estimate
rather than the 13.2 ms ideal. The "39-40% overlap saving" reported earlier
compared against a sum of dilated durations. About 12 ms/step, 19% of the step,
is available if the tiers genuinely overlapped. See the
[Grace bandwidth resolution](experiments/2026-07-25-grace-bandwidth/README.md)
and [Marlin decode tuning](experiments/2026-07-25-marlin-decode-tuning/README.md).

Phase 22 completes the host-round-trip work from one upstream line. The trace's
host API timeline showed each draft step blocking in a `cudaStreamSynchronize`
inside an `aten::to` on a 0-d int tensor; the caller is `get_dcp_local_seq_lens`
in `vllm/v1/attention/backends/utils.py`, which materialized the Python
`dcp_rank` as a 0-d device tensor. That is a pageable host-to-device copy, and a
pageable H2D blocks the host until the stream drains — once per speculative
draft step, via `_build_draft_attn_metadata`. The value is only used in a scalar
multiply, so a Python int is equivalent; equivalence was checked over
`dcp_size` in {2,4,8} and interleave in {1,4,64} on CPU and CUDA, with 40 unit
tests passing. The host moved from a 37.5 us launch lead to 104.9 ms, so
inter-graph GPU gaps fell from 4,547 to 607 us/step: draft-to-draft collapsed
786 -> 70 us and the gap containing the eager prologue collapsed 2,617 ->
173 us, covering both halves of the planned work. Realistic c4 output rises
179.47 -> 185.05 tok/s warmed and 4K c4 by 5.5-7.0%, with the golden 400K SHA
reproduced. The bug is upstream, not in this branch; any DCP plus
speculative-decoding deployment pays it. See the
[draft-sync fix](experiments/2026-07-25-draft-sync/README.md).

Phase 23 closes P1b, the largest lever the critical-path review had identified,
as a refutation. The review put ~12 ms/step of headroom in hot/cold MoE overlap.
A diagnosis sweep first showed that `blocks_per_sm` is honoured by `marlin_mm`
only when `thread_k` and `thread_n` are also passed, so an earlier
SM-partitioning sweep had been a null experiment; with a grid read-out proving
the knob live (132/264/396 blocks for bps 1/2/3), partitioning is genuinely
useless and the shipped 3/3 default is the best of nine combinations. An
isolated benchmark then suggested issuing the hot tier first, raising
co-residency 22% -> 86%. That change was implemented, verified live in the trace
(hot starts first in 98.7% of layers, up from 2.0%), measured at -0.94% on the
routed layer span, and reverted. Direct measurement explains why: in production
the tiers are **already 81.5% co-resident** and the layer span is within 4% of
the in-situ `max(hot, cold)`. The 12 ms came from comparing a production span
against isolated solo kernel costs, an ideal that assumes both tiers run at solo
speed while overlapped. The isolated benchmark's low co-residency was an
eager-launch artifact that does not survive CUDA-graph replay. Same error class
as the "39-40% overlap saving" corrected earlier; the rule is never to compare
an in-situ time against an isolated time and call the difference recoverable.
See the [diagnosis plan](experiments/2026-07-25-p1b-phase-a/README.md),
[tier-overlap diagnosis](experiments/2026-07-25-tier-overlap/README.md)
and [reorder qualification](experiments/2026-07-25-tier-order/README.md).

Phase 24 prices P4 before building it. The per-graph-node cost model implied by
the trace is confirmed by direct measurement: a serial chain of dependent
kernels gives 1.089-1.201 us marginal per node, and a controlled fusion holding
total work constant while halving the node count returns 1.074 us per node
removed, linearly across three halvings, with the gap flat at ~0.9-1.4 us for
every kernel width the glue occupies. So node removal genuinely pays, and the
trace's ~1.14 us inference was right. The ceiling is what disappoints: 3.78 ms
of intra-graph idle across 4,436 nodes is 0.85 us of exposed cost per node, so
removing every node would return 6.1% of the step and a realistic fusion
campaign of ~900 nodes returns 0.77 ms, about 1.2%. Recommendation is not to
start one, to carry the ~0.45 ms along if a single-tier MoE merge is built for
other reasons, and to reuse 0.85 us/node as a constant for pricing future
designs that change node count. See the
[graph node cost measurement](experiments/2026-07-25-graph-node-cost/README.md).

Phase 25 adds AutoRound W4G64 support and compares it with the W4G128 baseline.
The new checkpoint loads through the same tiered Marlin path and is viable, but
its qualified c1/MTP3 prompt-suite throughput is 3.5% lower. A subsequent
five-shot GSM8K screen runs 256 fixed questions twice across four parallel
nodes. MTP3 raises end-to-end output throughput from 41.52 to 80.05 tok/s for
W4G128 and from 42.40 to 78.09 tok/s for AutoRound. AutoRound averages 1.4-2.0
accuracy points higher, but identical W4 repetitions vary by 2.0 points, so the
quality difference is inconclusive without the full 1,319-question set. Active
minimum free HBM ranges from 7.35 GiB for W4+MTP3 to 12.76 GiB for target-only
W4. See the [AutoRound bring-up](experiments/2026-07-26-autoround-w4g64/README.md)
and [GSM8K comparison](experiments/2026-07-26-gsm8k-quant-mtp/README.md).

Phase 26 finds the mechanism behind every failed hot/cold overlap experiment and
removes it. Marlin's MoE launch passes `deviceSharedMemOptin / blocks_per_sm` as
its dynamic shared memory rather than the `sh_cache_size` it computes one line
earlier: at the production tile that is 76,458 bytes per CTA against 25,856
actually indexed, so one wave claims 229,374 of 233,472 bytes and **no second
Marlin CTA can be placed on any SM at any `blocks_per_sm`**. The two tiers were
serialized by the block scheduler before any stream, priority, launch-order or
occupancy knob applied, which is why the `blocks_per_sm` sweeps, the SM-budget
split, the hot-first reorder and green contexts all returned nothing, and why
`2026-07-25-tier-overlap`'s "81.5% co-resident" (a kernel time-range overlap,
not CTA residency) read as no headroom. Booster reports the same shared-memory
geometry as the login node, and a per-CTA `%smid` probe confirms the peak never
exceeds the hot wave at the production request but reaches 4 once the request
fits. Requesting only what the kernel uses and launching hot at 2 CTAs/SM and
cold at 1 cuts the two-tier union to `max(hot, cold)`: on Booster under
CUDA-graph replay the full w13+w2 chain improves by a **median 34% across 15
activated-expert cells** (best 40.1%, worst 14.6%, every cell positive) and
**31-45% at the production c1/q4 shape**, with a twelve-combination grid search
confirming the shipped constants are optimal there. Two measurement traps were
found and corrected along the way, both of which had produced a confident wrong
answer first: timing the fork/join eagerly charges two stream barriers per
iteration, a ~110 us floor on Booster that exceeds the whole kernel at low
activated-expert counts and made an earlier sweep report ~0% exactly where the
gain is largest; and the login node's preferred cold grid (66 CTAs) is not
Booster's (132), so tuning the constant on the login node would have shipped a
value wrong on the machine that matters. Correctness is bit-exact at a fixed
grid; changing the grid moves the fp32 reduction order by at most one bfloat16
ulp, deterministically, so the exact-400K golden SHA has to be re-established
rather than treated as a regression. See the
[shared-memory monopoly](experiments/2026-07-29-marlin-smem-monopoly/README.md).

End to end, the fix is worth **5.3-8.1% of decode step time**, and the gain grows
with concurrency. The controlled server measurements, all same-node off/on pairs:

| protocol | off | on | step time |
| --- | ---: | ---: | ---: |
| no-MTP c1, two runs each | 22.25 ms | 21.07 | **-5.3%** |
| no-MTP c2 | 27.08 | 25.35 | **-6.4%** |
| no-MTP c4, two runs each | 32.46 | 30.30 | **-6.7%** |
| MTP3 c1 same-node, two runs each | 9.18 | 8.44 | **-8.1%** |

The no-MTP protocol is the discriminating one: it removes acceptance-rate noise
and reproduces to +-0.1 ms. Prefer these numbers when quoting the fix. The
kernel-level 34% median is the isolated two-tier chain, not a step, and the
profiler trace below is a smaller number again because it is c1/M=4 under an
active profiler on a single prompt - it is there to explain *where* the win comes
from, not how big it is.

The post-fix production trace closes the mechanism at engine-step scale. Two
same-configuration off/on captures on separate Booster nodes measure
**25.47 -> 24.31 ms mean step wall (-4.56%)**; routed-layer critical spans save
1.04 ms and account for about 90% of the step improvement, while GPU idle stays
flat. Attribution by CUDA launch correlation makes the per-step kernel census
exact in all 176 rank-steps (166 all-reduces, 306 Marlin GEMMs, 4 all-gathers)
and confirms the step budget's shares to within 0.5 points. Three results reframe
what is left. First, **a quarter of every step does no useful work**: 2.01 ms of
GPU-empty gaps plus 4.13 ms of all-reduce barrier spin that the profiler scores
as busy. Second, **the cold tier is at the C2C roofline** - 53 us/layer for a
20.1 MB W4G64 expert is 379 GB/s against the 373 GB/s this worklog measured as
achievable, it is node-invariant to 0.2% where hot varies 2.5%, and so cold
Marlin kernel tuning is retired as a lever; residual overlap headroom is a
bounded 1.72 ms/step. Third, the next lever is **per-layer EP-rank balance at
3.615 ms/step** of summed max-minus-mean routed-layer skew (correcting an earlier
1.485 ms figure), which now reconciles to within 12% with the 4.13 ms of
all-reduce synchronization excess - one lever, not two additive ones. Worst
layers are 34, 69, 18, 20, 61 and 67.

Splitting the GPU-empty bucket by CUDA graph position settles two more questions.
The step replays **seven** graphs, not one - the target at 3,466 kernels plus a
16- and a 20-kernel graph per draft round - and the six draft graphs hold
0.003-0.008 ms of internal idle each, so **the MTP drafts are already graphed and
capturing them is not a lever**. Two thirds of the between-graph idle is one
boundary: the sampling and logits region after the target graph, at 0.745 ms of
device idle plus 1.071 ms of un-graphed GPU work (0.561 ms of it the
`[4, 6144] x [6144, 38720]` vocabulary projection), about 1.8 ms of the step. For
0.703 ms of that idle the host is inside no CUDA call, so it is Python sampler
work and graph capture alone would not recover it. The gap structure is identical
off and on, the right invariance for a device-schedule fix. Full attribution and
reproducers are in the
[Phase 26 report](experiments/2026-07-29-marlin-smem-monopoly/README.md#post-fix-production-trace-2026-07-29).

Phase 27 measures the current c4/DCP4/MTP3 path on the original 18-prompt
mixed-domain suite: three prompts each for Python, PyTorch, CUDA C++, math, email
and technical explanation, 256 forced output tokens per request at concurrency
four. Two warmed repetitions give **18.687 ms mean per-request TPOT** and
**181.60 tok/s mean aggregate output throughput**, at 69.19% draft acceptance.
Quote the TPOT: it reproduces to 0.16% where the aggregate reproduces to 1.8%,
and that 1.8% is almost entirely the 1.9-point swing in draft acceptance between
repetitions. The phase also retracts its own per-domain table - `--disable-shuffle`
makes file order the submission order, the prompt file groups domains in blocks
of three, so domain aliases wave position and all six `explanation` requests are
draining requests. Drain position alone is worth **15.7%** (16.175 ms TPOT
against 19.190 ms steady), which is the whole apparent domain spread. A
round-robin `prompts-interleaved.jsonl` arm replaces it. See the
[mixed-domain c4 result](experiments/2026-07-29-c4-mtp3-mixed-suite/README.md).

Phases 28 to 30 were run between 2026-07-25 and 2026-07-28 but were never
indexed here; they are backfilled in date order and numbered after Phase 27
rather than renumbering the chain.

Phase 28 makes both third-party speculators work and adopts neither. DSpark
(`RedHatAI/GLM-5.2-speculator.dspark`, block 8) needed five distinct real
defects cleared, three of them upstream and two in this branch's tiered
contract, and then reproduced the exact deterministic completion at c1. DFlash
(`UCloud-org/GLM-5.2-FP8-DFlash`, block 16) loaded first try with no blockers
at all. Both lose to MTP3: 4-wide verification reaches 106.08 decode tok/s and
95.44 end to end, 9-wide DSpark ~93 and 84.67, 16-wide DFlash ~70 and 65.6.
**The ranking is strictly inverse to verify-batch width, and acceptance length
is anticorrelated with throughput** - DFlash accepts 6.84 of 15 against MTP3's
~2.9 of 4 and is 34% slower. The cause is the property Phase 20 measured: this
target's routed MoE is only 35% fixed weight streaming, so verification cost
grows nearly linearly with block width while acceptance grows sublinearly. Step
time against verify width is 27 ms at 4, 42 ms at 9 and 97 ms at 16. Source
changes were reverted. See the
[DSpark bring-up](experiments/2026-07-25-dspark/README.md) and
[DFlash result](experiments/2026-07-25-dflash/README.md).

Phase 29 sweeps AutoRound W4G64/MTP3 residency extremes on the same 256-question
five-shot GSM8K harness as Phase 25, from zero target experts in HBM up to
1,800 per rank, plus trace-cold placements. The literal all-Grace job failed
during pinned allocation before checkpoint loading, so 300 and 600 hot experts
bracket the practical floor. The phase also stands up the Claude Code routing
capture: routed-expert IDs are recorded from natural Claude Code turns, without
storing prompts, responses or repository contents, and aggregated into a
75-layer by 256-expert hotness grid. That capture set is what every later
placement replay is fitted and validated on. Its smoke run exposed a streamed
output-token metadata bug, fixed in `ffae5a399`. See the
[residency extremes](experiments/2026-07-26-autoround-residency-extremes/README.md)
and [routing capture](experiments/2026-07-26-claude-routing-capture/README.md).
The c1 and c4 AutoRound Claude hosts that serve these sessions are documented in
[`2026-07-26-claude-autoround`](experiments/2026-07-26-claude-autoround/README.md)
and [`2026-07-26-claude-autoround-c4`](experiments/2026-07-26-claude-autoround-c4/README.md).

Phase 30 tests hard SM isolation and fails it. The CUDA 13.0.48 runtime has no
green-context wrappers, so the probe uses the driver API; green contexts do
partition SMs and dispatch concurrently, which passes the mechanism gate. They
then lose to the existing production fork/join on every split and every axis:
at 8, 16, 24 and 32 cold SMs the green union is 388.5, 307.4, 300.5 and
304.1 us against a same-job production union of ~278 us, 9-40% worse, with hot
interference at +43-47% against production's +3-6%. Track A is dead. The pivot
measured Grace-to-HBM staging - overlap a pure C2C weight copy with hot Marlin,
then run cold Marlin from HBM after hot retires - at 19.8% faster at m=16 and
36-38% at m=256, bit-exact for the copy. **That result must not be read as
current**: its production control is the pre-Phase-26 co-run, which the
shared-memory monopoly had serialized, and Phase 32 later measured cold-from-
Grace at the C2C roof with the tiers 99.4% co-resident. Staging has to be
re-measured against post-fix production before it means anything. See the
[green-context probe](experiments/2026-07-27-green-context-marlin/README.md).

Phase 31 lands per-step replica assignment, the first thing to move the
EP-rank skew that Phase 26 identified as the top lever. Selected routed experts
get a second physical copy in otherwise free paired-Grace memory, and after
routing each active logical expert is assigned to exactly one physical copy so
that the predicted slowest-rank MoE time is minimized. The first implementation
optimized a harder objective than it needed to, paid 375 extra graph nodes per
rank-step, and was measured across nodes at n=2; it netted out flat and was
reverted in `53fe6dfcf`, preserving the Phase 26 shared-memory fix that the same
commit had carried. The v2 rewrite is simpler - the corrected min-max objective
needs no cost constants - and fuses assignment with dual-tier Marlin alignment
so the extra nodes disappear. Same-node alternating arms with paired statistics
measure **5.96% off the acceptance-free step at batch 4 and 6.50% at MTP3
batch 16**, with acceptance flat at +1.60% (t=0.88) and an acceptance-corrected
step time of -5.11% at t=-13.85. A paired trace over 160 rank-steps per arm
confirms the mechanism rather than the outcome: rank skew falls 61.5%
(8.230 -> 3.170 ms), TP all-reduce residency falls 55.4% (9.223 -> 4.114 ms),
GPU busy falls 9.1%, and the Marlin launch census and union are unchanged. The
v1 "counter-cost" is settled as a cross-node artifact - Marlin cumulative moves
+1.5% on one node, not +13.4%. Exactly-once is enforced by a graph-capturable
per-step route fingerprint over all 75 layers, replacing v1's check that
required `--enforce-eager` and covered only layer 0. See the
[v2 plan and results](experiments/2026-07-31-replica-scheduling-v2/README.md),
which supersedes
[v1](experiments/2026-07-31-replicated-expert-scheduling/README.md) and absorbs
[the alignment fusion](experiments/2026-07-31-replica-align-fusion/README.md).

Phase 32 stops at its own Phase 1 gate and is more useful for it. A layer's
four Marlin launches have a union of 310.0 us against a sum of 441.8 us, a
ratio of 0.70 where perfect overlap should approach 0.50. The cause is not
occupancy: the measured union is within 2.0 us of the 308.0 us floor the
observed chain lengths permit, the two streams start within 1.2 us of each
other, and kernel-level overlap work has at most 0.15 ms/rank-step in it. The
gap is that one tier does ~39% more work than the other within a layer. The
proposed fix - redistribute HBM slots across layers to minimize the sum of
per-layer `max(hot, cold)` - **models at exactly 0.0%**, because redistribution
only pays when some layer is hot-bound and none is. A cold expert costs 45.32 us
against a hot expert's 9.75 us, 4.6x, and 6.77 cold experts per layer in 306.9 us
is ~442 GB/s per rank, at or past the 421 GB/s C2C roof. Balance would need
about 4.7 GB more resident HBM against a measured peak free of 3,874 MiB. The
phase also corrects its own Phase 0: identifying tiers by CUDA stream is invalid
under graph replay, where stream ids rotate per layer, so the "cold longer in
56.6% of layers" direction claim was random labelling and is withdrawn; tiers
are distinguishable by grid, 264 blocks hot against 132 cold. The next lever is
not balance but residency. See the
[tier-balance result](experiments/2026-08-01-marlin-tier-overlap/README.md).

Phase 33 measures the deployed server on a long-context coding shape: 16 unique
16,384-token PyTorch code-generation prompts, 512 forced output tokens, at
concurrency 1 and 4, run inside the server's own allocation against the live
production job rather than a purpose-built one. Steady-state decode is
**101.3 tok/s at c1** and **66.3 per request / 265.2 aggregate at c4**, with
draft acceptance flat at 61.6% versus 61.5% and per-position 81/61/44. But
end-to-end output is only 50.16 and 70.98 tok/s, because at 32:1 input-to-output
the workload is prefill-bound: prefill runs at ~3,200 tok/s and takes 82 of the
115 s at c4. The gap is one mechanism - chunked prefill of a newly admitted
request stops all four decode streams for ~1.8 s, 118 times per run, holding
213 s of the 330 s of summed inter-token gap. Nothing was preempted and KV
peaked at 4.1%, so it is scheduling, not capacity, and a decode-plus-prefill
step would recover it. A wall-clock reconciliation from the two rates lands
within 2.4% of the measured total. Quote the steady-state number for decode
work and the end-to-end number for this shape; they answer different questions.
See the
[16K coding benchmark](experiments/2026-08-05-pytorch-16k-c1-c4/README.md).

Phase 34 profiles the deployed configuration on the Phase 33 shape and finds
the mixed step. Four torch captures - prefill, c1 decode, c4 decode, and a
window straddling the end of prefill - measure profiler distortion at +4.3% on
c1 and +0.8% on c4, since graph-replayed steps are barely instrumented, so the
decode absolutes are quotable. At c1 the 28.1 ms step is 24.1% routed Marlin,
19.1% TP/EP communication, 18.9% dense GEMMs, 15.6% glue, 10.8% GPU-empty and
8.0% attention; hot/cold overlap sits **1.287 ms above `max(hot, cold)`** and EP
skew is **2.454 ms/step**, down from the 3.615 ms Phase 26 measured before
replica assignment. At c4 the 42.9 ms step is **44.5% routed Marlin** while
dense, glue and attention barely move from c1 - the batch-one fixed cost
amortizes and the MoE becomes the step. Prefill costs 2.62 s per 8,192-token
chunk and had never been profiled: 34.0% routed Marlin, 22.4% communication,
20.4% attention. The V2 runner has also already closed most of the
sampling/logits boundary Phase 26 named, cutting between-graph idle from 2.0 to
0.642 ms and un-graphed kernels from 1.918 to 0.310 ms across 4 replays instead
of 7. The new lever is the **mixed step**: when a chunked prefill shares a step
with decode, that step issues **six** one-stage custom all-reduces instead of
166, the other 160 falling back to the un-graphed two-stage kernel, and the step
leaves the CUDA graph - 50.6 ms/step of un-graphed kernels, 15.0 ms of host
outside any CUDA call, and **61.5% of the step in communication against 15.8%
in pure decode**. On a shape that is 71% prefill-bound that is the largest
lever the trace exposes. Also priced: DCP4 collectives cost 2.4 ms/step, 8.5%
of the c1 step, buying capacity only concurrency 4 uses. See the
[production profile](experiments/2026-08-05-prod-profile/README.md).

Phase 35 takes the first prefill lever the profile named and finds the proposal
wrong in both halves. Phase 34 read prefill's routed MoE at 10.6% of bf16 peak
running the pre-Phase-26 legacy launch and proposed ungating the tight
shared-memory policy and lifting `ops.cu`'s `allow_count <= 2` cap. Sweeping the
launch config at the production chunk shape with explicit `(thread_k, thread_n)`
- which makes `ops.cu` skip `determine_exec_config` and take `blocks_per_sm`
verbatim - shows shared memory is worth **0.3%** here, because a prefill chunk
launches **1,051 blocks against 132 SMs** and co-residency stops mattering once
the grid oversubscribes the machine eightfold; and more CTAs per SM is
consistently *worse*, 0.82x at two and 0.92x at three, so the cap is above the
optimum rather than below it. The real lever is the tile: Marlin's heuristic
picks a 128-thread `(64, 128)` - confirmed at **0.0 ulp** against auto, which
doubles as the harness's positive control - where the 256-thread `(64, 256)` is
**1.10-1.12x** faster at every chunk size, differing by 0.2 ulp of
reduction-order noise. Honestly sized that is ~80 ms of a 2,619 ms chunk, 3.1%
of prefill and **~2.2% of c4 wall clock**, against the profile's ~12% estimate,
which was 5x optimistic because it assumed an occupancy limit that does not
exist; whatever holds W4A16 Marlin to ~11% of peak is not reachable from the
launch configuration. Shipped in `e59d34275` behind
`VLLM_TIERED_MOE_PREFILL_TILE`: `MarlinLaunchPolicy` gains an explicit tile and
a `min_tokens` bound, and `_fused_marlin_moe` takes a *list* of policies so
decode and prefill can have opposite launches without special-casing. A unit
test caught the trap that makes this dangerous - Marlin only instantiates some
`(thread_m, thread_n, thread_k)` combinations and **an explicit tile with no
instantiation raises at launch rather than falling back**, so `requires_block_m`
now pins a tile to the block size it was validated against. **The server-level
A/B has not been run and the flag defaults on.** See the
[prefill Marlin launch sweep](experiments/2026-08-05-prefill-marlin-launch/README.md).

Phase 36 retires bandwidth and protocol as prefill levers and re-attributes the
cost to rank skew. Prefill spends 593 ms/chunk, 23.8%, in collectives, at 0.0%
overlap with compute - comm 592.9 + compute 1902.6 + idle 123.7 against a
2,619.1 ms wall, with 99.9% of both issued to the main stream - so any real
bandwidth win would convert 1:1 into wall clock. A four-arm `NCCL_PROTO` x chunk
size A/B returned 1.003x and unchanged kernel names, meaning the variable never
reached the communicator; the 16K arms were refused by design at
`vllm/config/vllm.py:2316`. A standalone four-rank probe outside vLLM shows
`NCCL_PROTO` *is* honoured there (forced `LL` costs 2.1x) and that **NCCL's
default already picks the fast protocol**: 514 us unset against 520 us for
`Simple` on the same 96 MiB message. Production's *median* all-reduce is
**482-520 us - the hardware optimum** - with a p90 of 6.1-6.5 ms and only 4.0%
cross-rank spread of means. The 70 GB/s bus rate in the Phase 34 profile was
averaging wait time into transfer time; **~250 ms/chunk, 9.5% of prefill, is
waiting for a rotating straggler.** Ungating replica assignment does not take
it: aggregate rank load is already balanced to 1.21% on Marlin, and the
imbalance is *per-layer*, 45.2% spread at the median and a summed max-minus-mean
of **185.7 ms/chunk**, which reconciles the two because the straggler rotates.
The existing scheduler cannot address it - its own docstring assumes every
active cold expert costs the same Grace read regardless of token count, true at
decode's 45.32 us per expert and false at prefill, which is compute-bound by
8.1x over its streaming floor and activates nearly every expert at 8,192 tokens,
making a count-based min-max degenerate. The lever is real and priced at
**185-232 ms/chunk, 7-9% of prefill**, but needs a token-weighted objective and
a different kernel. Doubling the chunk was also dropped: the Marlin arenas are
`M * 393,216` bytes, so 16K costs +3.22 GB/rank against 7.15 GiB free, which
would mean moving 160 experts/rank out of HBM to save ~123 ms of prefill. This
is the fourth decode-only gate found in the same area. See the
[collective A/B](experiments/2026-08-05-prefill-comms-ab/README.md) and the
[protocol probe](experiments/2026-08-05-nccl-proto-probe/README.md).

Phase 37 pays off the runtime qualification `990b1d378` has owed since Round 9
of the DCP port, and mostly fails to. Symmetric-memory all-reduce is disabled
under DCP because job `976497` hung with one rank inside
`ncclCommWindowRegister` while the others had entered the matching all-reduce -
parking A2A+NVLS's 190.15 tok/s as experimental against the qualified `ag_rs`
default's 179.56. Instrumenting the allocator with one INFO line per window
registration and adding a `VLLM_NCCL_SYMM_MEM_ALLOW_DCP` escape hatch, three
production-config runs with the guard lifted all came up, returned identical
greedy text and finished the benchmark. **No hang, and registration is exactly
symmetric** - 20/20/20/20 in the first run and 25/25/25/25 on the `ag_rs` arm,
agreeing on window index and size. But **the `a2a` arm registered zero windows**
despite identical environment and backend dispatch order, so the arm that would
have tested the configuration that actually hung never exercised the path, and
A2A+NVLS remains unqualified. A collection bug nearly hid this: `job.sh` greps
`*-server.err` where the lines go to `*-server.out`, making all three saved
`*-registrations.txt` byte-identical re-dumps of the first run - the arm data
had to be rebuilt from the raw server logs. The guard was **kept**; three clean
runs are weak evidence against a race, and the instrumentation was reverted. See
the [symmetric-memory qualification](experiments/2026-08-05-symm-mem-dcp/README.md).

Phase 38 prices KV-on-Grace in isolation, at the kernel rather than the server,
and the answer is a wash. Decode reads the KV cache in exactly one place, so if
the attention kernel could read it over C2C the ~19.3 GiB/rank it occupies would
go to expert residency - the lever Phase 32 named. Measured on the production
FlashMLA sparse fp8 path over a full 78-layer 400K-capacity cache, with MTP3's
query-token counts (4 at c1, 16 at c4), Grace costs **+8.5 ms/step at c4**:
2.197 ms against 10.691, 763 GB/s against **157**. Output was exact across tiers
in all nine pattern x token combinations, so the tier is transparent to the
arithmetic; index locality is worth far more than MTP's index sharing, with a
clustered pattern reaching 281 GB/s against random's 157 and `independent`
indices only 6% behind `shared`. Against a residency gain estimated at
~9.4 ms/step that is a wash before counting prefill, or the fact that the cold
expert tier already contends for the same link. The ceiling probes matter more
than the verdict: contiguous streaming reaches **400-420 GB/s**, a TMA
descriptor load on host-mapped UVA memory reaches **419.5**, and the copy engine
422 - so neither the link nor TMA is the constraint. Two readings taken during
this work were wrong and are recorded as such: 52 and 157 GB/s were read as
*hardware* limits when they are kernel limits, and the gap to a plain gather's
361 GB/s was read as 2.3x of headroom when that gather was running 1,056 CTAs
against the kernel's 132. Left unexplained: **FA3 GQA sits at 52 GB/s from
Grace and does not move** across a 32x change in batch or context, while HBM
scales 151 to 385. See the
[isolated KV tier measurement](experiments/2026-08-05-kv-grace-attn/README.md).

Phase 39 acts on that 2.3x headroom, discovers it was an artifact, and stops at
its own first gate. The plan was to deepen the sparse MLA kernel's pipeline so
it tolerates C2C latency, targeting 300 GB/s against a 220 GB/s stop condition.
`NUM_K_BUFS` is 2 and cannot be raised: one K buffer is 72 KiB, and with Q and S
the kernel already sits at **224 KiB of Hopper's 227**, so every candidate had to
buy latency tolerance without buying shared memory. The cheapest such candidate,
an L2 prefetch of the next block's rows using indices the kernel already holds a
block ahead, was built and measured at **159.0 GB/s against a 157.9 baseline** -
nothing. A near-miss is worth more than the result: the first build compiled
cleanly and changed nothing, because the define was appended to
`VLLM_FLASHMLA_GPU_FLAGS` while `_flashmla_C` takes `COMPILE_FLAGS
${VLLM_GPU_FLAGS}`, so the `#if` silently evaluated false. It was caught only by
diffing SASS `CCTL` counts against a backed-up binary (10 vs 10; after the fix,
10 vs 22) - without that check the run would have been logged as "prefetching
does not help far memory", a false refutation. **Verifying the instruction
reached the binary belongs inside the rebuild loop.** The premise then fell:
224 KiB of shared memory admits one CTA per SM, so the kernel runs **132 CTAs at
19% warp occupancy**, and at matched parallelism a plain gather gets **73 GB/s**
where the kernel gets 159 - it is already **2.2x better** than a straightforward
gather, not 2.3x worse than achievable. Worse for the idea, higher occupancy
makes the ratio *worse*: HBM keeps scaling while Grace saturates near 385, so
HBM/Grace goes from 1.5x at 132 CTAs to 6.5x at 2,112. Reaching 382 GB/s needs
~1,056 CTAs, i.e. ~28 KiB/CTA - a different kernel, not a modification. All
source changes were reverted and the pre-experiment binary restored; the work
survives as `kernel.patch` and `cmake.patch`. See the
[sparse MLA kernel attempt](experiments/2026-08-06-mla-grace-kernel/README.md).

In flight, undocumented: `experiments/2026-08-01-marlin-grid-fit` holds raw
sweep results from jobs 1197398, 1197614, 1197769 and 1198412 with no report and
an empty `analysis/`, and is not committed. Its `policy-1197614.json` puts the
shipped 264/132 hot/cold grid at 93.67 us against 129.91 us for the pre-fix
production launch, with the best alternative found (264/99) at 92.20 us, 1.6%
better and covering 51.6% of the c1 operating point. Read that as provisional.

Phase 42 reopens the speculator-width lever for DFlash2 and ports it. Phase 28
refuted DFlash1 at verify width 16 and DSpark at 9, and the refutation was
about width: this target's routed MoE is only 35% fixed weight streaming, so
verify cost grows with block width while acceptance grows sublinearly.
**DFlash2 is width 8, half of what was refuted, and Phase 28's own three points
fit `tok/s = acceptance / step_time` to within 1.4%,** which turns the question
into arithmetic. Width 8 brackets at 31.2 ms (holding draft cost at the 1.06
ms/token routed-MoE slope, defensible because DFlash2's draft is 6 SWA layers
with 8 KV heads against DFlash1's 5 full-attention layers with 64) and 39.0 ms
(interpolating Phase 28's measured curve). Break-even acceptance against MTP3
is therefore 3.35 to 4.19, and DFlash2 publishes 4.19 on its worst task and
6.02 on its best. Every remaining unknown collapses into one measurable
quantity: whether a drafter trained on GLM-5.3 hidden states retains acceptance
against this 5.2 W4G64 target. GLM-5.3 is the same base model as 5.2 with
different post-training, and the compatibility audit clears every static
mismatch, including the mask token — DFlash2's `mask_token_id` 154856 is a
reserved embedding row in **both** tokenizers, which take no gradient and are
therefore the same tensor on both models. Upstream carries DFlash2 in exactly
two commits (#52816, #53435), both after this fork's 2026-08-01 base; six of
the seven cherry-pick conflicts are upstream drift in files this fork does not
modify, and the seventh is this branch's own DCP support for draft attention,
which is upstream PR #48392 and still open there. **The answer is no.** DFlash2 measures **2.6823** acceptance at 32.85 ms
against MTP3's pooled 3.0721 at 27.69 ms, making it **26.4% slower**, and
break-even at its achieved step time was 3.6446. It reproduces the exact
deterministic completion, so this is a speed verdict and not a correctness
one. Two things are worth separating. The cost model held: width 8 was
bracketed at 31.2-39.0 ms before the run and measured 32.85, so **width 8 is
survivable on this target** and Phase 28's refutation should be read as
bounding DFlash1 at 16 and DSpark at 9, not width in general. What failed is
transfer: DFlash2 publishes 4.19-6.02 against a GLM-5.3 target and retains
about half of that here, and it is already worse than MTP3 **at position 0**,
42% against 87%, which is the signature of a drafter reading hidden states
that post-training moved underneath it rather than one decaying with block
depth. Reaching the number took five source blockers, four of them rederived
from the reverted Phase 28 work and one -- a compile-range conflict in the
candidate selector -- that Phase 28 never saw because its drafts had no
selector. Committed as `4349240546` on `dflash2-backport`, off the known-good
base, with the production branch untouched. See the
[DFlash2 backport](experiments/2026-08-28-dflash2-backport/README.md).

Phase 43 brings GLM-5.3 up on the tiered path and answers the fork-quality
question without the FP8 reference it set out to build. The goal was
near-lossless FP8 reference logits so every quantisation could be scored
against ground truth by KL rather than against another approximation. **That
was not achieved**: ten configurations across the fork, upstream 0.27 and
upstream 0.28.0 failed to serve GLM-5.3 FP8 at all -- PP3xTP4 hangs after
compile, TP4xDP4/EP16 dies in `DPMoEEngineCoreActor.__init__`, TP16 deadlocks
in c10d rendezvous with all 16 peers alive. The failures are recorded so the
next attempt does not rediscover them. Three independent tests answered the
underlying question anyway: KL between the tiered and non-tiered paths on the
same fork and checkpoint is 0.151 against a 0.144 noise floor; moving 2,800
experts between HBM and Grace costs 0.59pp on GSM8K, so **placement is
quality-neutral**; and GLM-5.2 AutoRound scores 95.12% on today's HEAD against
95.51% in July, so ~1,700 commits introduced no regression. The **~4pp GSM8K
drop from 5.2 to 5.3 is real** (95.51% -> 91.21%) and three explanations are
eliminated -- not the tiered path, not placement, and not the quantiser, since
Inferact's GPTQ-calibrated NVFP4 scores no better than incoai's RTN. The model
card claims 5.3's gains are coding, agentic and cyber, and GSM8K is not among
them, so post-training is the leading remaining explanation. One finding here
outlived the phase: **DFlash2 is not output-lossless on this stack**, 88.93%
against 91.21% target-only with nine times the run-to-run spread, and
speculative decoding at temperature 0 must reproduce greedy output exactly.
Phase 46 later found the spread itself is not DFlash2's -- greedy decoding is
not reproducible on this server at all. See the
[GLM-5.3 FP8 attempt](experiments/2026-08-29-glm53-fp8/README.md) and the
[NVFP4 tiered bring-up](experiments/2026-08-28-nvfp4-tiered/README.md).

Phase 43 also reopens DFlash2, and this is the part the index previously
buried inside an NVFP4 bring-up report. Phase 42 refuted DFlash2 **on the 5.2
target**, and its own diagnosis was transfer: a drafter trained on GLM-5.3
hidden states, already worse than MTP3 at position 0 (42% against 87%). On a
5.3 target that objection disappears, so `incoai/GLM-5.3-DFlash2` was rerun
against the 5.3 NVFP4 checkpoint -- and gave 3.323 acceptance at width 8 where
the card claims 5.94.

Replicating the card's own protocol (GSM8K, chat template, T=1.0, top_p 0.95,
natural EOS, 4096 max tokens, 64 samples) localised the fault rather than
arguing about it. **MTP reproduced the card to within 3%** -- 4.967 against
5.12 -- which validated sampling, chat template, EOS handling, prompt
distribution, target model and the whole tiered stack in one measurement, and
left the problem specific to DFlash2, which managed 2.815 on the same
protocol. Normalised for width the diagnosis sharpened: our MTP3 sat at 68.8%
of ceiling against the card's MTP7 at 64.0%, while DFlash2 at the *same* width
8 managed 41.5% against the card's 74.2%. The per-position profile then named
the mechanism -- acceptance fell geometrically at a near-constant 0.714 ratio
(stdev 0.020), the signature of plain autoregressive decay, when DFlash2's
entire premise is that its two-tap convolutions and candidate selector prevent
exactly that.

**Root cause: the backport sat on a base predating its own prerequisites.**
DFlash2 landed upstream 2026-08-20; this fork's base is 2026-07-16, and
cherry-picking DFlash2 alone left it running on a DFlash core missing five
weeks of its own bugfixes. Auditing all 1,704 upstream commits since the base
-- by message, by path, and by content presence, since cherry-picks change
hashes and ancestry is useless -- found five candidates, of which three were
missing and applied:

| PR | was present | effect |
| --- | ---: | --- |
| #53336 / #53002 | 50% | FlashAttention metadata built from the **target's** head geometry, not the draft's |
| #51256 | 0% | DFlash needs K extra scheduling slots; the budget reserved none |
| #44492 | 28% | draft `seq_lens_cpu_upper_bound` not populated |
| #50065 | 0% | ruled out on inspection: `max_query_tokens` equals the cudagraph capture size, so the fix's `max()` is a no-op |
| #48524 | 50% | `fc` sizing; ours is verified correct at 36,864 |

#53336 is the substantive one: the target is MLA with head_dim 192 and
effective head size 576 while the DFlash2 draft is dense with head_dim 128, 64
query heads and 8 KV heads, so describing draft attention with target geometry
degrades the draft without touching output -- verification rejects every bad
token -- and it also explains the separate c4 failure `scheduler_metadata must
have shape (metadata_size)`. The three fixes are on this branch as
`dcf5ceb2b9` and `c8a723e0c9`, and they **recovered 2.815 to 4.029, +43%,
while leaving MTP unchanged at 4.912 against its pre-fix 4.967** -- surgical,
by the only test that could show it.

**A 32% gap to the card remains and DFlash2 is still not qualified.** Two audit
entries deserve re-examination before it is blamed on the card's
GB300/FlashAttention-4 configuration: #48524, because DFlash2 has 6 hidden and
6 target layers so a wrong code path still yields a right-sized tensor, and
#50487, which changes which hidden state is tapped as the aux input and was
dismissed on its Kimi-K3 title alone. What has **never been measured is the
comparison that decides it**: matched MTP3-against-DFlash2 throughput on a 5.3
target. Phase 42 priced break-even at 3.65 acceptance at DFlash2's achieved
32.85 ms/step, and DFlash2 now measures 4.029 -- but on the card's GSM8K
protocol rather than Phase 42's harness, so those two numbers **cannot be
subtracted**. They are a reason to run the matched experiment, not a result.

Phase 44 re-derives the GLM-5.3 hot-expert ranking. Every 5.3 placement
profile shipped so far carried GLM-5.2's ranking as an acknowledged placeholder:
5.3 shares 5.2's base model but its post-training moved the router, and
placement is quality-neutral, so the stale ranking cost throughput silently.
A capture host on the NVFP4 checkpoint (V1 runner, no DCP, verification size 4
-- the same three constraints the 2026-07-26 GLM-5.2 capture established)
recorded **129,392 routed positions from 377 agentic coding traces** driven by a
real tool-calling agent loop over 16 tasks against this tree. On the held-out
split the shipped ranking scores 0.4047 cold-hit against the re-derived 0.2290,
with a linear/even placement at 0.4977 -- **the GLM-5.2 ranking was recovering
about a tenth of the gap between no ranking and a correct one**, and 37% of the
resident set sat on the wrong experts. The residency strategy is not the cause:
frequency, tail-aware, and layer-concentrated residency all converge to the same
hot set. A strict task-family split (train 9, evaluate 7 unseen, no conversation
crossing the split) reproduces the win on a third less data with 0.918 hot-set
overlap, so this is GLM-5.3's router under coding traffic rather than an
artefact of the task list. See the
[GLM-5.3 routing capture](experiments/2026-08-29-glm53-routing-capture/README.md).

Phase 45 starts an AutoRound W4G64 quant of GLM-5.3 and halts it. The 5.2
production target is `GLM-5.2-AutoRound-W4G64-MTP`, so a matching 5.3 quant
would hold the quantiser fixed across the 5.2/5.3 comparison, and 5.3 had no
AutoRound release. The recipe was recovered from the 5.2 checkpoint's own
`quantization_config` rather than guessed -- bits 4, group 64, sym, batch 2,
grad-accum 4, 512 samples, auto-round pinned to 0.14.0 so its defaults fill in
the five parameters the config does not record. `zai-org/GLM-5.3-BF16` (753.3B
BF16, published 2026-08-28) is the right source: the main 5.3 repo is FP8, so
quantizing from it would have made the FP8 release the quality ceiling. What
gets quantized was confirmed against the 5.2 weight index rather than inferred
-- only routed expert MLPs carry `qweight`/`qzeros`/`scales`; the router,
shared experts, attention, the three dense layers and `eh_proj` stay BF16, and
the MTP head is excluded. **Halted before the quantization job ran**, on the
judgement that Intel would publish a 5.3 release within days. Two operational
notes survive: `--layer_config` takes JSON on the command line rather than a
path, and auto-round must be installed in an isolated venv -- installing it
into the shared vLLM environment downgraded transformers under running jobs.
See the [AutoRound attempt](experiments/2026-08-29-glm53-autoround/README.md).

Phase 46 promotes a community W4A16 checkpoint to the c4 production launcher
and, in qualifying it, finds an unrelated defect that is still open.
`JANGQ-AI/GLM-5.3-W4A16` is compressed-tensors pack-quantized int4 at **group
32** -- 282 shards, 420 GB, 176,321 tensors -- and four things blocked the
load. Three were guards stricter than the kernel: `tiered_moe_manifest.py`,
`tiered_moe_conversion.py` (twice) and `compressed_tensors_moe_wna16_marlin.py`
each hardcoded `group_size == 128`, while Marlin's own
`SUPPORTED_GROUP_SIZES` is `[-1, 32, 64, 128]`, `_w2_scale_sharding` branches
only on `actorder`, and `runtime_expert_bytes` is derived from stored tensor
sizes so a smaller group is accounted automatically. All three now accept
Marlin's set. The fourth is a real publisher bug: the index declares
`total_size` 21,739,848 bytes above the summed tensor headers, tripping the
truncation guard. The checkpoint is *not* truncated -- 176,321 tensors in the
index, 176,321 on disk, every shard byte-exact against the hub manifest -- so
the local metadata is corrected and the published original kept at
`index-original.json`, rather than weakening a guard that is doing its job.

**W4A16 g32 beats NVFP4 g16 by 9.77% +/- 3.32%** on the real-code decode suite
(512-token prompts, 1024-token generations, c4/DCP4/MTP3): 213.1 +/- 1.4 against
194.2 +/- 4.6 tok/s aggregate, 18.77 against 20.60 ms TPOT, same sign in all
three paired runs. TTFT is unchanged at 477 against 479 ms, so the win is
decode. Three paired runs were necessary, not cautious: between-run spread on
this configuration is about 2.7%, wider than any within-run confidence
interval. Per-expert cost is within 8 bytes of NVFP4 -- int4 with bf16 scales
at group 32 costs what fp4 with fp8 scales at group 16 does -- and the real
difference is 13.9 GB less non-routed weight (43.2 against 57.1 GB), because
this checkpoint quantizes the MTP block where NVFP4 leaves it BF16. An earlier
claim that this 3.5 GB per rank *caused* the speedup is **retracted**: measured
HBM was identical at 63.97 GiB on both arms, so the saving showed up as spare
capacity, not throughput, and the mechanism behind the 9.77% is unattributed.

Spending that capacity is where the phase went wrong, and the correction is
**2026-08-30**. This entry originally read that raising `gpu-memory-utilization`
from 0.90 to 0.94 "buys 96 more hot experts per rank, taking held-out cold-hit
from 0.2290 to 0.2131", and that "more hot experts is not monotonically better"
because 2400 / 2496 / 3000 slots measured 213.1 / 214.8 / 205.2 tok/s.

**Both claims are withdrawn. A profile's slot count does not control
residency.** `tiered_moe_planner.py:368` computes

```python
hot_slots = min(primary_slots, available_hbm // manifest.runtime_expert_bytes)
if hot_expert_ids_by_layer is not None and envs.VLLM_TIERED_MOE_PROFILE_CAP:
    hot_slots = min(hot_slots, profile_slots)
```

`VLLM_TIERED_MOE_PROFILE_CAP` defaults to 0 and **nothing sets it -- not the A/B
arms, not the production launcher**. So the number of resident experts is set by
whatever HBM is left, and the profile's list is then padded up
(`_promote_underfilled_residency`) or trimmed down
(`_demote_overfilled_residency`) to match. Only the *ranking* survives.

The three arms therefore ran at **identical residency** and differed only in the
ranking their profiles carried, since `optimize_routing_profile.py` builds a
different greedy top-N and owner assignment for each `--hot-slots-per-rank`.
The 213.1 / 214.8 / 205.2 spread is a ranking comparison, and the sweep never
tested the question it was designed for. The 96 extra experts were never
resident, so the 0.2131 cold-hit was never realised either.

Buying HBM residency remains **untested** on this checkpoint. Testing it needs
`VLLM_TIERED_MOE_PROFILE_CAP=1`, or a lever that actually moves
`available_hbm`.

**The open defect: prefix caching degrades the first pass under MTP.** On
GSM8K at temperature 0 and seed 42, with identical questions, MTP3 with prefix
caching scores 58.6% and 56.3% on the first pass where MTP3 without prefix
caching scores 91.4% and no-speculator with prefix caching scores 90.6%.
Warmup is not the explanation (58.2% without it). It reproduces on both
checkpoints and predates the group-32 work, so it is pre-existing rather than
introduced here. **The obvious next test is blocked.** Speculative decoding at
temperature 0 must reproduce greedy output exactly, so the natural check is to
diff MTP against no-spec completions -- 63 of 64 differ. But the control diffs
two passes of the *same* server against each other and gets 60 of 64, so
greedy decoding is not reproducible on this server at all, and the diff test
cannot separate the two. That non-reproducibility is itself a defect, and it
is the same run-to-run spread Phase 43 attributed to DFlash2. Closing this
needs logit-level instrumentation -- asserting the target's argmax equals each
accepted token -- which is an engine patch, not a benchmark.

Two operational findings. Grace memory was characterized at peak across all
four NUMA nodes: minimum free is 7.7-7.9 GiB per node, and the ~69 GiB per node
reported as shared memory is the pinned UVA expert tier rather than files
(`/dev/shm` holds 377 MB), which `numastat` confirms as 72.6 GiB private per
worker on its own NUMA node. Because the tier is allocated with
`pin_memory=True` it is not reclaimable, so conservative replica headroom is
about 390 per rank with node 3 consistently tightest. Separately,
`jupiter-env.sh` points `XDG_CACHE_HOME` at GPFS and neither it nor the
launchers set the inductor or triton cache directories, so every torch.compile
artefact was landing on the filesystem that makes small-file work pathological;
all three are now pinned to fscratch alongside the vLLM and TRT-LLM caches.
See [GLM-5.3 W4A16](experiments/2026-08-29-glm53-w4a16/README.md).

Phase 47 stands up a capture host for deriving the hot-expert ranking from
**real Claude Code usage** rather than Phase 44's synthetic driver. The driver's
16 scripted tasks were written to look like agentic coding; real turns differ in
ways that plausibly move the routing distribution -- 100k+ contexts instead of a
few thousand, a real system prompt and real tool schemas in every prefill, long
tool-result spans, and the actual mix of reasoning, code, prose and JSON tool
arguments. The 2026-07-26 GLM-5.2 live capture recorded **154 requests in about
an hour** of ordinary use, with responses up to 2,230 tokens, so an afternoon of
normal work yields more real data than the entire synthetic run.

The host is the prod c4 launcher plus `--enable-return-routed-experts`, and
three deviations from prod are forced by the implementation rather than chosen.
`vllm/v1/worker/gpu/model_runner.py` -- the V2 runner -- contains no
`routed_experts` support at all, so `gpu_worker.py`'s
`init_routed_experts_capturer()` cannot run on it; `Scheduler.__init__` asserts
`dcp_world_size == 1`; and dropping DCP then forces `max_num_seqs` to 1, because
`config/vllm.py:2325` rejects more under DCP1 ("the replicated 400K MLA cache
does not fit more than one sequence per rank") and `:2337` pins `max_model_len`
to exactly 400000, so the context cannot be shortened to buy headroom. **The
capture host therefore serves one request at a time** -- Claude Code's parallel
subagent and title requests queue behind the foreground turn. None of the three
changes which experts the router picks, so a ranking derived here transfers to
the prod DCP4/V2 host; what they cost is throughput on the capture host.

Verified end to end. The concern was that Claude Code streams and the streaming
path calls `_record_routing_trace` per chunk, which would have written a file
per chunk; it writes **one `.npy` per request**. A 300-token streamed response
produced a single `[400, 78, 8]` uint8 trace -- 400 rather than 300 because
rejected MTP draft positions are recorded too -- with 217 distinct experts on
layer 40 alone, so the routes are real rather than the 0..7 identity fallback
the pipeline filters. `routed_experts_prompt_start` is set to `len(prompt) - 1`
in the serving layer, so traces begin at the last prompt token: generation
positions only, a conversation prefix is never counted twice across turns, and
only expert IDs reach disk -- no prompts, responses or tool arguments.

Not yet decided: whether to source the ranking from live capture or by replaying
the 41 Claude Code transcripts already on disk. Replay keeps DCP4 for real work
and draws on a larger corpus, at the cost that the assistant turns in those
transcripts were authored by whichever model ran the session. See the
[real-usage routing capture](experiments/2026-08-30-glm53-cc-capture/README.md).

## Reproducing

The scripts expect this directory to be `agent_space/` inside the vLLM checkout
and the pinned model to be a sibling of that checkout under `models/`.

```bash
srun --jobid=<jobid> --nodes=1 --ntasks=1 --gres=gpu:4 \
  --overlap --cpu-bind=none --unbuffered \
  bash run-cpu-offload-baseline.sh
```

After the health endpoint is ready, run `run-batch1-baseline.sh` or the commands
recorded in the result JSON files. Do not run model downloads from Booster.

## Layout

- `baseline/`: benchmark JSON, memory captures, reports, and diagnostic logs
- `jupiter-env.sh`: module, virtualenv, cache, and offline settings
- `run-cpu-offload-baseline.sh`: baseline server configuration
- `run-batch1-baseline.sh`: batch-one benchmark cases
- `benchmarks/`: focused hardware and kernel microbenchmarks
- `profiles/`: versioned physical machine profiles used by plan-only
- `cc-plugins/`: Claude Code plugins for this work (see `cc-plugins/vllm-tps`)
- `experiments/`: dated raw results and experiment notes
- `gh200-vllm-w4a16-tiered-moe-plan-v2.md`: implementation plan beyond baseline

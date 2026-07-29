# JUPITER GLM-5.2 vLLM worklog

Experimental worklog for serving the 361.06 GiB
`lowbitcoffee/GLM-5.2-W4A16` checkpoint on one JUPITER Booster node. The goal
is to improve batch-one decode at up to 400K context by keeping hot MoE experts
in HBM and executing colder experts from coherent Grace memory through CUDA
UVA. The current 37.5 tok/s result is the baseline; the project target starts
at 100 tok/s.

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
See the [tier-overlap diagnosis](experiments/2026-07-25-tier-overlap/README.md)
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
- `experiments/`: dated raw results and experiment notes
- `gh200-vllm-w4a16-tiered-moe-plan-v2.md`: implementation plan beyond baseline

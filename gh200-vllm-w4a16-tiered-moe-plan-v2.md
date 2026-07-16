# vLLM W4A16 Tiered-MoE Integration for GLM-5.2 on One JUPITER GH200 Node

## Document Status

This is the v2 implementation plan for serving GLM-5.2 W4A16 at batch-one decode on one
four-GH200 JUPITER Booster node. It supersedes
`gh200-vllm-w4a16-tiered-moe-plan.md` without modifying that file or its review.

The inspected vLLM base is:

```text
26ff616bbf43c5c5ecb847589705cebdbff46706
```

The selected checkpoint is an existing Hugging Face artifact, not a checkpoint-production
workstream:

```text
repository: lowbitcoffee/GLM-5.2-W4A16
revision:   55c92ae85b7ec564c94634964b6f5efe5c09a844
format:     Compressed-Tensors pack-quantized safetensors
weights:    symmetric INT4, group size 128, static activation order
compute:    BF16 activations
size:       approximately 388 GB in eight model shards
```

The pinned model card reports `GlmMoeDsaForCausalLM`, 78 layers, 256 routed experts plus one
shared expert, top-8 routing, and native vLLM serving. Its config declares
`quant_method=compressed-tensors`, `num_bits=4`, `group_size=128`, `symmetric=true`, and
`format=pack-quantized`. The safetensors index contains per-expert `weight_packed`,
`weight_scale`, and `weight_shape` tensors. These facts make it the primary artifact, but
community-published quality results are not accepted as our correctness gate.

v2 incorporates the review corrections:

- Checkpoint production is out of scope; immutable download, mirroring, format inspection,
  and independent quality qualification are explicit Phase 0 work.
- The vLLM route is primary. llama.cpp v4 is a documented contingency rather than a second
  active implementation.
- A new pageable-Grace-memory CUDA view is mandatory. The existing vLLM helper copies
  non-pinned input into a new `cudaHostAllocMapped` buffer and cannot implement zero-copy
  LPDDR placement.
- KV block-count sizing, profiling, and allocation all understand the physical cache tier.
- Batch-one Marlin and UVA timing use measured complete-layer latency, never byte floors.
- Grace- and HBM-cache fallback scenarios are priced, but pre-launch estimates are not
  described as achieved TPS.
- The GLM path asserts ordinary `MLAAttentionSpec` with the 656-byte V3.2 layout and rejects
  accidental DeepSeek-v4 or sliding-window cache selection.
- Grace CPU/SVE execution remains a conditional follow-on, not a required deliverable.

## Architecture Decision Record

### Decision

Use vLLM W4A16 tiered MoE as the primary implementation.

### Rationale

vLLM already supplies the hardest generic components:

- `GlmMoeDsaForCausalLM` and sparse DSA/indexer execution;
- FP8 sparse MLA cache kernels;
- TP4-derived EP4 ownership and distributed reductions;
- Compressed-Tensors W4A16 loading and Marlin MoE;
- CUDA graphs, torch.compile integration, serving, scheduling, and observability;
- EP weight filtering, expert maps, and worker NUMA binding.

Compared with llama.cpp v4, the selected checkpoint is about 67 GB smaller, removes the
mixed Q4_K/Q6_K kernel matrix, and avoids implementing a new DSA graph and distributed
runtime.

### Contingency

Retain llama.cpp v4 as a dormant contingency. Return to it only if at least one of these is
demonstrated and cannot be fixed within the project budget:

- the pinned W4A16 checkpoint fails independent quality gates;
- native vLLM GLM DSA fails semantic qualification;
- pageable Grace memory cannot be exposed safely to CUDA kernels;
- neither existing nor specialized W4A16 CUDA kernels can execute cold experts acceptably;
- vLLM release churn makes a maintainable pinned fork impossible.

Do not develop both paths concurrently after Phase 0.

## Target and Scope

- One exclusive JUPITER Booster node with four GH200 Superchips.
- One active request, batch-one autoregressive decode.
- Populated 400,000-token context.
- TP=4 for non-routed computation and EP=4 for routed experts.
- Each rank owns 64 of 256 routed experts per MoE layer.
- Per-expert HBM or paired-Grace-LPDDR residency.
- GPU UVA execution for cold experts in the required path.
- FP8 DSA indexer cache in HBM.
- FP8 main MLA cache in paired LPDDR when its measured sparse-access path wins; HBM fallback.
- At least 5 GB decimal planned HBM reserve and 8 GB decimal LPDDR reserve per rank.
- No expert-sized weight copy during decode.
- Minimum 100 TPS, primary 150 TPS, conditional 200 TPS stretch.

Initial non-goals are multiple concurrent sequences, throughput all-to-all EP, MTP,
speculative decoding, dynamic EPLB, runtime expert migration, context parallelism,
inter-node serving, and checkpoint quantization.

## Existing vLLM Reuse and Gaps

### Reused Without Redesign

- `model_executor/models/registry.py`: GLM DSA registry entry.
- `model_executor/models/deepseek_v2.py`: GLM DSA model implementation.
- `model_executor/layers/sparse_attn_indexer.py`: fused indexer and top-k.
- `v1/attention/backends/mla/flashmla_sparse.py`: sparse MLA and `fp8_ds_mla`.
- `v1/attention/backends/mla/indexer.py`: indexer metadata/cache.
- `fused_moe/config.py`: TP-derived EP configuration.
- `fused_moe/routed_experts.py`: routed expert lifecycle.
- `fused_moe/expert_map_manager.py`: global/local mappings.
- `fused_moe/experts/marlin_moe.py`: W4A16 Marlin execution.
- Compressed-Tensors WNA16 Marlin conversion/loading.
- vLLM multiprocessing executor, TP collectives, scheduler, serving, and graphs.
- `utils/numa_utils.py` and CUDA NUMA discovery.

### Required Extensions

1. Per-layer ownership maps beyond linear/round-robin.
2. Two compact expert stores per rank: hot HBM and cold host-UVA.
3. Destination-aware streamed Marlin conversion/loading.
4. Hot/cold expert ID partition and dual-kernel result join.
5. Direct pageable-LPDDR CUDA tensor views without registration or copies.
6. Main MLA cache placement in host-UVA memory.
7. Tier-aware KV capacity calculation and profiling.
8. Sequence-aware routing traces and offline owner/residency optimization.
9. JUPITER machine profiles and critical-path prediction.

The generic `UVAOffloader` is not used for routed experts. It operates at parameter
granularity, while a fused MoE parameter contains many experts, and post-load Marlin
conversion can replace its storage.

## Checkpoint Acquisition and Qualification

### Immutable Acquisition

Download only the pinned revision:

```bash
hf download lowbitcoffee/GLM-5.2-W4A16 \
  --revision 55c92ae85b7ec564c94634964b6f5efe5c09a844 \
  --local-dir <project-model-path>/GLM-5.2-W4A16-55c92ae
```

Do not serve directly from a mutable Hugging Face cache reference. Mirror the complete
revision to JSC project storage. Record:

- repository and commit SHA;
- every filename, byte size, ETag/LFS/Xet identifier, and SHA-256 checksum;
- `config.json`, tokenizer, generation config, chat template, and index checksum;
- vLLM, Transformers, Compressed-Tensors, PyTorch, CUDA, and Marlin revisions.

After download, production uses the local immutable path and refuses missing or changed
shards.

### Required Schema Checks

Before allocating model tensors, verify from config and safetensors headers:

```text
architecture       GlmMoeDsaForCausalLM
model_type         glm_moe_dsa
quant_method       compressed-tensors
format             pack-quantized
weight type        int
bits               4
group size         128
strategy           group
symmetric          true
actorder           static
activation quant   none
```

Verify expert tensors have packed weight, scale, and shape entries for all three projections
of all 256 experts in routed layers 3-77. Inspect actual tensor dtypes, shapes, offsets, and
shard ownership. The large config `ignore` list is not interpreted as proof that experts are
unquantized; the safetensors inventory and selected vLLM quant method are authoritative.

Reject AWQ, AutoRound/GPTQ variants with different packing, W4AFP8, NVFP4, MXFP4, MLX, GGUF,
and SGLang-specific layouts under this descriptor. They may be qualified later through new
format descriptors and new memory math.

### Independent Quality Gate

The model card reports parity with FP8 on small GSM8K/MMLU samples. Treat that as provenance,
not acceptance evidence. Compare the pinned artifact against official GLM-5.2-FP8 on a
temporary 16-GPU reference deployment:

- operator dequantization and layer output parity;
- logit KLD below 2,048 tokens, where the indexer selects all valid positions;
- greedy agreement where logits are stable;
- GSM8K, MMLU, coding, tool-use, and bilingual prompts;
- 4K and 32K retrieval tests; 128K if allocation permits;
- long-position DSA invariants and selected 400K retrieval probes.

Define thresholds before running the evaluation. A failing artifact triggers evaluation of a
different Hugging Face W4A16 candidate through a separate descriptor, not local requantization
inside this project.

## Hardware and Locality Contract

Each worker owns one complete Grace-Hopper pair:

| Per rank | Planning value |
|---|---:|
| Grace cores | 72 Neoverse V2 |
| Grace LPDDR5X | 120 GB, approximately 512 GB/s |
| Hopper HBM3 | 96 GB, use 3.5 TB/s conservatively |
| NVLink-C2C | 450 GB/s/direction |
| Hopper-to-Hopper NVLink | 150 GB/s/direction |
| Hopper BF16 throughput | 630 TFLOP/s |

One vLLM worker process is CPU- and memory-bound to the NUMA node paired with its GPU.
Startup records CUDA UUID, PCI/NVLink topology, CPU set, NUMA node, memory capacities, peer
matrix, ATS/pageable host-page-table attributes, and measured local/remote bandwidth.

Strict mode rejects ambiguous pairing or large allocations with <95% local pages or >=5%
remote pages. Remote Grace memory is never an overflow tier.

## W4A16 Memory Model

### Routed Experts

One expert contains three 6,144 x 2,048 matrices:

```text
parameters             = 37,748,736
INT4 data               = 18,874,368 bytes
BF16 G128 scales        =    589,824 bytes
nominal expert total    = 19,464,192 bytes = 18.5625 MiB
```

The actual plan adds Marlin padding and metadata from converted tensors.

```text
routed/node             = 373.7124864 GB
routed/rank             =  93.4281216 GB
active routed/token     =  11.6785152 GB/node
```

The pinned checkpoint's approximately 388 GB published size agrees with the 4.125-bit
mathematical floor, but exact local files and tensor headers replace that estimate.

### Native vLLM Caches at 400K

GLM must construct ordinary `MLAAttentionSpec`, `model_version != deepseek_v4`, with the
V3.2 `fp8_ds_mla` layout:

```text
main MLA cache/rank     = 656 x 400000 x 78
                        = 20.4672 GB

indexer cache/rank      = 132 x 400000 x 21
                        = 1.1088 GB

sparse main reads/token = 656 x 2048 x 78
                        = 104.792064 MB/rank
```

Reject `SlidingWindowMLASpec`, the 584-byte DeepSeek-v4 layout, BF16 main cache, or indexer
allocation on shared indexer layers in production mode.

### Preliminary Fit

Before exact inventory:

```text
checkpoint share       388 / 4           = 97.0 GB
replication correction                     2.0 GB
indexer HBM cache                          1.109 GB
HBM reserve                                5.0 GB
HBM capacity                              96.0 GB
-------------------------------------------------
preliminary cold pressure                  9.1 GB
```

This is approximately 470 expert slots or 9.8% of local routed slots with the main MLA cache
in LPDDR. The exact planner must account for BF16 indexer/router/lm-head tensors identified by
the checkpoint, TP replication, Marlin conversion, CUDA graphs, and workspaces.

Putting the 20.4672 GB main cache in HBM raises cold pressure to roughly 29.6 GB, about 1,520
nominal slots or 31.7% capacity-cold. Capacity-cold fraction is not routing cold-hit rate.

## Runtime Placement and Data Movement

### HBM

- TP-local non-routed weights and replicated tensors.
- Compact hot routed experts with scales and metadata.
- Shared experts under native TP placement.
- The 21-full-layer indexer cache.
- Main MLA cache only for the HBM fallback.
- Activations, maps, Marlin/attention workspaces, graph buffers, and NCCL state.

### Paired Grace LPDDR

- Compact cold routed experts with scales and metadata.
- Main MLA cache when host-UVA is selected.
- CPU owner tensors for CUDA aliases, trace staging, and allocator metadata.

Weights move once during load into their final tier. During decode:

- hot weights are read from HBM by Marlin;
- cold weights remain in LPDDR and are read by the GPU over C2C;
- main cache rows are written/read directly through their selected pointer;
- activations and all expert intermediates remain in HBM;
- only vLLM's normal hidden/result collectives cross GPU NVLink.

No expert is duplicated across HBM and LPDDR after startup. No full expert is prefetched or
staged per layer.

## Mandatory Pageable Grace CUDA View

The existing `get_cuda_view_from_cpu_tensor()` is unsuitable for non-pinned storage: its
current CUDA implementation allocates `cudaHostAllocMapped`, copies the tensor, and returns a
view of the new pinned buffer. With pinning disabled this violates zero-copy and doubles
storage.

Add a dedicated GH-capability-gated operation, for example:

```text
torch.ops._C.get_cuda_pageable_view_from_cpu_tensor(cpu_tensor, cuda_device)
```

It must:

1. Require CUDA pageable memory access and host-page-table support.
2. Accept a contiguous, aligned, NUMA-local ordinary CPU allocation.
3. Create a non-owning CUDA tensor over the same virtual address without host registration,
   allocation, or copy.
4. Preserve sizes, strides, dtype, device ordinal, and CPU-owner lifetime.
5. Reject unsupported platforms and pointers with an actionable error.
6. Synchronize destruction so no graph/stream uses freed host pages.

Add a `GraceAllocation` owner containing CPU tensor, CUDA alias, bytes, NUMA node, page
placement, and allocation kind. Use it for both cold experts and host MLA cache.

Worker launch uses `--numa-bind`; allocation then applies local memory policy, first-touches
from the bound worker, requests huge pages, and audits pages with `move_pages` and
`/proc/self/numa_maps`.

## Tiered Routed-Expert Implementation

### Storage

Each routed layer owns:

```text
hot bundle:
  compact w13/w2 packed weights, scales, optional metadata in HBM
  global_to_hot_local[256]

cold bundle:
  compact w13/w2 packed weights, scales, optional metadata in LPDDR
  CUDA aliases of identical storage
  global_to_cold_local[256]

common:
  global_to_owner[256]
  local_to_global maps
  exact byte counts and placement fingerprint
```

Every component of one expert shares one tier. Do not put packed weights in LPDDR while
leaving scales or permutation metadata in HBM.

### Loader

Add a safetensors manifest pass before allocation. It computes exact TP, replicated, expert,
cache, conversion-scratch, and workspace bytes. Placement is finalized before reading model
payloads.

Extend `ep_weight_filter.py` and `ExpertMapManager` with layer-aware static maps. Non-owned
expert tensors are skipped at the checkpoint iterator.

For each owned expert:

1. Read packed checkpoint slices and scales.
2. Convert/reorder to Marlin format with at most one expert of bounded GPU scratch.
3. Write the final hot result to compact HBM, or copy the final cold result once to its final
   local-LPDDR slot.
4. Store every associated tensor in the same tier.
5. Release source and scratch before the next expert.

Never materialize 64 local experts in HBM and then offload a subset. After load, eliminate
all duplicate tensors and reconcile observed bytes with the printed plan.

### Decode Dispatch

Use the existing router once. A CUDA remap operation maps global top-8 IDs to owner and
hot/cold local slots without CPU synchronization.

Hot and cold Marlin calls receive disjoint IDs and independent HBM workspaces. Hot work reads
HBM; cold work reads the LPDDR CUDA aliases. Run them on separate streams when measured
overlap helps. Join two rank-local partials before the existing shared/routed combination and
vLLM reduction.

Test existing Marlin `expert_map` `-1` behavior first. If it does not yield exact zero for
inactive tier entries, compact tier-specific IDs/weights in the remap kernel.

Pointers, maps, workspaces, and output shapes are static for CUDA graphs. If host pointers
cannot be captured, isolate only the cold operation behind events and retain graph capture
for the rest of the model.

## EP Ownership

For TP4, DP1, PCP1, DCP1, and no sequence parallelism, native vLLM should derive EP4 inside
MoE without token all-to-all. Phase 0 verifies the actual collective trace.

The tracer uses linear ownership:

```text
rank 0: experts 0-63
rank 1: experts 64-127
rank 2: experts 128-191
rank 3: experts 192-255
```

Production later permits arbitrary per-layer 64/64/64/64 maps. Because the hidden state is
mirrored, changing ownership adds no communication. Ownership remains static during a run.

## Tier-Aware KV Cache Integration

### Capacity Planning

Physical tier must be represented before vLLM calculates block count. Modify the KV planning
path, including `v1/core/kv_cache_utils.py::get_kv_cache_configs`, so each cache group has:

```text
memory_tier: HBM | HOST_UVA
capacity_domain: gpu_rank | grace_numa_node
bytes_per_block
planned_blocks
```

For a host main-MLA group:

- exclude its physical bytes from `gpu_memory_utilization` and available-HBM division;
- include its attention workspace and indexer group in HBM;
- calculate maximum blocks against the rank's LPDDR budget and 8 GB reserve;
- keep scheduler-visible block IDs/counts identical;
- require all four ranks to expose the same usable block count.

Profiling must not allocate a fake HBM main cache that distorts available-memory results.
Plan-only output shows HBM and LPDDR budgets separately.

### Allocation and Binding

Extend the worker allocation path in `GPUModelRunner`/the uniform cache allocator:

- `MLAAttentionSpec(fp8_ds_mla, non-v4)` main cache uses the selected tier;
- indexer cache remains HBM;
- all other cache groups keep native behavior;
- host allocation returns a CUDA alias with the same logical tensor shape/strides;
- block tables, slot mapping, cache append, and sparse attention receive no scheduler fork.

The ordinary vLLM CPU KV offload connector is not used; it copies blocks between tiers and is
the wrong semantic model for one permanently resident 400K cache.

## Full-Footprint Cache Qualification

Benchmark both HBM and local LPDDR over the full 20.4672 GB address range. One token performs
78 x 2,048 selected 656-byte row accesses, or 104.79 MB/rank. Use 21 distinct top-k position
sets reused by their shared-layer groups.

Test random, sorted, clustered, and sampled real-indexer patterns. Record p50/p95, effective
bandwidth, TLB/page behavior, cache writes, and end-to-end sparse kernel time.

Grace eligibility requires correctness, no large HBM gather, p95 <=0.5 ms/token/rank, and a
lower total trace simulation after expert displacement is included. Otherwise AUTO selects
HBM and replans experts.

## Trace and Placement Optimizer

Capture exact top-8 IDs with request boundaries:

```text
header: version, endian, checkpoint/model fingerprint, dimensions
request: request hash, first record, record count, domain label
counts: u64[75][256]
record: request index, context position, u16 selected_ids[75][8]
```

Capture on GPU into a bounded ring and drain asynchronously. Capture sampled 21-layer DSA
top-2048 position sets in a separate trace for cache benchmarking.

### Owner Optimization

Start from linear and round-robin maps. Use trace hyperedges and balanced swaps to partition
each layer into four sets of 64 while minimizing per-layer slowest-rank time. Validate on
held-out requests, not adjacent held-out tokens.

### Residency Optimization

For every trace token/layer/rank:

```text
rank_time  = measured schedule of selected hot/cold experts
layer_time = max(rank_time across four ranks)
token_time = sum(layer_time across 75 layers)
objective  = mean(token_time) + 0.25 * CVaR95(token_time)
```

Promote experts by marginal objective reduction per exact byte under independent rank HBM
budgets. The objective is non-submodular: audit lazy gains with periodic full recomputation
and finish with deterministic one-slot swap search. Alternate owner and residency improvement.

Report capacity-cold and routing cold-hit rates separately. Production placement is static.

## Performance Model

### Expert Floors

For one nominal 19,464,192-byte expert:

| Tier | Byte floor |
|---|---:|
| HBM at 3.5 TB/s | 5.56 us |
| C2C at 450 GB/s | 43.25 us |
| C2C at 350 GB/s | 55.61 us |

These are never used as final kernel timings. Batch-one Marlin may be dominated by launch,
occupancy, ID alignment, and two-stream join overhead. Benchmark complete layers with 0-8
selected local experts and feed measured distributions into the simulator.

### Preliminary Scenarios

The following use byte-floor routed times, a provisional 0.7 ms non-routed HBM term,
0.317 ms indexer scan, 0.233 ms Grace sparse-read floor or 0.030 ms HBM floor, and 1.56 ms
collectives. They are pre-launch sensitivity cases, not achieved throughput:

| Configuration | Routed critical | Pre-launch total | Floor TPS |
|---|---:|---:|---:|
| Grace cache, 9.9% routing-cold | 3.27 ms | 6.08 ms | ~164 |
| Grace cache, 5% routing-cold | 2.45 ms | 5.26 ms | ~190 |
| Grace cache, 3% routing-cold | 2.09 ms | 4.90 ms | ~204 |
| HBM cache, 31.7% routing-cold | 6.31 ms | 8.92 ms | ~112 |
| HBM cache, 15% routing-cold | 4.02 ms | 6.63 ms | ~151 |
| HBM cache, 10% routing-cold | 3.28 ms | 5.89 ms | ~170 |

The unprofiled Grace configuration has only about 0.6 ms margin to 150 TPS before omitted
overheads, so profiling is still expected for a reliable 150 TPS result. HBM fallback should
remain above the 100 TPS class, but 150 TPS likely requires a strong trace and low framework
overhead. Measured Marlin/cache/collective values replace every floor before committing targets.

## Configuration

Add dedicated configuration; reject simultaneous generic weight offload:

```text
--enable-tiered-moe
--tiered-moe-backend uva|cpu|auto
--tiered-moe-placement-profile PATH
--tiered-moe-routing-trace-output PATH
--tiered-moe-hbm-reserve-gb N
--tiered-moe-host-reserve-gb N
--tiered-moe-numa-strict
--tiered-moe-plan-only
--mla-cache-tier auto|hbm|host_uva
--grace-machine-profile PATH
```

Initial constraints: CUDA, pinned checkpoint revision, W4A16 descriptor, TP4, EP enabled,
DP1, PP1, PCP1, DCP1, max sequences 1, no EPLB/redundant experts, no generic
`cpu_offload_gb`, and no layer prefetch offload.

Production launch uses the immutable local checkpoint and `--kv-cache-dtype fp8_ds_mla`,
`--numa-bind`, `--enable-ep-weight-filter`, multiprocessing executor, and the tiered options.

## Delivery Sequence

### Phase 0: Artifact and Native-Path Qualification

1. Download/mirror the pinned Hugging Face revision and calculate checksums.
2. Inspect all config/index/header metadata and exact bytes.
3. Verify vLLM selects GLM DSA, Compressed-Tensors WNA16 Marlin, ordinary
   `MLAAttentionSpec`, and the 656-byte cache layout.
4. Run independent W4A16 quality evaluation against official FP8.
5. Trace TP4+EP4 collectives and confirm no token all-to-all.
6. Benchmark current batch-one HBM Marlin.
7. Implement/probe direct pageable Grace CUDA views.
8. Benchmark full-footprint HBM/LPDDR sparse cache access.

Exit: artifact, DSA, Marlin selection, distributed semantics, and direct system-memory access
are viable. Otherwise activate a documented contingency trigger.

### Phase 1: Config, Manifest, and Exact Planner

- Add config/CLI validation and format descriptors.
- Implement safetensors manifest and exact per-rank accounting.
- Add tier-aware KV block sizing and plan-only output.
- Implement linear ownership/even residency fallback.
- Enforce HBM/LPDDR reserves before allocation.

Exit: plan-only predicts every physical allocation without reading model payloads.

### Phase 2: Allocator and Destination-Aware Loader

- Implement mandatory pageable CUDA view and `GraceAllocation`.
- Stream Marlin conversion one expert at a time into final hot/cold bundles.
- Add layer-aware EP filtering and exact maps.
- Audit NUMA pages and remove duplicates.

Exit: target loads with reserves and no registered/copied cold backing.

### Phase 3: Tiered Marlin Execution

- Add ID partition/remap.
- Run hot HBM and cold UVA kernels with independent workspaces.
- Join once before native vLLM reduction.
- Integrate custom ops, torch.compile, and CUDA graphs.
- Validate empty/hot/cold/mixed/skewed routes.

Exit: tiered output matches qualified W4A16 reference.

### Phase 4: Four-Rank Tracer Bullet

Use linear ownership, even placement, HBM main cache, native collectives, and no profile.
Decode at 4K and 32K, then 400K if exact HBM planning permits. Record a full timeline; no
performance gate applies.

### Phase 5: Host-UVA Main MLA Cache

- Allocate main cache from tier-aware block plan.
- Bind host-UVA tensor into native sparse MLA.
- Keep indexer HBM-only and full-layer-only.
- Validate cache append, block boundaries, top-k reads, and graphs.
- Run 32K, 128K, and populated 400K.

Exit: AUTO selects the lowest predicted valid complete plan.

### Phase 6: Trace-Driven Owner and Residency Placement

- Capture sequence-aware routing and sampled index traces.
- Add arbitrary per-layer owner maps to loader/runtime.
- Run balanced owner search and tail-aware residency placement.
- Validate held-out prediction within 20% of replay.

### Phase 7: End-to-End Tuning

- Tune batch-one Marlin/UVA kernels, stream overlap, workspaces, and graph boundaries.
- Tune/measure vLLM collectives and exact operation counts.
- Compare cache tiers and profiled/unprofiled placement.
- Populate 400K and measure at least 256 warmed decode tokens.
- Report prefill separately.

### Conditional Phase 8: Grace CPU Experts

Activate only if GPU UVA is invalid or misses targets and a complete-expert CPU prototype is
around 10% faster including queue and handoff. Execute all three projections on CPU, transfer only
input/final partial, require persistent workers and <=20 us dispatch. Its absence does not
block a qualifying UVA build.

## Testing

### Unit and Loader

- Pinned revision/checksum and format-descriptor validation.
- Safetensors expert coverage, shapes, dtypes, and exact bytes.
- Plan/actual allocation reconciliation.
- Per-layer ownership, hot/cold maps, and exact-zero inactive entries.
- Tier-aware KV block counts and independent reserves.
- Sequence-level trace split and deterministic optimizer.
- Unsupported 4-bit format failures.
- Streaming conversion bounded peak and no duplicate cold tensor.

### Kernels and Memory

- HBM Marlin versus dequantized BF16.
- HBM versus pageable-UVA Marlin for 1/2/4/8 experts.
- Weights, scales, and metadata in system memory.
- No hidden H2D copy or registration.
- CUDA graph replay and owner lifetime.
- Full-footprint HBM versus host sparse MLA parity.
- Cache append/read across block and 2,048-token boundaries.
- Local/remote NUMA negative tests.

### Distributed and Model

- TP4+EP4 bootstrap/teardown and collective trace.
- Router/top-k identity across ranks.
- Ownership patterns `2/2/2/2`, `8/0/0/0`, `3/3/2/0`.
- Existing result reduction versus reference.
- HBM-cache versus host-cache logits.
- Linear versus optimized placement logits.
- Official FP8 comparisons, quantization KLD, retrieval, coding/tool-use, and 400K invariants.

## Acceptance Gates

### Correctness

- Pinned artifact passes declared quality thresholds.
- Native DSA and cache layout are correct.
- HBM and UVA W4A16 outputs meet identical tolerances.
- Distributed logits match reference with no missing/double expert contribution.
- Router and indexer IDs are rank-consistent.

### Memory and Locality

- 400K creates and warms successfully.
- >=5 GB HBM planned and >=4 GB observed free/rank.
- >=8 GB planned LPDDR reserve/rank.
- >=95% local and <5% remote pages.
- No expert-sized allocation, copy, registration, conversion, or migration during decode.

### Performance

- Local LPDDR >=400 GB/s and paired C2C >=350 GB/s.
- Selected UVA expert effective reads >=300 GB/s.
- Host MLA p95 <=0.5 ms/token/rank when selected.
- Selected 12 KiB collective <=15 us median, <=10 us desired.
- Simulator predicts replay within 20%.
- At least 256 warmed tokens at populated 400K.
- Minimum 100 TPS; primary 150 TPS; 200 only with measured <=5 ms evidence.

## Failure Policy

- Mutable/unpinned checkpoint or checksum mismatch: fail.
- Unsupported quant schema/backend fallback: fail.
- Artifact quality failure: evaluate another pre-existing HF candidate or activate contingency.
- Existing copying CUDA view used for cold storage: fail.
- Generic offloader combined with tiered MoE: fail.
- Wrong MLA spec/layout or cache-tier accounting: fail.
- Remote pages/ambiguous topology: fail strict mode.
- HBM pressure: replan more experts cold without weakening reserve.
- LPDDR pressure: fail; no remote/NVMe spill.
- UVA failure: reject UVA and invoke the decision record.
- Grace cache failure: select HBM and publish revised measured targets.
- Never reduce context, precision, reserves, or semantics silently.

## Commit Decomposition

1. Artifact descriptor, mirror manifest, and qualification tests.
2. Pageable Grace CUDA view and allocation/page tests.
3. Tier-aware KV capacity model and planner.
4. Tiered-MoE config and exact manifest.
5. Layer-aware maps and EP loader filtering.
6. Streamed destination-aware Marlin conversion.
7. Tiered Marlin hot/cold execution.
8. CUDA graph/custom-op/distributed integration.
9. Host-UVA sparse MLA allocation and execution.
10. Trace capture and owner/residency optimizer.
11. Metrics, JUPITER harness, and operations documentation.
12. Conditional CPU/SVE work only if gated in.

## Principal Risks

| Risk | Early evidence | Response |
|---|---|---|
| Community checkpoint quality is insufficient | Phase 0 evaluation | qualify another existing HF W4A16 descriptor or contingency |
| Marlin batch-one latency exceeds floors | complete-layer probe | specialized batch-one W4A16 CUDA kernel |
| Marlin cannot use pageable pointers | UVA probe | specialized UVA kernel, then conditional CPU gate |
| Sparse MLA assumes HBM | full-footprint probe | adapt pointer kernel or HBM fallback |
| KV planner sizes host cache from HBM | plan-only tests | tier-aware capacity path before allocation |
| Post-processing recreates all weights in HBM | loader peak trace | one-expert streamed conversion |
| EP emits unexpected all-to-all | Phase 0 collective trace | no-DP/no-SP prepare/finalize path |
| Linear ownership creates rank tails | trace replay | per-layer balanced ownership |
| vLLM revision drift breaks fork | CI against pinned base | controlled rebase cadence and decision record |

## References

- [Pinned W4A16 checkpoint](https://huggingface.co/lowbitcoffee/GLM-5.2-W4A16)
- [Pinned checkpoint files](https://huggingface.co/lowbitcoffee/GLM-5.2-W4A16/tree/55c92ae85b7ec564c94634964b6f5efe5c09a844)
- [Pinned quantization config](https://huggingface.co/lowbitcoffee/GLM-5.2-W4A16/blob/55c92ae85b7ec564c94634964b6f5efe5c09a844/config.json)
- [Official GLM-5.2-FP8](https://huggingface.co/zai-org/GLM-5.2-FP8)
- [JUPITER configuration](https://apps.fz-juelich.de/jsc/hps/jupiter/configuration.html)
- [vLLM INT4 W4A16](https://docs.vllm.ai/en/latest/features/quantization/llm_compressor/int4/)
- [NVIDIA Grace Hopper](https://resources.nvidia.com/en-us-grace-cpu/nvidia-grace-hopper)

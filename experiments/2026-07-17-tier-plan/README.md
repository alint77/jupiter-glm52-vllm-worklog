# Header-only manifest and expert-tier plan

Phase 1 implementation on branch `tiered-moe-grace-view`, using the pinned
GLM-5.2 W4A16 revision and JUPITER allocation `957083`.

## Exact artifact inventory

The new manifest reads all eight safetensors headers without materializing
payloads. It validates the pinned GLM architecture and W4A16 descriptor, index
coverage, shard contents, quantization fields, every routed expert component,
and the index `total_size`.

| Item | Exact bytes |
| --- | ---: |
| Complete checkpoint | 387,667,154,688 |
| Routed experts in checkpoint | 373,713,408,000 |
| Non-routed checkpoint tensors | 13,953,746,688 |
| One stored expert | 19,464,240 |
| One final fused-Marlin expert | 19,464,200 |
| Runtime experts per EP4 rank | 93,428,160,000 |

There are 175,527 tensors. Routed coverage is exactly layers 3-77, experts
0-255, and packed weight, scale, and shape entries for gate, up, and down
projections. Header scanning and four-rank planning take 1.47 seconds after
Python imports.

The 40-byte stored/runtime difference is intentional. The checkpoint stores
three `I64[2]` shape tensors per expert, while the current fused runtime keeps
two `BF16[2]` shape tensors. Static act-order leaves empty runtime `g_idx`
arrays, and the 2,048-wide expert intermediate needs no Marlin padding.

## Initial machine reconciliation

Inputs use the measured 97,871 MiB HBM and 121,677 MiB local Grace capacity,
5 GiB reserve in each tier, the exact 30,834,055,680-byte baseline KV cache,
and baseline runtime categories. The non-expert weight component is explicitly
named `nonexpert_weights_baseline_derived` because the source log reports model
memory to only two decimal GiB.

Each rank receives the same deterministic even-placement plan:

| Item | Per rank |
| --- | ---: |
| Fixed HBM allocations | 41,526,448,353 bytes |
| Hot expert slots | 2,863 |
| Hot expert storage | 55,726,004,600 bytes |
| Cold expert slots | 1,937 |
| Cold pinned-UVA storage | 37,702,155,400 bytes |
| Planned HBM including reserve | 102,621,162,073 bytes |
| Planned Grace including reserve | 43,070,864,520 bytes |

This is an exact expert placement and a fail-closed capacity calculation for
the supplied physical categories. It is not yet the final whole-process plan:
the next slice must derive non-routed TP/replicated runtime weights and cache,
workspace, graph, and scratch allocations directly rather than accepting the
rounded baseline-derived weight component.

## Exact non-routed reconciliation

The checkpoint inventory is now classified through the current GLM runtime
loader rules rather than inferred by subtracting rounded process-memory log
categories. The classifier accounts for replicated tensors, TP-sharded
tensors, checkpoint indexer copies dropped on non-indexer layers, and shape
metadata removed by runtime fusion.

| Non-routed item | Exact bytes |
| --- | ---: |
| Checkpoint tensors | 13,953,746,688 |
| Dropped checkpoint tensors | 1,068,397,056 |
| Replicated runtime tensors | 1,280,365,824 |
| TP-sharded checkpoint tensors | 11,604,983,808 |
| TP4 runtime shard | 2,901,245,952 |
| Runtime fusion savings | 2,496 |
| Runtime total per rank | 4,181,609,280 |

Using that exact runtime total with the same measured machine capacities and
baseline cache/runtime categories revises the deterministic plan to:

| Item | Per rank |
| --- | ---: |
| Fixed HBM allocations | 36,969,875,079 bytes |
| Hot expert slots | 3,097 |
| Hot expert storage | 60,280,627,400 bytes |
| Cold expert slots | 1,703 |
| Cold pinned-UVA storage | 33,147,532,600 bytes |
| Planned HBM including reserve | 102,619,211,599 bytes |
| Planned Grace including reserve | 38,516,241,720 bytes |

All four ranks receive the same byte totals. Their owned expert ranges are
0-63, 64-127, 128-191, and 192-255, with an even deterministic layer rotation.
The earlier 8.738 GB baseline-derived non-expert estimate was invalid because
rounded aggregate process memory cannot distinguish TP sharding, discarded
checkpoint copies, and runtime fusion. The exact classifier is fail-closed: an
unknown non-routed GLM tensor aborts planning.

## Native 400K cache-tier reconciliation

The planner now constructs the same ordinary, non-v4 `MLAAttentionSpec`
geometry used by the native GLM path. It rejects sliding-window MLA, the
DeepSeek-v4 584-byte layout, a non-`fp8_ds_mla` main cache, a block size other
than 64, or a different indexer pattern. At 400K tokens it derives:

| Cache item | Exact allocation per rank |
| --- | ---: |
| Main MLA: 656 bytes/token x 78 layers | 20,467,200,000 bytes |
| Indexer: 132 bytes/token x 21 layers | 1,108,800,000 bytes |
| Cache blocks | 6,250 |
| Allocated tokens | 400,000 |

Plan-only now prices both physical cache tiers without using the baseline KV
total. The main cache moves between HBM and Grace as one allocation; the
indexer cache always remains in HBM.

| Per-rank item | Host-UVA main cache | HBM main cache |
| --- | ---: | ---: |
| Fixed HBM allocations | 7,244,619,399 | 27,711,819,399 |
| Fixed Grace allocations | 20,467,200,000 | 0 |
| Hot expert slots | 4,643 | 3,591 |
| Cold expert slots | 157 | 1,209 |
| Hot expert bytes | 90,372,280,600 | 69,895,942,200 |
| Cold expert bytes | 3,055,879,400 | 23,532,217,800 |
| HBM total including 5 GB reserve | 102,616,899,999 | 102,607,761,599 |
| Grace total including 8 GB reserve | 31,523,079,400 | 31,532,217,800 |

Peak activation, non-Torch, and CUDA-graph categories are still converted from
the baseline's rounded GiB values. They are inside the fixed HBM total, while
the separate 5 GB HBM reserve covers the remaining measurement uncertainty.
Workspace and conversion-scratch accounting remain to be derived before the
Phase 1 exit criterion is complete.

## Layer-aware loader filter prerequisite

The safetensors iterator now accepts a strict per-layer expert ownership map.
Unlike the existing generic EP filter, this mode rejects missing layer maps and
skips every remote expert component before `get_tensor()`: packed weights,
scales, and shape metadata. The generic filter also now recognizes
`.weight_packed` as a heavy payload while retaining its existing conservative
scale/metadata behavior for other quantization backends.

For linear EP4 ownership, the strict map reduces this artifact's per-rank
checkpoint stream from 387,667,154,688 bytes to 107,382,098,688 bytes:
13,953,746,688 non-routed bytes plus one 93,428,352,000-byte routed shard. A
synthetic safetensors test instruments `get_tensor()` and confirms that no
remote quantized component is materialized. The iterator primitive is ready;
the tiered loader still has to install the planner-generated layer map and then
apply the dropped-indexer and streamed-conversion rules.

## Dedicated vLLM configuration

The v2 configuration surface is now represented by `TieredMoEConfig` and wired
through `EngineArgs` into `VllmConfig`. It exposes the enable/backend/profile,
routing trace, physical reserves, strict NUMA, plan-only, MLA cache-tier, and
Grace machine-profile options from the plan.

Validation fails closed unless the initial production contract is satisfied:
CUDA, pinned GLM architecture/model type, TP4 with DP/PP/PCP/DCP all one, EP
and EP filtering enabled, EPLB disabled, NUMA binding enabled, one sequence,
400K maximum length, `fp8_ds_mla`, at least 5/8 decimal GB reserves, and no
generic UVA or prefetch model offload. Building an engine config from the real
artifact with this exact contract succeeds without reading model payloads. The
standalone planner remains useful for isolated accounting, while the production
server command now uses the same planner through its early plan-only exit.

The selected planner, EP filter, safetensors, and configuration suite has 51
passing tests. All changed-file pre-commit hooks pass. The plan-only module is
invoked with:

```bash
.venv/bin/python -m vllm.entrypoints.tiered_moe_plan MODEL \
  --ep-size 4 \
  --max-model-len 400000 --mla-cache-tier auto \
  --hbm-capacity-bytes BYTES --hbm-reserve-bytes BYTES \
  --host-capacity-bytes BYTES --host-reserve-bytes BYTES \
  --fixed-hbm-allocation NAME=BYTES --summary-only
```

## Exact runtime buffers and server plan-only exit

The final Phase 1 accounting derives the additional steady HBM used by two
independent Marlin calls at the pinned 8,192-token scheduler ceiling. Hot and
cold calls each receive their own BF16 intermediate arenas and lock state; the
plan also includes alignment storage, tier maps, remapped IDs/weights, and a
bounded one-expert conversion allocation.

| Runtime allocation | Exact bytes per rank |
| --- | ---: |
| Two-tier Marlin intermediate arenas | 3,221,225,472 |
| Alignment buffers | 663,528 |
| Marlin lock workspaces | 4,224 |
| Placement maps | 249,600 |
| Remapped IDs and weights | 1,048,576 |
| Total steady HBM | 3,223,191,400 |
| Transient conversion scratch | 38,928,440 |

The checked-in [GH200 profile](../../profiles/jupiter-gh200-baseline.json) uses
exact rank-local capacities and budgets each two-decimal baseline runtime
metric at its upper rounding boundary. Its SHA-256 is
`180d0284b120ad21bcfadec4fffd553aae6d47e6ade073f4044876a73da19026`.

The production `vllm serve` parser now has an early plan-only exit. A real run
against the immutable checkpoint completed in about nine seconds without
starting an API listener, worker process, or CUDA context and produced equal
totals on all four ranks:

| Per-rank item | Host-UVA main cache | HBM main cache |
| --- | ---: | ---: |
| Hot expert slots | 4,477 | 3,425 |
| Cold expert slots | 323 | 1,375 |
| Planned HBM including 5 GB reserve | 102,625,140,327 | 102,616,001,927 |
| Planned Grace including 8 GB reserve | 34,754,136,600 | 34,763,275,000 |
| Peak HBM during one-expert conversion | 97,664,068,767 | 97,654,930,367 |

This completes the Phase 1 exit criterion: every planned physical allocation
is priced before any model payload is read. The earlier cache-only figures are
retained above to show how each accounting slice changed the plan.

Focused verification is 16/16 passing for manifest, planner, runtime-buffer,
machine-profile, and tiered argument tests. A broad `test_arg_utils.py` run was
98/112 passing; all 14 failures are pre-existing tests that instantiate fake
Hub IDs while this environment intentionally sets `HF_HUB_OFFLINE=1`.

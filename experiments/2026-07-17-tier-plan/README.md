# Header-only manifest and expert-tier plan

Phase 1 implementation on branch `tiered-moe-grace-view`, using the pinned
GLM-5.2 W4A16 revision and JUPITER allocation `957083`.

## Exact artifact inventory

The new manifest reads all eight safetensors headers without materializing
payloads. It validates the pinned GLM architecture and W4A16 descriptor, index
coverage, shard contents, quantization fields, every routed expert component,
and the index `total_size`.

| Item | Exact bytes |
|---|---:|
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

## First machine reconciliation

Inputs use the measured 97,871 MiB HBM and 121,677 MiB local Grace capacity,
5 GiB reserve in each tier, the exact 30,834,055,680-byte baseline KV cache,
and baseline runtime categories. The non-expert weight component is explicitly
named `nonexpert_weights_baseline_derived` because the source log reports model
memory to only two decimal GiB.

Each rank receives the same deterministic even-placement plan:

| Item | Per rank |
|---|---:|
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

Seven focused manifest, schema, byte-accounting, planner, capacity-failure, and
CLI tests pass. The plan-only module is invoked with:

```bash
.venv/bin/python -m vllm.entrypoints.tiered_moe_plan MODEL \
  --ep-size 4 \
  --hbm-capacity-bytes BYTES --hbm-reserve-bytes BYTES \
  --host-capacity-bytes BYTES --host-reserve-bytes BYTES \
  --fixed-hbm-allocation NAME=BYTES --summary-only
```

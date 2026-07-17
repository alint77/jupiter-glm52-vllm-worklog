# Tiered destination storage

Phase 2 implementation on branch `tiered-moe-grace-view`, using JUPITER
allocation `957083`.

## Loader ownership map

`DefaultModelLoader` now builds the strict layer-aware map from the validated
checkpoint manifest and the actual EP rank. It passes that map to the ordinary
safetensors iterator, which skips every remote expert component before payload
access. Linear EP4 maps each of the 75 routed layers to exactly 64 owned global
expert IDs. Multithreaded loading fails closed because that iterator does not
support the strict filter.

The existing instrumented iterator test proves remote packed weights, scales,
and shape metadata never reach `get_tensor()`. A new loader-level test proves
the planner map reaches the iterator. The focused loader/planner/filter suite
is 56/56 passing.

## Compact final destinations

The pinned GLM fused-Marlin format is represented as six component-major views
over one byte buffer per tier:

| Component per expert | Shape | Type | Bytes |
| --- | ---: | --- | ---: |
| Combined gate/up packed weight | 384 x 8192 | INT32 | 12,582,912 |
| Down packed weight | 128 x 12288 | INT32 | 6,291,456 |
| Combined gate/up scale | 48 x 4096 | BF16 | 393,216 |
| Down scale | 16 x 6144 | BF16 | 196,608 |
| Combined gate/up shape | 2 | BF16 | 4 |
| Down shape | 2 | BF16 | 4 |
| Total | | | 19,464,200 |

One backing per tier keeps each tensor contiguous for Marlin and keeps every
component of an expert in the same physical tier. The cold owner is retained
as a `GraceAllocation`; its CUDA alias supplies the component views.

## GH200 smoke test

A representative host-main-cache layer with 60 hot and four cold experts was
allocated on GPU 0 with rank 0 bound to CPU/Grace NUMA node 0:

| Item | Result |
| --- | ---: |
| HBM backing | 1,167,852,000 bytes |
| Pinned-Grace backing | 77,856,800 bytes |
| Total | 1,245,708,800 bytes |
| Allocation time | 0.683 s |
| First-touch and synchronize | 0.016 s |
| Cold alias device | `cuda:0` |
| Sampled cold pages | 256/256 on node 0 |

An intentionally unbound first run failed the strict locality audit with zero
pages on node 0. Repeating with `--cpu-bind=map_cpu:0`, the production rank-0
binding, passed at 100% locality. This confirms both the fail-closed audit and
the final compact allocation on hardware. The next step is bounded one-expert
checkpoint staging and Marlin conversion into these destinations.

## One-expert final commit

The destination API now validates the complete set, shape, and type of all six
converted components before writing anything. It resolves the expert's compact
slot and commits exactly that expert to one tier. A unit test verifies that an
adjacent expert slot remains untouched and that an unassigned ID fails closed.

On GH200, a single 19,464,200-byte synthetic converted expert was written from
HBM into its fresh pinned-Grace destination in 46.50 ms, including first page
faults and six component copies. Every component compared equal afterward and
the post-copy NUMA audit remained 256/256 pages on node 0. This is load-time
cost, not decode latency.

## Actual checkpoint conversion

The one-expert stream probe reads the real nine-component checkpoint bundle,
fuses gate/up/down into native one-expert GPU staging tensors, invokes vLLM's
Marlin conversion, and commits the result to pinned Grace. It also exposed and
corrected two planning assumptions before loader integration:

- Native repacking changes packed shapes to 384 x 8192 and 128 x 12288 while
  preserving their byte counts.
- Repacking uses internal temporaries. The measured HBM high-water is
  44,662,784 bytes, above the former 38,928,440-byte source-plus-result
  estimate. The plan now enforces a fixed 64 MiB conversion budget.

For layer 3, expert 0, the first real run measured:

| Stage | Result |
| --- | ---: |
| Checkpoint bundle | 19,464,240 bytes |
| Header-selected payload read | 0.080 s |
| Fusion and host-to-device staging | 0.133 s |
| Native Marlin conversion | 0.311 s |
| Final pinned-Grace commit | 0.0025 s |
| Final expert | 19,464,200 bytes |
| Peak HBM | 44,662,784 bytes |

All six final components compare exactly with the converter output and the
cold allocation remains 100% local in the post-commit NUMA audit. The 64 MiB
budget leaves 22,446,080 bytes above this observed high-water and keeps the
planner fail-closed if a later conversion exceeds it.

The probe now calls the production `OneExpertCheckpointStager` and
`convert_glm_w4a16_expert` rather than carrying a benchmark-only conversion.
The stager validates all nine stored shapes and types, refuses a second expert
until the first is complete, and fails on a partial final bundle. The hardware
rerun passed with the same 44,662,784-byte peak, 0.448 s combined fusion/H2D/
Marlin conversion, 0.0023 s final commit, exact output checks, and 100% NUMA
locality. The selected config, manifest, filter, loader, storage, and stager
suite is now 62/62 passing; all changed-file hooks pass.

## Pre-construction worker plan

Placement used to be resolved in `_init_ep_weight_filter`, after native model
parameter construction. That is early enough to skip checkpoint reads but too
late to choose physical parameter destinations. `DefaultModelLoader` now
resolves the worker's selected scenario before `initialize_model`, exposes it
through a scoped construction context, and reuses the same object for the
layer-aware iterator map.

Production loading now requires an explicit `hbm` or `host_uva` main-cache tier
and a versioned machine profile; `auto` remains available only to plan-only for
comparison. A direct real-artifact rank-0 resolution with `host_uva` selected
returned 4,477 hot and 323 cold slots, 75 ownership maps with experts 0-63,
and the same machine-profile hash and totals as server plan-only. It reads only
headers and exits without model parameter allocation. The scoped context is
unit tested to reset after model loading, and changed-file mypy/hooks pass.

The WNA16 Marlin construction path now consumes that context and resolves its
module prefix to exactly one `LayerExpertPlacement`. It fails before allocation
if the prefix has no plan entry or if native local-expert construction disagrees
with the planned hot-plus-cold count, then attaches the validated placement to
the routed layer. This is the final non-allocating guard before replacing the
native monolithic expert tensors with the compact destination bundles. The
selected suite is 64/64 passing.

## Full destination-aware load

Marlin construction now allocates only the planner's compact hot and cold
destinations. The standard loader diverts each complete nine-tensor GLM expert
bundle through the bounded converter and writes it directly to its final tier;
dense and shared weights continue through vLLM's ordinary loader.

Four experts have checkpoint components split across two safetensors shards.
The iterator discovers those cases from headers, skips their fragments during
the normal pass, then rereads only those fragments adjacently. This keeps the
one-expert staging bound without buffering a second expert.

The four-rank host-cache plan loaded all 4,800 rank-local layer-expert slots:

| Measurement | Result |
| --- | ---: |
| Routed expert stream | 4,800 experts |
| Stream and conversion time, warm GPFS cache | 29.73-30.85 s |
| All checkpoint weight loading | 36.62-37.74 s |
| Complete model loading | 40.58-43.02 s |
| vLLM model-memory delta | 89.2 GiB per rank |

The compact allocation crossed model post-processing without the former
64-expert monolithic HBM allocation. This completes the main Phase 2 loading
path; a full post-load NUMA audit remains to be added.

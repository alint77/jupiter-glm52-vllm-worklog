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
| Combined gate/up packed weight | 768 x 4096 | INT32 | 12,582,912 |
| Down packed weight | 256 x 6144 | INT32 | 6,291,456 |
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

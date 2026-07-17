# Host-UVA MLA cache tracer bullet

Initial Phase 5 tracer bullet on JUPITER allocation `957083`.

The tier-aware KV plan preserves the native 6,250-block count for 400K tokens,
but assigns physical memory independently: 78 main `fp8_ds_mla` tensors use
paired pinned Grace/UVA storage and the 21 DSA indexer tensors remain in HBM.
The main cache is 20,467,200,000 bytes per rank. Each allocation retains its CPU
owner, exposes a CUDA alias to the unchanged attention binding path, and audits
four NUMA samples per tensor.

The first complete startup used the original 4,477-hot plan. It proved cache
allocation and inference, but left only 270 MiB free per GPU. This violated the
v2 reserve gate. The machine profile was reconciled against the physical
97,281 MiB device capacity and measured steady runtime allocation after
releasing unused allocator cache. The final exact plan is:

| Per-rank item | Bytes / count |
| --- | ---: |
| Hot expert slots | 4,330 |
| Cold expert slots | 470 |
| Planned HBM including 5 GB reserve | 101,990,135,040 bytes |
| Planned physical free HBM | 5,016,386,816 bytes |
| Planned Grace including 8 GB reserve | 37,615,374,000 bytes |
| Machine-profile SHA-256 | `32a22c5baf8fac3922d16e479e8e0cca32724d02d8085465d12019f35f88224f` |

The final eager server reached the following hardware boundary on all four
ranks:

| Measurement | Result |
| --- | ---: |
| KV capacity | 400,000 tokens |
| Host-UVA allocation | 19.06 GiB across 78 tensors/rank |
| Minimum sampled local pages | 100.0% |
| Observed free HBM after cache release | 4.64 GiB/rank |
| `nvidia-smi` free before request | 4,754-4,756 MiB/rank |
| `nvidia-smi` free after request | 4,732-4,734 MiB/rank |
| Final streamed-load time | 158.19 s, GPFS-bound |
| Final complete model-load time | 215.15 s |
| Deterministic 5-in/8-out request | 4.05 s wall time, HTTP 200 |

The completion begins with `Paris`, establishing an end-to-end cache
append/read and tiered-MoE decode path. This is a correctness tracer bullet,
not a performance result. The server used `--enforce-eager`; compile, CUDA
graphs, cache-boundary tests, populated 32K/128K/400K validation, and reference
logit comparison remain open.

A post-warmup runtime audit now releases unused allocator cache, reads physical
free HBM, and fails closed below 4 GB for the default 5 GB planned reserve.
Focused tests cover tier metadata/block counts and reserve failure behavior.
Ruff and Python 3.12 mypy pass for the changed implementation.

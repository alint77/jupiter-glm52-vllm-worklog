# MiMo-V2.6-Pro-RL on tiered MoE — bring-up

Target shape (as requested): **250K context, c=1, MTP3, DCP off**.

## The model

`XiaomiMiMo/MiMo-V2.6-Pro-RL`, released 2026-09-21. 132 safetensors shards,
**532 GB**, downloading to `/e/fscratch/profound/$USER/models/MiMo-V2.6-Pro-RL`
via `download.sh`.

| | |
|---|---|
| architecture | `MiMoV2ForCausalLM` — **already registered** (`mimo_v2.py`) |
| layers | 70 (layer 0 dense, 69 MoE) |
| experts | 384 routed/layer, top-8 → **26,496 total** |
| attention | GQA: 128 Q heads, **8 KV heads**, QK head_dim 192, V head_dim 128 |
| hybrid pattern | **10 full-attention, 60 sliding-window (window 128)** |
| MTP | **3 layers** in the weights (`model.mtp.layers.{0,1,2}`) |
| also ships | `dflash/` drafter: 5 layers, SWA 1024, `is_causal: false` |
| head dims | **QK 192 / V 128 — asymmetric** (hence the fork's `diffkv` backend) |
| multimodal | 28 `visual.blocks.*` + an `audio_config` |
| quantization | `quant_method: fp8`, `store_dtype: mxfp4`, block `[128,128]` |
| max_position | 1,048,576 |

## Why tiered MoE is mandatory, not optional

At EP4, ~6,624 experts per rank × ~18 MB ≈ **116 GiB of experts per rank**,
against 95 GiB HBM. It cannot run without spilling experts to Grace
(`host_capacity_bytes` is 118.8 GiB per rank), so the tiered path is the only
way this model runs at all here.

Rough per-rank budget: 116 GiB experts + ~6 GiB sharded attention weights + KV,
against 95 HBM + 118.8 Grace = 213.8 GiB. It fits, but not loosely.

## KV cache: 19.21 GB/rank at 250K (measured against the real code)

Two separate questions, and an earlier revision of this file got the second
one wrong.

**Is the sliding window promoted to full attention?** No.
`kv_cache_utils.py:1568-1572` gates promotion on `has_mla and has_regular_swa`
-- that is the GLM shape, MLA target plus SWA drafter. MiMo is GQA, so
`has_mla` is False and the branch returns early. Its layers group by spec type
instead: one `FullAttentionSpec` group of 10, one `SlidingWindowSpec` group
of 60.

**So what does it cost?** Not "only the 10 full-attention layers". The pool is
sized by the *largest* group, because `_pool_bytes_per_block` returns
`page_size * max(len(g.layer_names))`. Measured by calling the real function:

```
full.page_size_bytes  = 81920      # 64 x 2 heads x (192 + 128) x 2 bytes
swa.page_size_bytes   = 81920
_pool_bytes_per_block = 4915200    # = page x 60, the larger group
num_blocks @250K c=1  = 3908
RESERVED TOTAL        = 19.21 GB
```

| | GB/rank at 250K |
|---|---|
| an earlier claim here, 10 full layers only | 3.20 — **wrong** |
| all 70 layers | 22.41 |
| **actual, measured** | **19.21** |

`head_size_v` *is* budgeted, despite `real_page_size_bytes` appearing to use
`head_size` for both K and V -- another reason this was measured rather than
read.

### What that does to the budget

| per rank | |
|---|---|
| HBM | 102 GB |
| KV at 250K | 19.2 |
| non-routed weights | ~8 |
| reserve | ~9 |
| **left for hot experts** | **~66 GB -> ~3,300 of 6,624 (~50%)** |

So KV *is* a real claim on HBM, not the rounding error the 3.2 GB figure
implied. Half the experts resident is still a workable hot/cold regime, and
`kv_cache_dtype=fp8` would halve the KV if more residency is wanted.

## Measured: runtime_expert_bytes = 20,054,016 (19.125 MiB/expert)

Job 1945336, one `FusedMoE` at MiMo's real dims on one GPU
(`expert_bytes.py`). The planner budgets in post-repack bytes, and nothing in
the tree recorded MiMo's, so it was measured rather than derived:

```
routed_experts.w13_weight         4608.00 MiB
routed_experts.w13_weight_scale    288.00 MiB
routed_experts.w2_weight          2304.00 MiB
routed_experts.w2_weight_scale     144.00 MiB
TOTAL 7344.00 MiB / 384 experts = 19.125 MiB/expert
```

**The Marlin mxfp4 repack is byte-neutral** — `process_weights_after_loading`
leaves every tensor the same size, so runtime layout equals checkpoint layout.
That is not true of GLM's `vllm_marlin_static_w4a16`, and it simplifies the
manifest: one constant, no separate checkpoint/runtime accounting.

For comparison GLM is 20.3 MiB/expert, so per-expert the two models are close;
MiMo simply has more of them.

| | per rank at EP4 |
|---|---|
| experts | 6,624 |
| expert bytes | **123.7 GiB** |
| against | 95 GiB HBM + 118.8 GiB Grace |

So roughly a third of the experts fit in HBM once weights, KV and reserve are
taken out — the hot/cold regime the overlap machinery exists for, which is what
makes tiered worth running here even with a linear (profile-less) placement.

## Structural blocker found while scoping: group count

`get_tiered_kv_available_memory` requires exactly one group and that it be a
`UniformTypeKVCacheSpecs` (`tiered_moe_kv.py:93-95`, "Tiered GLM requires one
uniform-type MLA cache group"). GLM satisfies that because the promotion path
unifies everything into one group.

MiMo does not, and for the same reason its KV is cheap: the promotion branch
returns early for GQA, so `get_kv_cache_groups` splits layers by spec type --
**a FullAttentionSpec group for the 10 GA layers and a SlidingWindowSpec group
for the 60 SWA layers. Two groups, neither uniform-type.**

So the tiered KV accounting has to be generalised to sum across groups, not
just taught about GQA specs. The cheap-KV result and this blocker are two faces
of the same branch.

## The tiered port: what is done

All committed, GLM unaffected throughout (every shared change is a defaulted
parameter or a `model_type`-gated branch; its assertions of 21,579,452,160
bytes at 6251 blocks still pass).

| piece | |
|---|---|
| multi-group KV accounting | `_pooled_kv_bytes`, sizes against `_pool_bytes_per_block` |
| GQA KV plan | `plan_mimo_v2_kv_cache`, 19,208,601,600 B/rank at 250K |
| manifest | `build_mimo_v26_manifest`, validated on the real 535 GB checkpoint |
| validator pins | scoped by `model_type` rather than relaxed |
| dispatch | `tiered_model_family`, `build_tiered_moe_manifest`, `plan_tiered_kv_cache_from_path` |

Measured constants, each cross-checked more than one way:

| | |
|---|---|
| `runtime_expert_bytes` | 20,054,016 (GPU probe, index arithmetic, uniform-size assertion over 26,496 experts) |
| KV at 250K, c=1 | 19,208,601,600 B/rank (planner and allocator pinned together in a test) |
| experts per EP4 rank | 123.7 GiB |
| non-routed, checkpoint-wide | 34.7 GB |

## The planner runs end to end

Job 1950886 reached, on every rank:

```
Tiered MoE residency: 2931 hot / 3693 cold experts per rank
                      (54.7 GiB available / 19.1 MiB per expert)
```

Manifest, non-routed inventory, KV plan and placement all reconciled, and the
19.1 MiB per expert matches the measured 20,054,016 bytes. The non-routed
inventory came out at 8.00 GiB per rank: 428 replicated tensors, 588
tp-sharded, 48 dropped -- exactly the MTP tensor count, since speculative
decoding is off.

## The remaining piece: Mxfp4MoEMethod has no tiered hook

The run then OOMed. `Mxfp4MoEMethod.create_weights` allocated all 6,624 local
experts (123.7 GiB) while the planner had budgeted 2,931 hot (54.7 GiB), and
died at 94.5 GiB. **No launcher setting closes a 69 GiB gap** -- the reserve
and utilisation knobs are worth about 2 GiB each.

The tiered allocation hook lives in the quantisation methods, and only three
have it: `auto_gptq.py`, `modelopt.py`, and
`compressed_tensors_moe_wna16_marlin.py`. No mxfp4 caller has needed tiering
before, so `mxfp4.py` has none.

Scope, read rather than estimated -- it is more than mirroring auto_gptq's
construction branch:

| | |
|---|---|
| `create_weights` | tiered branch: resolve placement, attach it, allocate tier storage, register zero-sized placeholders |
| `allocate_layer_expert_storage` (`:193`) and `build_expert_component_views` (`:109`) | both call `glm_marlin_components(group_size)` **hardcoded**; mxfp4 needs its own specs plus dispatch at both |
| `setup_tiered_moe_kernels` | reads components generically off storage, but builds WNA16/NVFP4 Marlin kernels keyed on group_size; mxfp4 needs its own kernel branch for hot and cold |
| staged cold path | `_staged_cold_tier` / `_cold_prefetch_for` rebuild the cold kernel from the same quantisation description, so they need the mxfp4 variant too |
| `process_weights_after_loading` | delegate to `setup_tiered_moe_kernels`, as the other three do |

The component specs are already measured, so that input is not in doubt:
`w13_weight (4096, 3072) u8`, `w2_weight (6144, 1024) u8`,
`w13_weight_scale (4096, 192) u8`, `w2_weight_scale (6144, 64) u8` --
20,054,016 bytes, matching the probe and the index arithmetic.

This is effectively adding a quantisation backend to the tiered runtime rather
than finishing the port, and it is the only thing between here and a server.

## Superseded: the earlier "one remaining piece"

`build_glm_non_routed_runtime_inventory` classifies every non-expert tensor
into `TP_SHARDED` / `REPLICATED` / `EP_SHARDED` / `DROPPED` from GLM's tensor
names and its sparse indexer, then asserts the total reconciles with
`manifest.non_routed_bytes`. MiMo needs its own, and it is more work than GLM's
because the Omni class builds a 28-block vision tower and an audio encoder
whose TP sharding has to be read out of `mimo_v2.py` and `mimo_v2_omni.py`
rather than assumed.

That is the 34.7 GB above. A wrong sharding rule there yields a plausible
per-rank figure and fails much later as an unexplained OOM -- the same shape as
the draft-cache gap and the `3 <= layer_id < 78` window, both of which produced
quietly wrong numbers rather than exceptions.

`build_rank_load_plan` therefore raises for a non-GLM family, naming the gap,
instead of calling the GLM classifier on MiMo tensors.

## Status
## Status

Port written except the non-routed inventory above. Nothing has launched with
tiered MiMo yet.

Download **complete** — 130 shards, 535 GB, every shard in the index present
(the HF API's "132" counts the dflash drafter).

`smoke.sbatch` (job 1943383) submitted: non-tiered, TP4, `cpu_offload_gb=60`,
64K context, eager. It answers whether the weights load at all and what the KV
actually costs per token, which is the input the tiered port needs.

## Server bring-up (tiered, 250K, c=1, no SD)

MTP dropped at the user's request; the target is a working tiered server and
its numbers. Fixes in order, each found by the run before it:

1. **mxfp4 tiered backend** (commits 0268435749..f69bb6fa2d): construction
   hook, measured resident component specs (`w13_weight (384,8192) i32`,
   `w2_weight (128,12288) i32`, `w13_weight_scale (192,4096) e8m0`,
   `w2_weight_scale (64,6144) e8m0`; 20,054,016 B/expert), per-expert
   conversion through `convert_weight_to_mxfp4_moe_kernel_format`, per-tier
   Marlin mxfp4 kernels, `apply` routed to `apply_tiered_moe`.
2. **Pinned allocations rounded to a power of two** (ea51bc67a1).
   `pin_memory=True` goes through `CachingHostAllocator`, which rounds up
   (`PowerOf2Ceil`): a 1.009 GiB request pinned 2.000 GiB (`pin_probe`), so
   the cold tier took ~105 GiB/rank against 69 planned and the workers were
   SIGKILLed silently. `GraceAllocation.allocate_pinned` now allocates pageable
   memory at the exact size and `cudaHostRegister`s it. This also affected GLM
   (51 GiB planned vs 75 pinned).
3. **Load succeeds** (job 1959828): weights 132 s, model load 64.16 GiB and
   184 s, residency 2931 hot / 3693 cold per rank.
4. **KV pool undersized, planner oversized** (0781a3a81b). vLLM splits the
   10 full + 60 sliding layers into **seven 10-layer groups** (group size =
   the smaller type's count), all sharing one pool of `81,920 B x 10` blocks.
   Admission charges every group: 3907 full blocks + 6 x 259 sliding blocks
   (window 127 + 2 x 8192 in-flight tokens, +1) = 4.17 GiB. The runtime sized
   the pool for full attention only (3908 blocks, 2.98 GiB) and failed the
   admission check; the planner charged 60 layers per block and reserved
   19.2 GB -- about 14.7 GB/rank of HBM that could hold ~770 more hot experts.
   Both now count 5462 blocks = 4.47 GB, and the tests build groups with
   vLLM's own `_get_kv_cache_groups_uniform_page_size` rather than a hand
   model of it (the old test pinned the wrong model on both sides).

Bench jobs 1960037 (HBM reserve 8 GB) and 1960038 (16 GB) run in parallel:
the freed KV reservation goes to hot experts, so the tighter plan is the one
to watch for OOM.

## Results: MiMo-V2.6-Pro-RL, tiered MoE, 1x GH200 node (TP4/EP4), 250K, c=1, no SD

Both servers came up with a 250,045-token KV pool. A greedy chat check returned
coherent text ("The capital of France is Paris. The Seine River runs through
the city."), so the mxfp4 tiered conversion is correct end to end.

| | 1960037 (reserve 8 GB) | 1960038 (reserve 16 GB) |
|---|---|---|
| hot / cold experts per rank | 3666 / 2958 (55% hot) | 3267 / 3357 (49%) |
| observed HBM free (min) | 7.49 GiB (6.52) | 14.94 GiB (13.97) |
| decode TPOT P50 / P90 (coding suite, 16 x 1024) | **23.13 / 23.17 ms = 43.2 tok/s** | 24.28 / 24.31 ms = 41.2 tok/s |
| TTFT P50, short prompts | 244 ms | 267 ms |
| 32K prefill TTFT | **5.10 s (6.4K tok/s)** | 5.29 s |
| 128K prefill TTFT | **20.8 s (6.3K tok/s)** | pending |
| decode TPOT after 128K prompt | 21.9 ms | |

No speculative decoding, so TPOT is per token. Mean TTFT on the decode suite
(804 ms) includes the first request's cold path; P50 is the steady value.
Each extra ~400 hot experts per rank bought ~1.15 ms/token (24.28 -> 23.13).
Decode after the 128K random prompt is not comparable to the coding suite:
random tokens route differently.

Configuration: `serve.sbatch` (block 64, batched tokens 8192, CUDA graph
capture [1], tiered uva, numa-strict, grace profile
jupiter-gh200-baseline.json, cold tier on the GPU-local Grace NUMA node).

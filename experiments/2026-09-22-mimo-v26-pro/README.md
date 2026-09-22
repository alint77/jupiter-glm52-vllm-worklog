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

## KV cache: answered from the code, not a launch

`kv_cache_utils.py:1568-1572` gates the promotion:

```python
has_mla = any(isinstance(spec, MLAAttentionSpec) for spec in ...)
has_regular_swa = any(isinstance(spec, SlidingWindowSpec) for spec in ...)
if not (has_mla and has_regular_swa):
    return None      # no promotion; the hybrid manager handles mixed specs
```

Promotion needs **both** MLA and sliding-window specs — that is the GLM case (MLA
target plus an SWA drafter), which is why the "page sizes cannot be unified"
warning appeared there. **MiMo V2.6 is GQA, so `has_mla` is False and the branch
returns early. The sliding-window layers are not promoted.**

Per rank at TP4, 250K, bf16 (K and V differ: `2*192 + 2*128 = 640` elements per
token per layer):

| | GB/rank at 250K |
|---|---|
| if promoted (what GLM does) | 22.4 |
| **actual — 10 full-attention layers only** | **3.2** |

The 60 sliding-window layers cost their 128-token window, negligible even with
block rounding. The convertor stripping `attention_chunk_size` keeps the hybrid
manager enabled, which is what makes this path reachable.

**KV is not a constraint for the 250K target.** The constraint is entirely the
expert weights.

## The gap: tiered MoE is hard-pinned to GLM

Every pin fails for this model:

| pin | GLM | MiMo V2.6 |
|---|---|---|
| cache spec | MLA only (`tiered_moe_kv.py:52`) | GQA / `FullAttentionSpec` |
| main layers | 78 + MTP | 70 + 3 |
| sparse indexer | 21 + MTP layers required | none |
| cache dtype | `fp8_ds_mla` | n/a |
| manifest | `build_glm_w4a16_manifest`, W4A16 compressed-tensors | mxfp4 + fp8 scales |
| config validator | `max_model_len=400000`, `block_size=64` | 250K wanted |

So this is a second model family in the tiered path, not a config change.

## Blocker for the requested shape: MTP3 is not supported

The checkpoint ships **3** MTP layers. vLLM pins MiMo-V2 to **1**:

```python
# mimo_v2_mtp.py:53
_MIMO_V2_PRO_NUM_MTP_LAYERS = 1
# speculative.py:391
# vLLM currently supports only the first MiMo-V2 MTP layer.
```

`MiMoV2MultiTokenPredictor.__init__` also hardcodes `num_mtp_layers = 1`
(`mimo_v2_mtp.py:173`), and `forward` indexes `spec_step_idx % num_mtp_layers`
— so asking for 3 speculative tokens today reuses **layer 0 three times** and
silently ignores the weights for layers 1 and 2. It would run, and it would
under-accept, with nothing in the logs saying why.

Three ways forward, for the user to pick:

1. **MTP1 now.** Supported, correct, lower acceptance. Good enough to get the
   model serving and to measure everything else.
2. **Lift the pin to 3.** The constant, the hardcoded `num_mtp_layers`, and the
   weight mapping for `model.mtp.layers.{1,2}`. Contained, but it is a real
   change to a shared model file and needs an acceptance measurement to show it
   helped.
3. **Use the shipped `dflash/` drafter** (5 layers, SWA 1024, `is_causal:
   false`) — the card calls this the model's speculative decoder. Note the
   DFlash CUDA-graph acceptance defect found earlier today; the eager default
   now in `DFlash2Speculator` would cover it, but MiMo's drafter routes through
   `DFlashSpeculator`, which still defaults to the captured graph.

## What already works, and does not need porting

- `MiMoV2ForCausalLM` is registered and handles attention sinks,
  `attention_value_scale`, chunked attention, the hybrid layer pattern, and the
  asymmetric `v_head_dim`.
- `store_dtype: mxfp4` under `quant_method: fp8` routes to `Mxfp4MoEMethod`
  (`fp8.py:204`).
- the `visual.*` / `audio.*` towers are **built, not skipped** — an earlier
  note here claimed otherwise and was wrong. `config.json` declares
  `MiMoV2ForCausalLM`, but `MimoV2ModelArchConfigConvertor` rewrites it to
  `MiMoV2OmniForCausalLM` whenever `vision_config` is present
  (`model_arch_config_convertor.py:509`), and the run log confirms
  `Resolved architecture: MiMoV2OmniForCausalLM`.
- `trust_remote_code` **is** required — the config has an `auto_map` and
  transformers does not know `model_type: mimo_v2` natively. Confirmed by a
  failed `AutoConfig.from_pretrained` without it.

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

## The tiered port, scoped

Ordered, with the GLM analogue for each. None of it should start before the
smoke test says what the KV actually costs.

0. **Multi-group KV accounting** — see the blocker above; this gates everything
   else in `tiered_moe_kv.py`. Smaller than it first looked:
   `UniformTypeKVCacheSpecs.page_size_bytes` is defined as
   `sum(spec.page_size_bytes ...)` (`kv_cache_interface.py:847`), exactly what
   the GLM path computes by hand, so

   ```python
   num_blocks * _pool_bytes_per_block(vllm_config, kv_cache_groups)
   ```

   reproduces GLM's number identically and handles MiMo's two groups for free.
   That replaces the main/indexer/draft classification with a generic formula
   rather than adding a second bespoke path.

1. **Manifest** — `build_mimo_v26_manifest` beside `build_glm_w4a16_manifest`
   (`tiered_moe_manifest.py:454`), with a `_validate_mimo_v26_config` mirroring
   `_validate_glm_w4a16_config:178`. The per-expert byte size drives every
   downstream budget: **measure it off one loaded shard**, do not derive it
   from the mxfp4 packing. Deriving expert/page bytes is what cost four
   launches on GLM this session.

2. **KV plan** — `plan_glm_kv_cache` rejects anything non-MLA at
   `tiered_moe_kv.py:52`. A GQA plan is structurally simpler (no sparse
   indexer, one spec kind) but must encode two things the GLM plan never had:
   asymmetric K/V widths (192/128) and whatever the smoke test shows about
   sliding-window promotion.

3. **Spec-kind classifier** — `_get_tiered_kv_spec_kind` treats "not
   MLAAttentionSpec" as *draft*. With a GQA target every spec is a
   `FullAttentionSpec`, so target-vs-draft has to come from somewhere else
   (layer name, or the owning module). This one silently mislabels rather than
   raising, so it needs a test before it is trusted.

4. **Validator pins** — `config/vllm.py:2318+` hardcodes GLM's
   `max_model_len=400000`, `block_size=64`, `kv_cache_dtype=fp8_ds_mla` and
   layer counts. Parametrise by model family; do **not** loosen them for GLM in
   the process. The requested 250K context trips the `max_model_len` pin.

5. **Draft-cache budgeting** — the fix landed earlier today adds draft specs to
   `fixed_hbm_allocations`. It reads the drafter geometry from a
   `FullAttentionSpec`, so it should carry to MiMo's dflash drafter unchanged,
   but it assumes the drafter config exposes `num_key_value_heads` and
   `head_dim` — MiMo's dflash config does.

## The non-tiered smoke test is a dead end (4 launches)

| job | runner | offload | outcome |
|---|---|---|---|
| 1943437 | V2 | 60 | CUDA OOM, `mxfp4.py:577`, 94.48 GiB resident |
| 1944037 | V2 | 105 | CUDA OOM, 94.47 GiB — a 45 GiB budget change moved 0.01 GiB |
| 1944307 | V1 | 105 | offloader installed, workers die silently after Marlin init |
| 1944556 | V1 | 60 | identical silent death; `ExitCode 1:0`, no signal, no core |

The V2 pair diagnosed a real bug (see below). The V1 pair shows the offload
budget is not the variable: 60 and 105 fail identically. Exit code carries no
signal, so it is neither the OOM-killer nor a segfault — a worker exception that
never reached the log.

It does load far enough to validate the interesting parts:

```
mimo_v2.py:322   Using FLASH_ATTN_DIFFKV for attention.
fa_utils.py:217  Diff-KV with sinks: upgrading FlashAttention 3 -> 4
mxfp4.py:622     Using 'MARLIN' Mxfp4 MoE backend.
```

Asymmetric QK/V heads route to diff-KV and auto-upgrade to FA4; mxfp4 experts
pick up Marlin. Neither needed porting.

Not worth a fifth launch: the KV question it existed to answer is settled above
from the code, and tiered MoE is where this model has to run regardless.

## Status

Download **complete** — 130 shards, 535 GB, every shard in the index present
(the HF API's "132" counts the dflash drafter).

`smoke.sbatch` (job 1943383) submitted: non-tiered, TP4, `cpu_offload_gb=60`,
64K context, eager. It answers whether the weights load at all and what the KV
actually costs per token, which is the input the tiered port needs.

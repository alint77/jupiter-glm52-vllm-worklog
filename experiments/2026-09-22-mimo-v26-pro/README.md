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
| attention | GQA: 128 Q heads, **8 KV heads**, head_dim 192 |
| hybrid pattern | **10 full-attention, 60 sliding-window (window 128)** |
| MTP | **3 layers** (`model.mtp.layers.{0,1,2}`) — MTP3 is exactly right |
| also ships | `dflash/` drafter: 5 layers, SWA 1024, `is_causal: false` |
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

## KV cache: the open sizing question

Per rank at TP4 (2 KV heads/rank), per token per layer = `2 x 2 x 192` elements.

- if the 60 sliding-window layers are **promoted to full attention** (which is
  what happened on GLM — `kv_cache_utils.py:1581`, "page sizes cannot be
  unified"), 250K costs **~13.4 GiB/rank at fp8**, ~26.9 at bf16.
- if the window is **honoured**, the 10 full layers dominate: **~1.9 GiB**.

A 7x swing. This has to be measured, not assumed.

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

## Status

Download running. Integration scoping in progress — nothing implemented yet.

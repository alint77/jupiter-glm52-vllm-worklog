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

## KV cache: the open sizing question

K and V differ in width, so per token per layer per rank (2 KV heads at TP4)
is `2*192 + 2*128 = 640` elements, not `2*2*192`.

- if the 60 sliding-window layers are **promoted to full attention** (which is
  what happened on GLM — `kv_cache_utils.py:1581`, "page sizes cannot be
  unified"), 250K costs **~22.4 GB/rank at bf16**, ~11.2 at fp8.
- if the window is **honoured**, the 10 full layers dominate: **~3.2 GB**.

A 7x swing. Being measured by `smoke.sbatch`, not assumed.

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
- `visual.*` / `audio.*` towers load harmlessly: `load_weights` drops anything
  not in `params_dict`, so the text-only class ignores them.
- `trust_remote_code` **is** required — the config has an `auto_map` and
  transformers does not know `model_type: mimo_v2` natively. Confirmed by a
  failed `AutoConfig.from_pretrained` without it.

## Status

Download **complete** — 130 shards, 535 GB, every shard in the index present
(the HF API's "132" counts the dflash drafter).

`smoke.sbatch` (job 1943383) submitted: non-tiered, TP4, `cpu_offload_gb=60`,
64K context, eager. It answers whether the weights load at all and what the KV
actually costs per token, which is the input the tiered port needs.

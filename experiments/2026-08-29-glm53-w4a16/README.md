# GLM-5.3 W4A16 (group 32) against NVFP4

Benchmarks `JANGQ-AI/GLM-5.3-W4A16` on the tiered MoE path and compares it to
the incoai NVFP4 checkpoint that has been the production target.

## The checkpoint

compressed-tensors `pack-quantized`, int4 symmetric, **group_size 32**,
`actorder: None`, targeting `re:.*mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)$`
with `lm_head`, embeddings, norms, `self_attn`, `shared_experts`, `mlp.gate`,
the dense MLPs and `eh_proj` all ignored. 282 shards, 420 GB, 176,321 tensors.

Two things had to change before it would load.

### group 32 was rejected by three hardcoded guards

`tiered_moe_manifest.py`, `tiered_moe_conversion.py` (twice) and
`compressed_tensors_moe_wna16_marlin.py` all required `group_size == 128`.
Nothing depended on that value:

- Marlin supports it -- `quant_utils.SUPPORTED_GROUP_SIZES` is `[-1, 32, 64, 128]`.
- `_w2_scale_sharding` branches only on `actorder`, never on the group size.
- `runtime_expert_bytes` is derived from the stored tensor sizes, so a smaller
  group simply costs more resident scale bytes and is accounted automatically.

All three now accept Marlin's set. 43/43 manifest tests pass and the existing
checkpoints still resolve unchanged (Int4-Int8Mix g128, NVFP4 g16).

### the published index declares the wrong total_size

`total_size` is 450,925,945,160 against 450,904,205,312 of actual tensor
headers, a 21,739,848 byte overstatement, which trips the truncation guard in
`build_glm_w4a16_manifest`.

The checkpoint is not truncated: 176,321 tensors in the index, 176,321 on disk,
none missing, none extra, and every shard byte-exact against the hub manifest.
So this is publisher bookkeeping. The local metadata is corrected to the summed
value rather than weakening a guard that is doing its job; the published
original is kept at `index-original.json`.

## Footprint against NVFP4

| | W4A16 g32 | NVFP4 |
| --- | --- | --- |
| runtime_expert_bytes | 21,233,672 | 21,233,680 |
| routed_expert_bytes | 407.7 GB | 407.7 GB |
| non_routed_bytes | **43.2 GB** | **57.1 GB** |

Per-expert cost is within 8 bytes: int4 with bf16 scales at group 32 costs the
same as fp4 with fp8 scales at group 16. The difference is **13.9 GB less
non-routed weight, about 3.5 GB per rank**, because this checkpoint quantizes
the MTP block (768 packed expert tensors at layer 78) where NVFP4 leaves it
BF16 at roughly 4.5 GiB per rank.

## Method

`arm-quant-ab.sh` is the real-code decode suite -- 512-token code prompts,
1024-token generations -- which is the only suite in this tree that measured the
placement ranking honestly: the 16K suite contaminates TPOT with chunked-prefill
stalls and the synthetic `random` dataset cannot test anything content-derived.

Both arms use the re-derived GLM-5.3 ranking at 2400 hot slots per rank, so the
placement, concurrency, speculator and prompts are fixed and the checkpoint is
the only variable. `restamp_profile.py` re-fingerprints the profile onto the
W4A16 checkpoint without trimming slots or relabelling the ranking, which is
what `port_profile.py` would have done.

Three paired runs per side: between-run spread on this configuration is about
2.7%, wider than any within-run confidence interval, so a single pair cannot
size a small effect.

**At a fixed 2400 slots the MTP saving shows up as spare HBM, not throughput.**
This run therefore isolates format and kernel speed. If W4A16 holds up, the
follow-up is to spend those 3.5 GB per rank on more hot slots.

## Status

Jobs `1534564`-`1534569` submitted.

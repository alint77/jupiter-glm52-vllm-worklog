# GLM-5.3 NVFP4 through the tiered path

Goal: serve `incoai/GLM-5.3-NVFP4` on this fork's tiered MoE path, with the
cold tier running from Grace over UVA Marlin as it does for W4G64 today.

`incoai/GLM-5.3-NVFP4` — 433 GiB, 141 shards, NVIDIA ModelOpt format
(`quant_method: modelopt`, `quant_algo: NVFP4`, `group_size: 16`), NVFP4
weights with static NVFP4 activation scales and an FP8 KV cache. Base model
`zai-org/GLM-5.3`, which is GLM-5.2's base with different post-training.

## Survey before touching anything

Four findings, in descending order of how much they change the estimate.

### 1. The tiered path and the NVFP4 path already share an experts class

This was the open question and the answer is favourable. The tiered code does
not own a Marlin kernel. `tiered_moe_execution.py:_apply_tier_launch_policy`
takes whatever experts object the layer built, asserts it is a
`MarlinExpertsBase`, and sets a `MarlinLaunchPolicy` on it — that is the whole
of the tier machinery's contact with the GEMM.

`select_nvfp4_moe_backend` maps `NvFp4MoeBackend.MARLIN` to `MarlinExperts`,
and `MarlinExperts(LoRAExpertsMixin, MarlinExpertsBase)` satisfies that
assertion. The NVFP4 scale layout is handled inside Marlin's own weight
preparation (`convert_to_nvfp4_moe_kernel_format` →
`prepare_nvfp4_moe_layer_for_marlin`), which the tiered code never inspects.

So "make Marlin cold support NVFP4" is likely **not** kernel work. The tier
split is about launch policy and physical residency, and both are quant-scheme
agnostic. This needs proving on hardware, not just by reading, but it moves
the expected work from the kernel to the loader.

### 2. Only the routed experts are NVFP4, and layer 78 is not

`hf_quant_config.json` excludes 1,880 modules. Every attention projection,
every `mlp.gate`, every shared expert, the three dense MLP layers, the
indexer, `lm_head` and `embed_tokens` stay BF16. Of the excluded modules,
1,024 are expert modules — **all of them in layer 78**, the MTP head.

Routed layers 3-77, which are exactly the layers the tiered manifest manages
(`routed_layers = tuple(range(3, 78))`), are fully NVFP4. The MTP head's 256
experts are BF16 and are budgeted separately by the planner. This is the
cleanest possible split for the tiered design: the tensors that move between
HBM and Grace are the quantized ones.

### 3. The expert footprint barely changes

| | bits/param | per expert |
| --- | ---: | ---: |
| W4G64 (int4 + fp16 scale per 64) | 4.25 | 19.12 MiB |
| NVFP4 (fp4 + fp8 scale per 16) | 4.50 | 20.25 MiB |

About 5.9% larger per expert, so the placement profile's slot arithmetic
carries over almost unchanged. It does mean fewer hot slots fit in the same
HBM, which the profile refit has to account for.

### 4. The tensor layout is genuinely different, and the manifest is the gate

| | W4G64 (AutoRound) | NVFP4 (ModelOpt) |
| --- | --- | --- |
| weight | `qweight` | `weight` |
| scales | `scales` (fp16, per 64) | `weight_scale` (fp8 e4m3, per 16) |
| zeros | `qzeros` | — (no zero point) |
| global | — | `weight_scale_2` (fp32, per tensor) |
| activation | — | `input_scale` (static) |

`tiered_moe_manifest.py` accepts exactly two quant methods and rejects
everything else:

```
elif quant_method == "auto-round": ...
else: raise ValueError(f"Unsupported GLM W4A16 quant method: {quant_method!r}")
```

It also hardcodes `runtime_expert_format = "vllm_marlin_static_w4a16"`, and
derives `runtime_expert_bytes` by subtracting a per-format constant — 294,912
bytes of `qzeros` for AutoRound, 48 for compressed-tensors — from the stored
size. NVFP4 has no `qzeros`; it has two extra scale tensors instead. That
arithmetic has to be rederived, not adjusted.

## Plan

1. **Manifest**: accept `quant_method: modelopt` with `quant_algo: NVFP4` and
   `group_size: 16`; add the NVFP4 component set; derive
   `checkpoint_expert_bytes` and `runtime_expert_bytes` from it. Fail closed on
   anything else, as the existing validator does.
2. **Config**: whatever `validate_tiered_moe` pins that NVFP4 violates. The
   architecture, model type and shapes all match; the dtype field is
   `bfloat16` as required. Expect the friction to be in the loader, not here.
3. **Placement profile**: a new one is required regardless — the profile
   carries a `config_sha256` fingerprint of its target and the loader fails
   closed on a mismatch. The router also changed with 5.3's post-training, so
   the hot-expert ranking must be re-derived rather than ported.
4. **Bring-up**: dense first inside the tiered contract, then the cold tier.
5. **Verify the claim in finding 1 on hardware**: confirm the layer actually
   builds `MarlinExperts` under this checkpoint and that the tier launch
   policy applies to it.

## Status

- [x] Survey
- [ ] Checkpoint pulled (in flight, ~350 MB/s)
- [ ] Manifest accepts NVFP4
- [ ] Placement profile for the NVFP4 target
- [ ] Dense bring-up under the tiered contract
- [ ] Cold tier over UVA

## Known constraint

NVFP4 has no fast path on GH200. `select_nvfp4_moe_backend` tries
FLASHINFER_TRTLLM, FLASHINFER_CUTEDSL, FLASHINFER_CUTEDSL_BATCHED,
FLASHINFER_CUTLASS and VLLM_CUTLASS first, all of which need SM100, and falls
through to MARLIN on SM90. That is the same kernel class the W4G64 path
already runs, so the expectation is parity rather than speedup — the reason to
do this is the 5.3 weights and the NVFP4 accuracy, not throughput.
`ModelOptNvFp4Config.get_min_capability()` returns 75, so nothing hard-refuses.

# GLM-5.3 NVFP4 through the tiered path

Goal: serve `incoai/GLM-5.3-NVFP4` on this fork's tiered MoE path, with the
cold tier running from Grace over UVA Marlin as it does for W4G64 today.

**Why NVFP4 and not a W4G64 quantization of 5.3.** No AutoRound W4A16 5.3
checkpoint exists yet and producing one in-house takes too long to be
practical. NVFP4 is what is available — and it comes with `incoai/GLM-5.3-DFlash2`,
a drafter from the same quantizer against the same target. That second half
matters more than it looks: Phase 42 refuted DFlash2 on the 5.2 target
specifically because a 5.3-trained drafter reads hidden states that 5.3's
post-training moved, collapsing position-0 acceptance from 87% to 42%. On a
5.3 target that objection disappears, and Phase 42 already priced what DFlash2
needs to win: **3.65 acceptance at ~33 ms/step**, against the 4.19-6.02 it
publishes on a 5.3 target. So this phase is not a throughput play on the
quantization; it is the route to a working speculator.

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

### 3. The expert footprint changes little on disk and a lot in HBM

| | stored | **resident** |
| --- | ---: | ---: |
| W4G64 (AutoRound) | 19.406 MiB | **19.125 MiB** |
| NVFP4 (ModelOpt) | 20.250 MiB | **22.500 MiB** |
| delta | +4.3% | **+17.6%** |

Measured, not estimated: both from the checkpoints' own safetensors headers,
and the runtime figure confirmed by building the manifest.

The runtime gap is larger than the stored gap because
`prepare_nvfp4_moe_layer_for_marlin` does `scales.to(param_dtype)`. The
fp8_e4m3 block scales are held in HBM as bfloat16, doubling 2.25 to 4.50 MiB
per expert. Marlin needs no tile padding for this model: hidden 6144 is a
multiple of 128 so N rounds up to 64, and the 2048 intermediate size is
already aligned, so the weight itself is unchanged.

The consequence is residency. At production's 2,870 hot slots per rank, the
same HBM buys **2,348 NVFP4 slots, about 15% fewer**. This also had to be
right in `runtime_expert_bytes`, which feeds the fail-closed HBM planner:
charging the stored size would under-reserve by roughly 6.3 GiB per rank.

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

### 5. The MTP head is BF16, and that is fine under DFlash2

`hf_quant_config.json` excludes layer 78's experts, and they are stored as
plain BF16:

| MTP layer 78 experts | per expert | x256 | per rank at EP4 |
| --- | ---: | ---: | ---: |
| W4G64 5.2 (int4) | 19.406 MiB | 4.85 GiB | **1.21 GiB** |
| NVFP4 5.3 (BF16) | 72.000 MiB | 18.00 GiB | **4.50 GiB** |

That is +3.29 GiB per rank, worth roughly 150 hot expert slots, on top of the
+17.6% on the routed layers. It only bites if MTP3 is run on this target:
Phase 28 established that the grafted layer 78 **is not instantiated under
DFlash**, costing nothing but a fingerprint match. Since DFlash2 is the point
of this phase, the BF16 MTP head is dead weight on disk rather than in HBM.

An MTP3 control on the NVFP4 target is therefore possible but expensive, and
would need its own deeper profile trim.

## Plan

1. **Manifest**: accept `quant_method: modelopt` with `quant_algo: NVFP4` and
   `group_size: 16`; add the NVFP4 component set; derive
   `checkpoint_expert_bytes` and `runtime_expert_bytes` from it. Fail closed on
   anything else, as the existing validator does.
2. **Config**: whatever `validate_tiered_moe` pins that NVFP4 violates. The
   architecture, model type and shapes all match; the dtype field is
   `bfloat16` as required. Expect the friction to be in the loader, not here.
3. **Placement profile**: a new file is required regardless, because the
   profile carries a `config_sha256` fingerprint of its target and the loader
   fails closed on a mismatch. The *ranking* inside it is deliberately
   deferred: `port_profile.py` carries the GLM-5.2 ranking across,
   re-fingerprinted and trimmed 2,870 to 2,348 slots to pay for the larger
   experts. That ranking is wrong — 5.3's post-training moved the router — but
   placement affects neither loading nor output, so it is fine for bring-up
   and must not be used for any performance number. Re-derive from a routing
   capture on this target once it serves.
4. **Bring-up**: dense first inside the tiered contract, then the cold tier,
   then DFlash2 on this target as the pairing the quantizer intended.
5. **Verify the claim in finding 1 on hardware**: confirm the layer actually
   builds `MarlinExperts` under this checkpoint and that the tier launch
   policy applies to it.

## Status

- [x] Survey
- [x] Checkpoint pulled: 87 shards, 432.90 GiB, index-verified, none missing
- [x] Manifest accepts NVFP4 — 20.250 MiB stored, 22.500 MiB resident per
      expert; AutoRound rebuilds unchanged at 20,054,024 bytes and the 57
      tiered tests pass
- [x] Placeholder placement profile loads through the real fail-closed loader
      at 2,348 slots/rank with the NVFP4 fingerprint accepted
- [ ] Stage to fscratch
- [ ] Dense bring-up under the tiered contract
- [ ] Cold tier over UVA
- [ ] DFlash2 on this target

## Known constraint

NVFP4 has no fast path on GH200, and it costs 17.6% more resident HBM per
expert than W4G64 plus 3.29 GiB per rank if the MTP head is instantiated. The
case for it is not throughput: it is that 5.3 is only available this way, and
that it brings a matching DFlash2 drafter.

`select_nvfp4_moe_backend` tries
FLASHINFER_TRTLLM, FLASHINFER_CUTEDSL, FLASHINFER_CUTEDSL_BATCHED,
FLASHINFER_CUTLASS and VLLM_CUTLASS first, all of which need SM100, and falls
through to MARLIN on SM90. That is the same kernel class the W4G64 path
already runs, so the expectation is parity rather than speedup — the reason to
do this is the 5.3 weights and the NVFP4 accuracy, not throughput.
`ModelOptNvFp4Config.get_min_capability()` returns 75, so nothing hard-refuses.

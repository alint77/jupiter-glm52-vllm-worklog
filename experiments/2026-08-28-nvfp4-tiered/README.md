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

So "make Marlin cold support NVFP4" is **not** kernel work.

**Correction, found while reading further.** The first pass of this finding
claimed the tiered code was quant-scheme agnostic. It is not. Beyond the
launch policy, `tiered_moe_execution.py` builds each tier's quant config
itself, and it is hardcoded to INT4:

```python
quant_config = make_wna16_moe_quant_config(
    w1_scale=components["w13_weight_scale"],
    w2_scale=components["w2_weight_scale"],
    group_size=group_size, num_bits=num_bits, ...)
```

and addresses weights as `components["w13_weight_packed"]` /
`components["w2_weight_packed"]`. NVFP4 needs
`nvfp4_w4a16_moe_quant_config(g1_alphas, g2_alphas, w1_scale, w2_scale, ...)`
— the weight-only variant, since Marlin does not quantize activations — which
additionally requires the global alphas derived from `weight_scale_2`, a
tensor the tiered storage does not currently carry. The component keys differ
too: NVFP4 stores `weight`, not `weight_packed`.

The revised claim: the GEMM needs nothing, the launch policy needs nothing,
and the tiered execution path needs a format branch plus one more stored
component. Bounded, but not zero.

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

### 3. The expert footprint costs about 6% more, on disk and in HBM alike

| | stored | resident |
| --- | ---: | ---: |
| W4G64 (AutoRound) | 19.406 MiB | 19.125 MiB |
| NVFP4 (ModelOpt) | 20.250 MiB | **20.250 MiB** |
| delta | +4.3% | **+5.9%** |

At production's 2,870 hot slots per rank the same HBM buys **2,710 NVFP4
slots, 5.6% fewer**. Marlin needs no tile padding for this model: hidden 6144
is a multiple of 128 so N rounds up to 64, and the 2048 intermediate size is
already aligned, so the weight is unchanged at 18 MiB.

**This finding was first recorded wrong and is corrected here.** The original
reading was that `prepare_nvfp4_moe_layer_for_marlin` does
`scales.to(param_dtype)` and therefore holds the fp8_e4m3 block scales
resident as bfloat16, doubling 2.25 to 4.50 MiB per expert, for a +17.6%
penalty and 2,348 slots. That conversion exists only to permute them.
`nvfp4_marlin_process_scales` then packs the result into the S0E5M3 layout and
ends on `view(torch.float8_e4m3fn)` followed by `[:, 1::2]`, returning one
byte per scale. The error was to read the first transform and stop.

The consequence matters for the argument, not just the arithmetic: NVFP4 was
written up as a meaningful residency loss on GH200, worth taking only for
accuracy. It is close to free.

The number still had to be right in `runtime_expert_bytes`, which feeds the
fail-closed HBM planner. It was caught by that planner: the tiered path
rejected the converted expert with "Converted w13_weight_scale does not match
its final layout", the schema asserting bfloat16 against fp8 reality.

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
+5.9% on the routed layers. It only bites if MTP3 is run on this target:
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
   re-fingerprinted and trimmed 2,870 to 2,609 slots to pay for the larger
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
- [x] Manifest accepts NVFP4 — 20.250 MiB stored and resident per expert;
      AutoRound rebuilds unchanged at 20,054,024 bytes and the 57 tiered
      tests pass
- [x] Placeholder placement profile loads through the real fail-closed loader
      at 2,609 slots/rank with the NVFP4 fingerprint accepted
- [ ] Stage to fscratch (in flight)
- [ ] `tiered_moe_execution.py`: format branch for the tier quant config,
      carry `weight_scale_2` through tier storage
- [x] Dense NVFP4 baseline: reaches Marlin on SM90 and OOMs on capacity,
      confirming the format works here and the tiered path is necessary
- [x] **Serving under the tiered contract with the UVA cold tier**, at 2609,
      2400, 2200 and 1900 hot slots/rank
- [ ] Re-derive the placement ranking on this target
- [ ] DFlash2 on this target
- [ ] Correctness eval

## Result: it serves

Round nine, commit `1c0998a09a`, four allocations, four hot-slot counts. **All
four serve GLM-5.3 NVFP4 through the tiered path**, with coherent output:

```
Using 'MARLIN' NvFp4 MoE backend
quantization=modelopt_fp4      kv_cache_dtype=fp8_ds_mla
tiered_moe_backend: uva        GPU KV cache size: 400,064 tokens
Streamed 4800 routed experts into final tier
Tiered MoE observed HBM reserve: 12.80 GiB free (minimum 5.59 GiB)
```

`4800` is 75 routed layers x 64 experts per rank at EP4, so every expert went
through the new conversion path. The fail-closed HBM audit passed with 12.80
GiB to spare at 2,609 slots per rank, which means the ceiling is higher than
anything tested here.

| hot slots/rank | serves | prompt continuation |
| ---: | --- | --- |
| 2609 | yes | ` Paris. Distance from London to Paris is 344 km...` |
| 2400 | yes | ` Paris. Distance from London to Paris is 344 km...` |
| 2200 | yes | ` Paris. Distance from London to Paris is 344 km...` |
| 1900 | yes | ` Paris. Distance from London to Paris is 343 km...` |

**Residency was never the binding constraint.** Nine rounds of a four-way
parallel sweep failed identically at every slot count, every time, which said
so from the first round; the sweep's value was ruling residency out
immediately rather than bisecting toward it.

This is a smoke test, not an evaluation. The output is coherent English and
plausible code, which is what distinguishes a working GEMM from a broken one.
It does not match the project's 5.2 golden string, and should not: different
model, different quantization. A real correctness gate needs an eval.

## The DFlash2 acceptance investigation

Our 16K-code harness gave DFlash2 3.323 acceptance at width 8. The model card
reports 5.94 on GSM8K. Replicating the card's protocol -- GSM8K, chat
template, T=1.0, top_p 0.95, natural EOS, 4096 max tokens, 64 samples --
settled where the gap was:

| | our 16K suite | card protocol | card | after fixes |
| --- | ---: | ---: | ---: | ---: |
| MTP (7 draft tokens) | — | **4.967** | 5.12 | 4.912 |
| DFlash2 | 3.323 | **2.815** | 5.94 | **4.029** |

**MTP reproducing the card to within 3% validated the harness** and localised
the fault: sampling, chat template, EOS handling, prompt distribution, target
model and the tiered stack were all sound, and the problem was specific to
DFlash2. Note the card's MTP baseline also proposes seven tokens, so its
comparison is width-8 against width-8; ours had been width-4 against width-8.

Normalised for width the diagnosis was sharper still: our MTP3 sat at 68.8% of
its ceiling against the card's MTP7 at 64.0%, while DFlash2 at the *same*
width 8 managed 41.5% against the card's 74.2%.

The per-position profile named the mechanism. Acceptance fell geometrically at
a near-constant 0.714 ratio per position (stdev 0.020) -- the signature of
plain autoregressive decay, when DFlash2's entire premise is that its two-tap
convolutions and candidate selector prevent exactly that.

### Root cause: a backport onto a base that predates its prerequisites

DFlash2 landed upstream on 2026-08-20. This fork's base is 2026-07-16. Between
those dates upstream shipped several core DFlash fixes that #52816 assumes,
and cherry-picking DFlash2 alone left it running on a DFlash core missing five
weeks of its own bugfixes. Auditing all 1704 upstream commits since the base,
by message, by path, and by content presence -- ancestry is useless here since
cherry-picks change hashes -- found:

| commit | present | effect |
| --- | ---: | --- |
| #53336 / #53002 | 50% | FlashAttention metadata built from the **target's** head geometry, not the draft's |
| #51256 | 0% | DFlash needs K extra scheduling slots; the budget reserved none |
| #44492 | 28% | draft `seq_lens_cpu_upper_bound` not populated |
| #50065 | 0% | query buffers vs cudagraph padding — **ruled out**, see below |
| #48524 | 50% | `fc` sizing — our `fc` is verified correct at 36,864 |

#53336 is the substantive one. Our target is MLA with head_dim 192 and an
effective head size of 576; the DFlash2 draft is dense with head_dim 128, 64
query heads and 8 KV heads. Describing draft attention with target geometry
degrades the draft without touching output, because verification rejects every
bad token. It also explains the separate c4 failure, `scheduler_metadata must
have shape (metadata_size)`.

#50065 was checked rather than assumed and does not apply: `max_query_tokens`
is 1 x (1+7) = 8 at c1 and our cudagraph capture size is also 8, so the fix's
`max()` is a no-op. Same at c4, 32 against 32.

**The three fixes recovered 2.815 to 4.029, +43%, with MTP unchanged at 4.912
against its pre-fix 4.967** -- so they were surgical. A 32% gap to the card
remains, and the two unapplied audit entries deserve re-examination before
blaming the GB300/FlashAttention-4 configuration difference: #48524 because
DFlash2 has 6 hidden and 6 target layers, so a wrong code path still yields a
right-sized tensor, and #50487 because it changes which hidden state is tapped
as the aux input, which was dismissed on its Kimi-K3 title alone.

## What it took

Nine rounds, each one integration gap, none of them in a kernel:

| round | stage reached | gap |
| --- | --- | --- |
| a | `create_weights` | full expert params allocated beside the tier buffers |
| b | import | `NvFp4MoeBackend` not imported |
| c | `load_weights` | placeholders had no `weight_loader` |
| d | conversion | no NVFP4 component table |
| e | conversion | staged-byte check was a binary conditional |
| f | conversion | dispatch sat behind an INT4-only backend check |
| g | Marlin repack | expert count read from the layer, not the tensor |
| h | tier storage | scale dtype recorded as bf16 when it is fp8 |
| i | model forward | `apply` never dispatched to the tiered kernels |

Three hooks make a quant method a tiered participant: `create_weights` to
register placeholders instead of real parameters,
`process_weights_after_loading` to build the tier kernels, and `apply` to
route the forward. Missing any one of them fails late and unhelpfully.

## Still open

- **The placement ranking is a GLM-5.2 placeholder.** No performance number
  from this configuration means anything until it is re-derived from a routing
  capture on this target.
- **DFlash2 on this target**, which is the reason for the phase. Phase 42
  priced it: 3.65 acceptance at ~33 ms/step to beat MTP3.
- **The hot-slot ceiling**, which is above 2,609 and unmeasured.
- **A real correctness gate.** Coherent output is not an eval.

## Known constraint

NVFP4 has no fast path on GH200 and costs 5.9% more resident HBM per expert
than W4G64, plus 3.29 GiB per rank if the MTP head is instantiated. The case
for it is not throughput: it is that 5.3 is only available this way, and that
it brings a matching DFlash2 drafter.

`select_nvfp4_moe_backend` tries
FLASHINFER_TRTLLM, FLASHINFER_CUTEDSL, FLASHINFER_CUTEDSL_BATCHED,
FLASHINFER_CUTLASS and VLLM_CUTLASS first, all of which need SM100, and falls
through to MARLIN on SM90. That is the same kernel class the W4G64 path
already runs, so the expectation is parity rather than speedup — the reason to
do this is the 5.3 weights and the NVFP4 accuracy, not throughput.
`ModelOptNvFp4Config.get_min_capability()` returns 75, so nothing hard-refuses.

# DSpark speculator for GLM-5.2

`RedHatAI/GLM-5.2-speculator.dspark` — a DSpark draft trained against GLM-5.2,
published 2026-07-24. This is the first GLM-targeted DSpark checkpoint; the three
architectures registered in vLLM before it were DeepSeek-V4, Qwen3-8B and
Gemma4-12B, none of which apply here.

## What the checkpoint is

| | |
| --- | --- |
| Format | speculators (`speculators_model_type: dspark`, v0.7.0.dev108) |
| Weights | 6.31 GB bf16, single `model.safetensors` |
| Backbone | 3 layers, `model_type: qwen3`, hidden 6144, 64 heads, head_dim 64, sliding_window 2048 |
| Block size | **8** (so `num_speculative_tokens >= 8`, enforced in `speculative.py`) |
| Draft vocab | 154,880 — same as target, so no d2t/t2d remap |
| Aux layers | `[2, 20, 39, 58, 75]` from the target |
| Heads | Markov logit-bias head (`vanilla`, rank 256) + confidence head |
| `sample_from_anchor` | **true** — anchor-as-first, N query slots |
| Verifier it was trained against | `RedHatAI/GLM-5.2-NVFP4-FP8` (`GlmMoeDsaForCausalLM`) |

DSpark is semi-autoregressive *block* drafting: it reuses DFlash's machinery
(context-KV precompute plus a query-block forward) to draft a whole block in one
parallel pass, then injects intra-block dependency with a sequential Markov head.

The README marks it "preliminary and subject to change", trained one epoch, and
**validated only on B200 — other hardware pending**. Our target is also W4A16
rather than the NVFP4-FP8 it was trained against, so acceptance has to be
measured, not assumed.

## Integration audit against this branch

Checked before running anything.

| Requirement | Status |
| --- | --- |
| DSpark support present at all | Yes — arrived with upstream base `d08eebad1`, not added by this branch |
| Config translation | `algos.py:update_dspark` rewrites `architectures` to `Qwen3DSparkModel`, so the checkpoint's `DSparkDraftModel` label does **not** route to the DeepSeek-V4 loader |
| `sample_from_anchor` | **BLOCKER in our base** — see below |
| Target aux hidden states | `deepseek_v2.py` (which implements `GlmMoeDsaForCausalLM`) has `set_aux_hidden_state_layers` and emits them in `forward` |
| Aux layer plumbing | V2 runner sets `use_aux_hidden_state_outputs` for `dspark`; `eagle3_utils.get_eagle3_aux_layers_from_config` reads `eagle_aux_hidden_state_layer_ids` |
| Runner | V2 only — no V1 path exists. Our c4/DCP4 config already runs V2 |
| Target layer ids | translated to `[1, 19, 38, 57, 74]`, all valid for our 78-layer target |

### The blocker: `sample_from_anchor`

Our base hardcodes the opposite of what this checkpoint needs:

```python
# our base, algos.py:update_dspark
pre_trained_config["dspark_bonus_anchor"] = True   # -> sample_from_anchor False
```

The checkpoint sets `sample_from_anchor: true`. Under our base it would be loaded
with the 1+N fill-in layout instead of anchor-as-first. Per the speculator's own
docstring that changes both the number of query slots and where sampling happens,
so it produces wrong output rather than merely lower acceptance.

Upstream `642076d26` ("Support loading sample_from_anchor flag from speculators
config", #48639) replaces the hardcode with a config read. It is a 3-file, 24-line
commit and cherry-picks cleanly onto this branch. Verified afterwards by running
the translation offline:

```
architectures: ['Qwen3DSparkModel']
sample_from_anchor: True
block_size: 8   mask_token_id: 154856   markov_rank: 256
eagle_aux_hidden_state_layer_ids: [2, 20, 39, 58, 75]
target_layer_ids: [1, 19, 38, 57, 74]
```

### Known gap: the tiered planner does not budget the draft

`tiered_moe_physical.py` accounts for draft weights only when
`speculative.method == "mtp"`; for `dspark` it budgets zero. The ~6.3 GB bf16
draft (~1.6 GB/rank under TP4, plus its sliding-window KV) is therefore invisible
to the physical plan, and the post-warmup audit is fail-closed.

Worked around for the smoke by raising `TIERED_MOE_HBM_RESERVE_GB` to 16, which
costs hot expert slots but keeps the audit satisfied. A proper fix would teach
the planner about non-MTP drafts.

### Contract constraints that shape the smoke

`VllmConfig.validate_tiered_moe` pins `max_model_len=400000`,
`max_num_batched_tokens=8192`, `block_size=64`, `fp8_ds_mla`, TP4 + EP + NUMA,
and `max_num_seqs` 1–4 with DCP required above 1. So the smoke cannot use a
short context to reduce risk — `max_model_len` sizes the KV cache, not the
prompt, so short smoke prompts are still fine, but the KV allocation is the full
400K one.

## Configuration

```bash
sbatch --job-name=dspark-smoke \
  agent_space/experiments/2026-07-25-dspark/job-smoke.sh dspark-t8 8
```

Target is the plain `GLM-5.2-W4A16-55c92ae` — the grafted FP8 MTP layer is
unused under DSpark. `num_speculative_tokens=8`, c1, DCP1, V2 runner, graph
capture size 9 (block + 1).

## Jobs

| Job | Config | State |
| --- | --- | --- |
| 1042092 | t=8, c1, DCP1, 16 GB reserve | submitted |

## Results: five blockers cleared, one structural blocker remains

Five smoke attempts, each surfacing a distinct real defect rather than a
misconfiguration.

| # | Job | Origin | Defect | Fix |
| --- | --- | --- | --- | --- |
| 1 | — | stale base | `dspark_bonus_anchor` hardcoded, so a `sample_from_anchor: true` checkpoint loads the 1+N fill-in layout | cherry-pick upstream `642076d26` (#48639) |
| 2 | 1042092 | our config | `Placement profile config fingerprint does not match` | target must stay `GLM-5.2-W4A16-FP8-MTP` (profile carries its `config_sha256` `1c6c983c...`) |
| 3 | 1042100 | config wiring | `No valid attention backend`: the dense draft (head_size 64, sliding window) inherited the target's MLA-only `fp8_ds_mla` | `speculative_config.kv_cache_dtype: auto` |
| 4 | 1042147 | **our contract** | `validate_tiered_moe` rejected the draft-derived `VllmConfig` | scope the dtype check to ignore the configured draft dtype |
| 5 | 1042261 | stale base | fc sized `[6144, 18432]` vs checkpoint `[6144, 30720]` — 3x6144 vs 5x6144 | cherry-pick upstream `a7d00ec05` (#48524) |
| 6 | 1042319 | stale base | KV page-size unification: MLA target + SWA draft | cherry-pick upstream `e18f0037a` (#48776) |
| 7 | 1042647 | **our tiered KV** | `Tiered GLM KV allocation only supports MLA cache specs` | classify the draft spec, charge its pages into `bytes_per_block` |
| 8 | 1042864 | **our planner** | fail-closed HBM audit, 1.64 GB short | unresolved — see below |

Blocker 5 is worth noting: the speculators translation writes `target_layer_ids`
as a top-level attribute, but `qwen3_dflash.py` read it only from the nested
`eagle_config`/`dflash_config` dicts and silently fell back to the *draft's*
`num_hidden_layers`. That default is 3, which coincidentally matches Eagle3's
usual three aux layers — so existing DSpark checkpoints never trip it. Ours has
3 draft layers but 5 aux layers, which breaks the coincidence. The upstream fix
is titled for exactly this case.

### Blocker 6 was already fixed upstream

`unify_kv_cache_spec_page_size` requires a smaller KV spec either to divide the
maximum page or to be paddable via `indexes_kv_by_block_stride`. The DSA indexer
is neither once a dense draft sets the maximum. No configuration escapes it: the
draft page is `131072 * b` bytes (a power of two for any dtype) while the indexer
page is `64 * (128 + 4) = 8448 = 2^8 * 33`.

Upstream **#48776 "Support sparse-MLA targets with SWA drafts"** (merged
2026-07-23) fixes exactly this, and its validation section uses this very
checkpoint: *"RedHatAI/GLM-5.2-speculator.dspark: 100/100 requests, 236.19 output
tok/s, 40.44% draft-token acceptance, mean accepted length 3.83."* It promotes
only the draft's allocation spec to `FullAttentionSpec` at the target's block
size, leaving draft attention sliding-window. Cherry-picked; the nine KV
unification tests pass and the server then reached weight loading.

### The remaining blocker: the planner does not budget the draft

```
Tiered MoE observed free HBM is below the runtime reserve:
13,357,547,520 bytes available, 15,000,000,000 required.
```

Short by 1.64 GB, which is the DSpark draft's per-rank weight share
(6.31 GB / 4 = 1.58 GB) plus change. This is the gap noted above:
`tiered_moe_physical.py` budgets draft weights only when
`speculative.method == "mtp"`.

**Raising `TIERED_MOE_HBM_RESERVE_GB` cannot fix it.** The audit computes

```python
required_free = max(MINIMUM_OBSERVED_HBM_RESERVE_BYTES,
                    planned_reserve - _OBSERVED_HBM_RESERVE_TOLERANCE_BYTES)
```

so `required` scales 1:1 with the planned reserve — and so does `available`,
because the planner simply places fewer hot experts to hit the reserve. The
deficit stays ~1.64 GB at any reserve value. The 16 GB workaround used for
attempts 1-7 was never going to succeed; only accounting fixes this.

The fix is to extend the draft-weight accounting in `tiered_moe_physical.py`
beyond the `method == "mtp"` special case: derive the draft's per-rank parameter
bytes from its model config and subtract them from the HBM budget before placing
hot experts. Bounded work, but it changes the fail-closed planner the project's
memory safety rests on, so it wants deliberate review rather than a quick patch
at the end of a session.

### The recurring pattern

Three of the eight blockers (4, 7, 8) share one root cause: **the tiered code
assumes it is the only thing allocating HBM or validating configuration.**
`validate_tiered_moe` re-ran on the draft's derived config; the tiered KV
allocator rejected any non-MLA spec; the physical planner budgets only an MTP
draft. Each was patched individually here. If DSpark becomes a real option, that
assumption deserves one proper fix rather than three.

## c1 result: DSpark runs, and loses to MTP3

Jobs 1043105 (t=8) and 1043106 (t=7), c1, DCP1, V2 runner, 2,720 hot slots/rank
via `VLLM_TIERED_MOE_PROFILE_CAP` at the 7 GB reserve. **Both start and both
reproduce the exact deterministic completion** ` Paris. Distance from Paris to
Lyon is`. DSpark on GLM-5.2 works on this stack.

| | t=8 | t=7 | MTP3 c1/q4 (V2) |
| --- | ---: | ---: | ---: |
| acceptance length | **3.98** / 8 | 2.76 / 7 | ~2.9 / 4 |
| acceptance rate | 37.30% | 25.11% | — |
| position 0 | 78.12% | 65.26% | ~90% |
| realistic mean TPOT | 10.65 / 10.90 ms | 10.45 / 10.32 ms | — |
| implied decode tok/s (1/TPOT) | ~93 | ~96 | **106.08** |
| realistic end-to-end tok/s | 84.67 / 83.39 | 87.51 / 88.71 | **95.44** |

DSpark is **~10-12% slower than MTP3** at c1 on either setting.

t=8 is clearly the right configuration despite the model card's
`--spec-tokens 7`: acceptance length 3.98 vs 2.76. And 3.98 of 8 is consistent
with the card's claimed 3.4-3.8 average, so the checkpoint behaves as
advertised — it is our step economics that do not suit it.

### Why, and why it was predictable

DSpark verifies **9 tokens per step** (block 8 + anchor) against MTP3's 4. The
[critical-path review](../2026-07-25-c4-mtp3-critical-path/README.md) measured
the routed MoE as only **35% fixed weight streaming, 65% proportional to token
count** (9.02 ms + 1.057 ms/token). Widening the verify batch from 4 to 9 tokens
therefore adds roughly 5 ms/step of routed MoE alone — far more than a higher
acceptance length repays.

So the same measurement that explained why c=4 yields 1.35x rather than 4x also
predicts DSpark's failure mode here: on a W4A16 tiered MoE target whose verify
cost scales nearly linearly with batch, a wide speculative block is the wrong
shape. MTP3's narrow 4-token verify is better matched. A target with a flatter
verify curve (the NVFP4 B200 setup DSpark was trained and validated against)
would see the opposite.

### Verdict

Not worth pursuing for this deployment:

1. It is slower than the qualified MTP3 default at c1.
2. c4 — the actual target regime — is unreachable anyway: upstream #48392
   (DCP for DFlash/DSpark) is open, and #48381 explicitly fails fast on DCP.
3. The economics work against it structurally, not incidentally, so a better
   checkpoint would not obviously flip the result; a narrower block might.

Worth re-testing only if DSpark gains a small block size, or if the routed-MoE
per-token slope drops substantially.

## State

Cherry-picked (local, unpushed): `642076d26`, `a7d00ec05`.
Local fix: `validate_tiered_moe` draft-dtype scoping in `vllm/config/vllm.py`.
Checkpoint downloaded to `models/GLM-5.2-speculator-dspark` (6.3 GB).
Launcher and smoke job in this directory; both re-runnable unchanged once the
KV page-size blocker is resolved.

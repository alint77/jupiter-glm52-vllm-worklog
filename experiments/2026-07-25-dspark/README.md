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

## Results

Pending.

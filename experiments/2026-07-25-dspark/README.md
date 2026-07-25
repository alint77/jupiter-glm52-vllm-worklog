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
| 6 | 1042319 | **structural** | KV page-size unification — see below | unresolved |

Blocker 5 is worth noting: the speculators translation writes `target_layer_ids`
as a top-level attribute, but `qwen3_dflash.py` read it only from the nested
`eagle_config`/`dflash_config` dicts and silently fell back to the *draft's*
`num_hidden_layers`. That default is 3, which coincidentally matches Eagle3's
usual three aux layers — so existing DSpark checkpoints never trip it. Ours has
3 draft layers but 5 aux layers, which breaks the coincidence. The upstream fix
is titled for exactly this case.

### The remaining blocker

```
NotImplementedError: Layer model.layers.0.self_attn.indexer.k_cache: page size is
not divisible by the maximum page size and cannot be padded.
```

`unify_kv_cache_spec_page_size` requires every KV spec smaller than the maximum
either to **divide** it (so its block size can be scaled up) or to be an
`AttentionSpec` with `indexes_kv_by_block_stride` (so it can be padded). The DSA
indexer satisfies neither once a dense draft sets the maximum page.

This is why MTP works and DSpark does not: the MTP draft is an MLA layer with
the *same* spec as the target, so unification never triggers. DSpark's draft is
dense attention, and its page becomes the new maximum.

The arithmetic shows no configuration escapes it:

- draft page = `block 64 x 16 kv heads (TP4) x head_dim 64 x 2 (K,V) x b`
  = `131072 * b` bytes — **always a power of two**, for any dtype `b`
- DSA indexer page = `block 64 x (128 index_head_dim + 4 scale)` = `8448`
  = `2^8 x 33`

A power of two is never a multiple of 33, so neither `auto` (bf16) nor `fp8`
draft KV can divide. Changing the draft dtype cannot fix this, and block_size is
pinned at 64 by the tiered contract.

### Recommendation: rebase before going further

Three of the six blockers (1, 3's support, 5) are upstream fixes our base
predates — this branch forked at `d08eebad1` on 2026-07-16 and DSpark has been
actively developed since (275 upstream commits, including
`76bf55240`, `642076d26`, `a7d00ec05`, `4a394bfcd`).

The remaining blocker most likely also has upstream work behind it:
`f3a920a07 [Core][DSV4] Compact MXFP4 indexer KV cache and packed group overlays
(#48993)` touches precisely the indexer KV layout and page grouping, and
upstream has since rewritten both files heavily —
`kv_cache_utils.py` (+296/-213) and `indexer.py` (+317).

Cherry-picking further is the wrong shape of work here: both files carry
substantial local modifications from the tiered DCP port, so the sensible move
is a rebase onto current upstream, then re-run this smoke. That is a large,
risky operation on a branch with 4,462 lines of local changes across 41 files,
so it is a decision for the human rather than something to attempt unilaterally.

The alternative — giving the DSA indexer `indexes_kv_by_block_stride=True` — is
not safe without confirming the backend genuinely indexes by block stride; it is
a correctness-affecting flag, not a sizing hint.

## State

Cherry-picked (local, unpushed): `642076d26`, `a7d00ec05`.
Local fix: `validate_tiered_moe` draft-dtype scoping in `vllm/config/vllm.py`.
Checkpoint downloaded to `models/GLM-5.2-speculator-dspark` (6.3 GB).
Launcher and smoke job in this directory; both re-runnable unchanged once the
KV page-size blocker is resolved.

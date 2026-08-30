# DFlash2 upstream audit: the 32% acceptance gap against upstream vLLM

Phase 48 (audit; no cluster time spent - every finding below is from git,
`gh`, and the checkpoint configs). This audit follows up on Phase 43, which
reopened DFlash2 on a GLM-5.3 target after three missing upstream fixes
(#53336/#53002, #51256, #44492) recovered acceptance from 2.815 to 4.029 and
left a 32% gap to the card's 5.94. The question here is narrower: **is
anything else missing or divergent between this fork's DFlash2 port and
upstream vLLM**, so that the remaining gap is blamed on the right thing?

Audit date: 2026-08-30. Source tree: `dflash2-backport` at `af4aa8c493`.
Upstream reference: `origin/main` fetched to `680e2177e4` (2026-08-29).
All PR facts pulled via `gh` from vllm-project/vllm, all code diffs via
`git diff HEAD origin/main` over the DFlash pathspec.

## Method

1. Direct file diff of every DFlash-touching path against `origin/main`:
   `qwen3_dflash.py`, `dflash2/speculator.py`, `dflash/` (the DFlash2 parent
   class), `spec_decode/speculator.py`, `gumbel.py`, `eagle3_utils.py`,
   registration, and the spec-decode tests.
2. `gh` census of every merged and open PR matching dflash/DFlash/DSpark
   (40 merged, 30+ open reviewed), each judged against our exact
   configuration: GLM-5.3 MLA target (`GlmMoeDsaForCausalLM`), DFlash2
   all-SWA draft, FlashMLA sparse backend, TP4/EP4, DCP1/DCP4, tiered MoE,
   prefix caching on, V2 runner, greedy drafting.
3. Config-level analysis of `incoai/GLM-5.3-DFlash2` (local copy at
   `../models/GLM-5.3-DFlash2/config.json`, byte-identical to the HF
   original) and of the GLM-5.3 target configs.

## Finding A: the backported runtime code is clean

`vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py` differs from upstream
main by **exactly one hunk** - the #54282 Gumbel-salt change - and that hunk
**is provably inert in every acceptance measurement taken so far**:

- Our arms draft greedily (`draft_sample_method` defaults to `greedy`,
  `vllm/config/speculative.py:282`), so `draft_logits is None`, and the
  selector's noise draw sits inside `if temp != 0.0` in the kernel. The draft
  never draws a noise vector; only the target's verification sampling does,
  which the salt does not touch.
- Therefore #54282 cannot explain any part of the 4.029-vs-5.94 gap under the
  protocols we ran. It becomes relevant only if `probabilistic` drafting is
  ever evaluated (bundle it with #53017, the stride fix in the same kernels;
  keep our deliberate fp32 `draft_logits_spec` from the reverted #50910).

Other correctness questions resolve in the port's favor:

| Question | Verdict | Evidence |
| --- | --- | --- |
| Draft causality | Correct | Config's top-level `"is_causal": false` is honored by the local resolver (`qwen3_dflash.py:58-76`); draft runs non-causal, same as upstream |
| `fc` sizing (#48524) | No-op here | 6 aux layers x 6144 = 36,864; upstream's refactor computes the same value when `num_target_layers == len(target_layer_ids)` |
| Draft/target layer-count mismatch (#48113) | In base | Merged 2026-07-10, ancestor of our base `d08eebad1`; content is a `ValueError` in `combine_hidden_states`, unreachable at our 6==6 geometry |
| Hybrid SWA+full drafting (#47914) | In base, N/A | Its `_dflash_needs_multi_kv_group` gate (`config/vllm.py`) requires a *mixed* draft; all-SWA is deliberately excluded |
| Speculators-format loaders (#53797, #42376, #48639) | N/A | The incoai checkpoint is plain HF format (see checkpoint facts below); it loads through the architecture registry today |
| Rejection-sampler -1 placeholders (#50939) | N/A now | We run `rejection_sample_method="standard"` + greedy, already guarded by in-base #46533; required the moment block verification is enabled |
| Scheduler/query sizing (#51256, #44492, #50065, #53336) | All ported | `dcf5ceb2b9`, `c8a723e0c9`, earlier audit; #50065 verified no-op |
| #50487 aux tap | Kimi-specific | Changes Kimi-K3-AttnRes target code GLM does not have; the generic aux-tap question is a separate hypothesis (below) |

## Finding B: the draft's RoPE layout diverges from upstream vLLM

**This is the audit's headline finding and the top candidate for the 32%.**

Upstream `load_dflash_model` (`vllm/v1/worker/gpu/spec_decode/dflash/utils.py`,
content from #51655 / commit `6adad08767`, merged 2026-08-14, i.e. **before**
DFlash2 landed) copies the *target's* RoPE layout into the draft head before
constructing it:

```python
is_neox_style = dflash_target_rope_is_neox_style(target_model)
if is_neox_style is not None:
    draft_model_config.hf_config.is_neox_style = is_neox_style
```

with the docstring: *"A DFlash head must rotate Q/K the way the target it was
distilled against does, and a mismatch is silent - acceptance collapses but
nothing errors and the output stays correct."*

For our target, the answer is unambiguous:

- `GlmMoeDsaForCausalLM` is a `DeepseekV2ForCausalLM` subclass served from
  `deepseek_v2.py`, whose MLA attention rope is hardcoded
  `is_neox_style=False` (`deepseek_v2.py:549`, also `:1082`), and whose
  indexer rope is `is_neox_style = not indexer_rope_interleave`
  (`deepseek_v2.py:1129`) = False, since GLM-5.3 sets
  `indexer_rope_interleave: true`.
- The target config confirms: `rope_interleave: true`,
  `qk_rope_head_dim: 64` - every rotary in the GLM-5.3 target is
  **interleaved**, so upstream's probe returns False no matter which rotary
  module its scan hits first (the probe's module-order bug, open PR #54373,
  is moot for this target - though #54373 must ship with any #51655 port).

This fork's draft head has no `is_neox_style` parameter at all:
`DFlashQwen3Attention` builds its rope at `qwen3_dflash.py:201` via
`get_rope` with no override, and `get_rope` defaults `is_neox_style=True`
(`rotary_embedding/__init__.py:33`). **Our fork rotates every drafted Q/K
NeoX-style; upstream vLLM serves this identical checkpoint interleaved.**

Honest caveats:

- The card's 5.94 was measured on **SGLang on 4x GB300** (FA4 draft
  attention), not vLLM - so this proves a divergence from upstream vLLM, not
  from the card's stack per se. What layout the draft was *trained* with is
  not recoverable from its config (`model_type: qwen3`, `rope_type: default`
  reads as NeoX in HF conventions; the target reads interleaved).
- Mirroring evidence upstream's mechanism works when layouts agree: #52816's
  own benchmark hits GSM8K acceptance 5.34 on a Qwen3.8-27B DFlash2 (native
  target, rope copies NeoX, matching Qwen training).
- A wrong-but-consistent rotation need not collapse acceptance to 1.0: on a
  6-layer non-causal SWA drafter it degrades position information while
  preserving coarse locality - consistent with a moderate shortfall like
  4.03-vs-5.94 and with the near-constant geometric per-position decay
  (0.714) Phase 43 measured.

**Decisive, cheap experiment (Plan P1):** force the draft's rope interleaved
locally and rerun the card-protocol replication. If acceptance moves toward
5.94 the gap is solved; if it collapses, NeoX was right and the gap is
elsewhere. Either result is decisive for the divergence question.

## Finding C: the local draft ctx-KV insert can write into the null block

The draft's context-KV insertion kernel in `dflash/speculator.py` (base class
of `DFlash2Speculator`) is missing two guards upstream added via #51538
(commit `97388c44f9`, "DSV4 sparse MLA e2e", 2026-08-15 - a title the Phase 43
audit's five-candidate list skipped because it says DSV4/DSpark):

- **Null-block guard**: evicted sliding-window context rows and rejected
  suffix rows can resolve to physical block 0 and write draft KV there.
  Upstream masks both to `PAD_SLOT_ID` ("Block 0 is the null block... Neither
  kind of row may write draft KV into physical block 0"). Our drafter is
  all-SWA at window 2048 - any context beyond 2048 tokens produces evicted
  rows, which is every long-context shape we serve.
- **CP-interleave-aware slot math** (`cp_local_slot`, `block_size * CP_SIZE`
  division): moot at DCP1 where all acceptance was measured; required before
  trusting DFlash2 at c4/DCP4.

Not the GSM8K-gap answer (prompts sit inside the 2048 window, so no eviction
fires), but a latent quality/correctness hazard for the 16K-code harness
(whose acceptance, 3.32, sits well below GSM8K's 4.03) and for production
long-context traffic.

## Finding D: applies from the open-PR sweep

- **#50457** (open): with an all-SWA drafter and prefix caching on - our exact
  production configuration - the drafter's KV group never serves partial
  hits, so **the drafter re-reads the full context on every agentic turn**.
  Not an acceptance fix; a production-throughput lever. High conflict risk
  against our DCP port (`kv_cache_coordinator.py`, `kv_cache_utils.py`) -
  port surgically, never as a merge.
- **#53292** (open): `num_speculative_tokens` is missing from
  `SpeculativeConfig.compute_hash()` for dflash/dspark. Four lines. Our arms
  sweep widths (MTP3/MTP7/DFlash2 at 7 and 8) from one checkout with shared
  persisted cache roots, so a wider run can inherit a narrower run's captured
  plan. Cheap insurance; apply before the next width sweep.
- #53096 (log hygiene) folds into any future dflash-speculator change.

## Checkpoint and card facts (recorded for future replication)

Local config `../models/GLM-5.3-DFlash2/config.json` is byte-identical to the
HF original. Structure: `DFlash2DraftModel`, `model_type: qwen3`, 6 layers all
`sliding_attention` window 2048, `is_causal: false`, head_dim 128, 64 query
heads / 8 KV heads, hidden 6144, MLP 12288, `dflash_config`: block_size 8
(7 draft tokens + anchor), conv_kernel_size 2, conv_group_size 16,
selector_top_k 16, selector_rank 256, mask_token_id 154856,
`target_layer_ids [5, 19, 33, 47, 61, 75]`, `num_target_layers` 78, no
`target_hidden_size`, no quantization (BF16, 96 tensors, ~4.9 GB; borrows
tokenizer, embed_tokens and lm_head from the target). License CC BY-NC-ND 4.0.

Card claims: GSM8K 5.94 / MATH-500 6.02 / HumanEval 5.48 / MBPP 4.95 /
MT-Bench 4.19 acceptance, against GLM-5.3's own MTP at 5.12/5.05/4.85/4.34/
3.81; protocol SGLang, 4x GB300 TP4, FA4 draft attention, GLM-recommended
sampling (T=1.0, top-p 0.95), 4096 max tokens, 128 samples at concurrency 1.
Our MTP control reproduced the card's MTP figure to 3% (4.967 vs 5.12), so
sampling, template, EOS, target and tiered stack are validated as a harness;
the residual is draft-side. Upstream card requirement: vLLM v0.28.0+ and the
**V2 model runner** - a DFlash2 checkpoint reaching a V1 proposer silently
drafts as DFlash1. Our arms set `VLLM_USE_V2_MODEL_RUNNER=1`; a standing
assertion in the launch scripts would keep it that way.

## Where the 32% gap stands, ranked

1. **RoPE layout divergence** (Finding B) - proven stack-level divergence, one
   cheap decisive experiment.
2. **Aux-tap semantics on the tiered target** - *which* tensor
   `GlmMoeDsa`/`deepseek_v2` exposes at layers [5, 19, 33, 47, 61, 75] as aux
   states (the #50487-shaped question, untested for our stack). The +1
   convention is validated by DFlash1's position-0 = 79% through the same
   path, but pre-norm vs post-residual has never been diffed against what
   training consumed.
3. **#51538 null-block pollution** (Finding C) - long-context harnesses only;
   cannot touch GSM8K.
4. Genuine checkpoint behavior on NVFP4 + our routing vs the card's BF16 on
   GB300 - the matched-MTP control argues against this being large, but it is
   not excluded.

Ruled out: everything in Finding A's table, plus the Gumbel-salt coupling
(#54282) under greedy drafting.

## Plan

**P1 - RoPE arm (decides the gap question).** Port the #51655 threading into
`qwen3_dflash.py` and `dflash/utils.py` (`is_neox_style` param on
`DFlashQwen3Attention`/`DFlashQwen3DecoderLayer`, `dflash_target_rope_added`
probe with #54373's attention-first reading). Run the Phase 43 card-protocol
replication unchanged otherwise:

```text
arm: experiments/2026-08-28-nvfp4-tiered/replicate_dflash2_eval.py
     on GLM-5.3-NVFP4, DFlash2, width 7, GSM8K, T=1.0, top-p 0.95
gate: acceptance vs 4.029 (pre-fix baseline) and 5.94 (card)
```

Both outcomes are publishable: a jump closes the 32% question; a collapse
proves NeoX was correct for this checkpoint and permanently retires the
hypothesis. One node, one job, no new harness code.

**P2 - hash fix (#53292).** Add `num_speculative_tokens` to
`SpeculativeConfig.compute_hash()` for dflash/dspark before the next
width-swept batch. Four lines plus a unit test.

**P3 - null-block guard (#51538 hunk).** Port `ctx_resident`/`q_resident`
and `PAD_SLOT_ID` masking (and, for c4 later, `cp_local_slot` slot math) into
the local ctx-KV insert kernel before any long-context DFlash2 acceptance is
quoted again. Unit-test by pointing evicted-window rows at block 0 and
asserting PAD_SLOT_ID.

**P4 - #50457 production evaluation.** Verify the symptom first (turn-2
draft-prefill re-reads full context) on the production c4 host; only then
port the booking change against the DCP-modified coordinator. Acceptance is
expected not to move; the metric is draft-prefill token count on turn 2.

Do not port: #54282/#53017 (until probabilistic drafting is evaluated, keep
the branch fork-clean like upstream), the #50910 dtype change (deliberately
reverted for MTP3 fp32 bit-identity, documented at
`speculate/speculator.py:280-283`), #51718's layout refactor (collides with
DCP/tiered; never piecemeal).

## Sources

- Upstream commits: `6adad08767` (#51655 rope), `b389ac2946` (#52816 DFlash2),
  `a9a17e7095` (#53435), `97388c44f9` (#51538), `fe755c8899` (#54282 salt),
  `d4f4d3f40f` (#53017), `81bc196913` (#50910), `dcf5ceb2b9`/`c8a723e0c9`
  (local cherry-picks of #53336, #51256, #44492).
- Local commits: `2a26f151ac` (backport), `985cc85109`, `8d94773e01`
  (#48776), `4349240546` (tiered plan), Phase 42/43 reports
  (`experiments/2026-08-28-dflash2-backport/`,
  `experiments/2026-08-28-nvfp4-tiered/`).
- Card and config: `huggingface.co/incoai/GLM-5.3-DFlash2`, local
  `../models/GLM-5.3-DFlash2/config.json`,
  `../models/GLM-5.3-NVFP4/config.json`.
- This audit's transcript: gh census of 40 merged + 30 open dflash-matching
  PRs, 2026-08-30; direct git diffs `HEAD..origin/main` over the DFlash
  pathspec.

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

---

# Results (2026-08-30, same day)

## P1: the RoPE hypothesis is refuted

Job 1537299, both arms on one node (jpbo-002-16), Phase 43's harness unchanged:
GLM-5.3-NVFP4, DFlash2 width 7, GSM8K, T=1.0, top-p 0.95, 64 samples.

| arm | acceptance (mean) | median | pooled |
| --- | ---: | ---: | ---: |
| `rope-neox` — this fork's behaviour | **3.9381** | 3.9271 | 3.8652 |
| `rope-interleaved` — copied from the target | **1.9174** | 1.9126 | 1.9029 |
| Phase 43 post-fix (reproduction gate) | 4.0293 | | |
| our MTP7 control, same width | 4.9121 | | |
| card, SGLang/GB300 | 5.94 | | |

The NeoX arm reproduces Phase 43 to within 2.3%, so the pair is valid rather
than two unrelated runs. **Copying the target's layout halves acceptance.**

The audit was right that this fork diverges from upstream #51655, and wrong
about the direction: the draft was trained NeoX, as its own config says
(`model_type: qwen3`, `rope_type: default`), even though every rotary in the
DeepseekV2-derived target is interleaved. Upstream's copy is correct for a
draft trained against its target's layout and wrong for this checkpoint.

This was the audit's own stated decisive outcome — *"if it collapses, NeoX was
right and the gap is elsewhere"* — so hypothesis 1 of 4 is retired with
evidence. **The 32% gap is elsewhere.** The next candidate is unchanged:
aux-tap semantics at layers [5, 19, 33, 47, 61, 75], pre-norm versus
post-residual, never diffed against what training consumed.

Committed as `be382fe44d` (the port) then `21644f8b9c` (NeoX restored as the
default, upstream's behaviour kept behind `VLLM_DFLASH_DRAFT_ROPE=target`).
Keeping the plumbing is deliberate: the question is real for other targets, and
the measurement is recorded beside the code so upstream's copy is not
re-adopted on parity grounds alone.

## P2: shipped

`num_speculative_tokens` added to `SpeculativeConfig.compute_hash()` for
dflash/dspark (`8a0f60ddd8`). The test is verified to fail without the fix:
widths 7 and 8 both hashed to `c3d9d001b44d…` before it.

## Two harness defects found on the way, both caught before they produced a number

1. **`RESULT_DIR` also redirected the harness.** `arm-replicate.sh` resolved
   `replicate_dflash2_eval.py` relative to the results directory, so redirecting
   output moved the script. Split into `script_dir` and `result_dir`.
2. **A dying server answers `/health`.** Two arms in one allocation share port
   8027, and the second arm declared readiness against the first arm's
   shutting-down server — its own server log was still zero bytes. With defect 1
   fixed and this one left, the interleaved arm would have been measured against
   the NeoX server and returned a plausible number for a run that never
   happened. The arm now waits for the port to go quiet before trusting
   `/health`.

Both were caught only because `load_dflash_model` logs the layout it chose and
its source, which made "this arm did not run my code" visible instead of
inferable. Any future arm that selects behaviour by environment should log the
selection the same way.

---

# Fixes implemented (2026-08-30)

| # | Defect | Commit | Effect |
| --- | --- | --- | --- |
| P2 | `num_speculative_tokens` missing from the DFlash/DSpark config hash | `8a0f60ddd8` | a width sweep on a warm shared cache could replay a narrower run's plan |
| — | Draft RoPE layout ported, then reverted to NeoX by measurement | `be382fe44d`, `21644f8b9c` | upstream's target-copy **halves** acceptance here; now opt-in |
| #51538 | Draft KV written into the null block | `6fea8f2850` | correctness; not the acceptance gap |
| #54282 | Draft and verifier shared one Gumbel stream | `7c1b93ebf5` | prerequisite for probabilistic drafting |
| #52188 | DFlash slot math is CP1-only | `45e14de4b6` | now fails closed at `cp_size > 1` |
| — | `seq_lens` unclamped, `sample_from_anchor` unguarded, aux `layer_ids` fallback missing, `hc_mult` ungated | `45e14de4b6` | latent acceptance losses that raise nothing |

## What the three benchmarks showed

Acceptance length, card protocol, guard off (i.e. the state before today's work):

| benchmark | measured | card | gap |
| --- | ---: | ---: | ---: |
| HumanEval | **4.9510** | 5.48 | **−10%** |
| GSM8K | 4.0226 | 5.94 | −32% |
| 16K code | **2.8395** | — | — |

**The gap is not uniform, and that is the most important result here.** A
single-benchmark reading -- "DFlash2 is 32% down" -- is wrong. DFlash2 is close
to the card on code, far off on math, and collapses on long context. Any fix
evaluated on GSM8K alone would have been measuring the least representative
shape of the three.

The null-block guard moved GSM8K 4.0226 -> 4.0033, i.e. not at all, which is
what its own mechanism predicts: short answers never evict. It is kept as a
correctness fix, not an acceptance one.

`guardoff-gsm8k` at 4.0226 against Phase 43's 4.0293 (0.17% apart) is a tight
reproduction, so these numbers are comparable to the historical record.

## The open hypothesis, unmeasured

`draft_sample_method` defaults to `"greedy"`
(`vllm/config/speculative.py:282`), so `SAMPLE_PROBABILISTIC` is False
(`dflash2/speculator.py:156`), the candidate selector takes an **argmax walk**,
and the 16-candidate lattice's proposal distribution `q` never reaches the
verifier: the ratio test degenerates from `sum(min(p, q))` to `p(argmax q)`.
SGLang -- the stack that produces 5.94 -- samples the path per request and hands
`q` to verification.

This would hit DFlash2 and not MTP, because MTP's next-token head is near-peaked
so argmax approximates sampling, while DFlash2's lattice is not and ships an
fp32 logits cache whose only consumer is the ratio test.

**Not changed, deliberately.** Upstream vLLM also defaults to greedy, so this is
a divergence from SGLang rather than a fork defect, and flipping a shipped
default on an unmeasured hypothesis is the exact mistake made this morning with
the RoPE port. The lever exists (`DRAFT_SAMPLE_METHOD=probabilistic` in
`arm-replicate.sh`) and its prerequisite has landed; the arm was cancelled
before running. Run it across all three benchmarks before touching the default.

---

# Validation (2026-08-30, jobs 1541340-42): all fixes together, both sampling modes

Every arm verified from its own log: rope neox, null-block guard 1, and the
stated draft_sample_method. Six arms, two per benchmark on one allocation.

| benchmark | pre-fix | fixed + greedy | fixed + probabilistic | card |
| --- | ---: | ---: | ---: | ---: |
| GSM8K | 4.0226 | 3.9951 | **4.0736** | 5.94 |
| HumanEval | 4.9510 | 4.8780 | **5.1276** | 5.48 |
| 16K code | 2.8395 | 2.8455 | **3.1564** | — |

Three conclusions, each cross-checked across benchmarks:

1. **The six fixes do not move acceptance, and were not expected to.** Greedy
   arms land within 0.7% of pre-fix everywhere. They are correctness and
   landmine fixes; the implementation is now clean against upstream, which is
   what the audit was for.

2. **Probabilistic drafting helps on every benchmark: +2.0%, +5.1%, +11.0%.**
   The gain *grows with context length*, which is mechanistically sensible --
   the longer the sequence, the more the argmax walk's early mistakes compound
   through the lattice. This is a real effect, not noise: all three pairs move
   the same direction.

3. **And it is still nowhere near the card.** Even at its best (HumanEval,
   5.13) DFlash2 only matches our own MTP7 control (4.91) plus a little; on
   GSM8K it remains 31% under 5.94. **Sampling was a real but minor effect. The
   dominant residual is not in our implementation.**

## Where that leaves the goal

The implementation has been audited twice -- once by name, once by subsystem
against both upstream vLLM and SGLang -- and every divergence found is either
fixed or measured-and-refuted (RoPE, aux tap, null block, sampling). The
remaining gap to the card is most plausibly one of:

- **The card's stack itself**: SGLang on 4x GB300 with FA4 draft attention.
  GB300 is Blackwell-Ultra; this is GH200. The card's MTP baseline (5.12) also
  exceeds what any MTP measurement here has produced on 5.3, which suggests a
  hardware/stack difference in the *baseline*, not just the drafter.
- **The checkpoint's training protocol**: acceptance is a property of the
  drafter against the exact hidden states it was distilled from. Our aux tap
  matches SGLang's code, but "what SGLang's code does" and "what training
  consumed" are only the same thing if SGLang's serving stack matches the
  training-time capture, which is unverifiable from here.

**DFlash2 is now working as well as this implementation can make it work**, and
the honest verdict on the goal's premise is: on this hardware, at this
protocol, DFlash2 does not beat MTP3 on acceptance (4.07 vs 4.91 at matched
width), and the card's 5.94 is not reachable by fixing fork defects, because
there are none left that we can find.

The one comparison that would still be decisive and has never been run:
**matched MTP3-vs-DFlash2 end-to-end throughput on a 5.3 target**, since
acceptance is not throughput and DFlash2's cheaper draft could still win on
tok/s. That is a benchmark run, not an implementation question.

---

# Per-position decomposition (2026-08-30, jobs 1542631-33): the fault is at position 0

The user's challenge -- acceptance is hardware-invariant, so if MTP reproduces
the card and DFlash2 does not, the fault is DFlash2-specific -- was accepted,
and the hardware explanation retracted. The per-position survival curve is the
diagnostic that localises a draft-accuracy fault, so `replicate_dflash2_eval.py`
was extended to capture `vllm:spec_decode_num_accepted_tokens_per_pos` and three
arms were run: DFlash2 greedy, DFlash2 probabilistic, and MTP7 as the control at
the same width.

**Measurement caveat**: the JSON capture of the per-pos counter proved
unreliable (counts ~55x too small -- the summed per-pos series does not match
the scalar accepted counter). The server's own 10-second
`Per-position acceptance rate` log lines are the trustworthy source; the numbers
below are the mean of the last six steady-state windows of each run.

| position | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MTP7 | **0.915** | 0.784 | 0.656 | 0.520 | 0.422 | 0.314 | 0.241 |
| DFlash2 greedy | **0.615** | 0.552 | 0.491 | 0.436 | 0.381 | 0.332 | 0.288 |
| DFlash2 prob | **0.619** | 0.547 | 0.486 | 0.413 | 0.355 | 0.314 | 0.257 |

Reading:

- **Position 0 is the fault line.** It is the identical next-token task for
  both drafters -- predict the token after the just-committed bonus, same
  conditioning. MTP lands 0.915, DFlash2 0.615. That is the "draft inputs are
  wrong" signature, not the lattice-walk signature.
- **Greedy ~= probabilistic at every position**, so the sampling path is not
  the differentiator.
- **Away from position 0 the DFlash2 curve is a clean geometric decay**
  (ratio ~0.87) -- no structural break. The flatness is why the headline
  number looked "plausible" while the first prediction was 30% under control.

While the arms ran, the SGLang `dflash_worker_v2.py` / `dflash.py` /
`dflash_utils.py` input-construction diff was completed (the subsystem audit
had compared math but not runtime input construction). Cleared, line by line:
`_score_edges` lattice math, grouped conv, context-KV pipeline (fc ->
hidden_norm -> fused kv_proj -> k_norm -> RoPE, same order both stacks), query
block (bonus at offset 0, masks after, positions prefix+off), aux taps (server
log confirms `(6, 20, 34, 48, 62, 76)` = after layers 5/19/33/47/61/75 =
SGLang's documented +1 convention), `noise_embed_scale` (retired: no SGLang
model defines the hook, defaults 1.0), checkpoint weight shapes (all 96 keys),
and the non-causal SWA window (FA symmetrises `(2047, 0) -> (2047, 2047)`).

## Unary-walk A/B: separating lattice from hidden states

Position 0's score is `unary + pairwise(anchor, h)`. Two distinct suspects
remain: the **unary/candidates** (draft hidden -> lm_head top-16) are weak, or
the **lattice walk** drags the choice off the unary argmax. These separate with
one diagnostic: force the walk to take the top-1 candidate at every position
(candidates are sorted), ignoring the codebook scores.

- unary ~= lattice -> the draft's hidden states are the bottleneck (input problem);
- unary >> lattice -> the trained lattice is hurting us in this run (selector problem).

Implemented as `VLLM_DFLASH2_SELECTOR_WALK=lattice|unary` (envs.py + a
`UNARY_WALK` constexpr in the walk kernel), submitted as `submit-unary-walk.sh`:
unary on GSM8K and HumanEval plus a same-build lattice control on GSM8K.

## Unary-walk result (2026-08-30, jobs 1543475-77): the lattice is NOT the fault

| arm | acceptance (mean) | per-position (steady-state mean) |
| --- | ---: | --- |
| GSM8K lattice control | 3.9880 | 0.618 0.538 0.476 0.430 0.377 0.335 0.292 |
| GSM8K unary walk | 3.8732 | 0.618 0.545 0.470 0.397 0.348 0.305 0.265 |
| HumanEval lattice (validation run) | 4.8780 | 0.877 0.755 0.665 0.580 0.508 0.445 0.405 |
| HumanEval unary walk | 4.7382 | 0.853 0.704 0.596 0.506 0.428 0.374 0.325 |

Reading: **ignoring the trained lattice does not help -- it hurts slightly**
(-2.9% GSM8K, -2.9% HumanEval). The codebook scores are doing their job
(path coherence is worth a few percent), and position 0 is *identical*
between the two walks (0.618 both ways on GSM8K), which is expected: at
position 0 the lattice can only choose among the same 16 candidates and the
walk's first pick is dominated by the unary term.

So the bottleneck is upstream of the lattice: **the draft's hidden states
themselves, or the candidate/unary computation over them**. Position 0 on
HumanEval is 0.85-0.88 -- nearly MTP-class -- while GSM8K sits at 0.62, so
the drafter's first-prediction quality is also task-dependent, which points
at the draft forward (context KV, attention, conv) rather than a constant
plumbing error: a wrong aux tap or a wrong mask embedding would depress
position 0 uniformly across tasks, not just on GSM8K.

Next diagnostic in the queue: candidate recall -- is the target's sampled
token inside the 16 candidates at each position? Recall bounds everything:
if recall is ~0.62 on GSM8K, the draft's hidden states are placing the true
token outside the top-16, and the fault is in the draft forward or the
unary head application; if recall is high (~0.95) while acceptance is 0.62,
the fault is in how the target's distribution compares to the proposal
(i.e. the verify side), not the draft at all.

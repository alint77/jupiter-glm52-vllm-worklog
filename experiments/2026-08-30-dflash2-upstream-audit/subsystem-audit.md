# DFlash2 subsystem audit: behavioural dependencies, not names

Follow-up to `README.md` (Phase 48). That audit and its predecessor searched by
string ("dflash"/"DFlash"/"DSpark") in code, commit titles and PR titles, and
retired the RoPE hypothesis by measurement. This pass deliberately ignores names
and audits the *subsystems DFlash2 depends on*, fork vs `origin/main`
(`680e2177e4`), and — new here — **fork vs SGLang**, the stack that actually
produced the card's 5.94.

Tree: `dflash2-backport` at `6fea8f2850`. Upstream: `origin/main` `680e2177e4`.
Measured configuration under audit: GLM-5.3 (`GlmMoeDsaForCausalLM`, deepseek_v2)
target, `incoai/GLM-5.3-DFlash2` draft, V2 model runner, width 7, TP4/EP4, DCP1,
prefix caching on, `draft_sample_method` at its default `greedy`, GSM8K at
T=1.0 / top-p 0.95. Measured 3.94; card 5.94; matched MTP7 control 4.91 vs card
5.12.

New material used in this pass: the SGLang reference implementation, read
directly from `sgl-project/sglang` via `gh api`
(`python/sglang/srt/models/dflash.py`,
`.../speculative/dflash_worker_v2.py`, `.../dflash_utils.py`,
`.../model_executor/model_runner_components/spec_aux_hidden_state.py`,
`.../models/deepseek_v2.py`, `.../layers/communicator.py`,
`.../layers/attention/flashattention_backend.py`).

**Headline.** No fork-vs-upstream divergence is live under the protocol that was
measured. Everything that differs from `origin/main` in the DFlash2 blast radius
is either (a) gated behind `temperature != 0` on the *draft* side and therefore
inert while `draft_sample_method="greedy"`, or (b) gated behind `cp_size > 1`
and therefore inert at DCP1. The strongest remaining explanation for the 32% gap
is a **vLLM-vs-SGLang** divergence that is not a bug in any single file: vLLM
throws away DFlash2's proposal distribution by default, and SGLang does not.

---

## 1. The draft's proposal distribution `q` never reaches the verifier

**Rank: 1. Live at the measured protocol. This is the candidate to test first.**

### (a) What SGLang does

DFlash2's whole point is a K=16 candidate lattice with a *usable* proposal
distribution over those 16 candidates. SGLang samples a path from it and hands
`q` to the verifier:

- `python/sglang/srt/models/dflash.py`, `CandidateSelector.sample_path` (~L1004):
  returns `(tokens, q_rows)`. `greedy_mask` is resolved *per request* from
  `sampling_info` (`resolve_greedy_mask`), so a T=1.0 request takes the
  softmax-sampled branch, not the argmax branch. For greedy rows it returns a
  one-hot `q`, "selected rather than branched, so one captured graph serves
  greedy and sampling batches alike".
- `python/sglang/srt/speculative/dflash_worker_v2.py` L261-270 and L992-1031:
  `tokens, q_rows = self.selector.sample_path(...)`, then
  `draft_probs.scatter_(-1, candidate_ids, q_rows.float())` into a zero-filled
  `[bs, gamma, vocab]` buffer, which is passed into verification.

So on the card's protocol (T=1.0, top-p 0.95) SGLang runs
`accept ~ min(1, p/q)` with a real 16-support `q`, and resamples rejections from
the residual `(p-q)+`.

### (b) What this fork does

- `vllm/config/speculative.py:282` — `draft_sample_method: DraftSampleMethod = "greedy"`
  (identical upstream, `origin/main:vllm/config/speculative.py:578`).
- `vllm/v1/worker/gpu/spec_decode/speculator.py:130-134` — `self.draft_logits`
  is allocated **only** when the method is not greedy; otherwise it stays `None`.
- `vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py:156` —
  `SAMPLE_PROBABILISTIC = self.draft_logits is not None` → `False`. The walk
  kernel then passes `temperature if SAMPLE_PROBABILISTIC else 0.0` into
  `gumbel_noised_argmax` (`dflash2/speculator.py:62`), i.e. a pure argmax path.
- `dflash2/speculator.py:214` — `_cache_draft_logits` is skipped entirely.
- `vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py:585-625` — with
  `HAS_DRAFT_LOGITS=False` the test degenerates to
  `target_logprob > log(u)`, i.e. `q` is treated as a point mass at the drafted
  token.

### (c) Why this depresses acceptance *specifically*

Per position, expected acceptance is `p(argmax q)` under the fork and
`Σ_x min(p(x), q(x))` under SGLang. These coincide only when `q` is nearly a
point mass. MTP's next-token head is very peaked, which is why the MTP7 control
lands within 4% of its card figure under the same greedy default. DFlash2's
selector is explicitly *not* peaked — it is a lattice over 16 candidates per
slot whose scores are pairwise transition logits, and it ships a dedicated
fp32/`-inf` logits cache (`dflash2/speculator.py:131-135`) whose only consumer
is the probability-ratio test. Discarding it costs the most at exactly the
positions DFlash2 is designed to win: the deep mask tokens, where the draft is
least certain. The measured per-position decay (0.714 geometric, Phase 43) is
the signature of a proposal that is being all-or-nothing tested rather than
ratio-tested.

Honest counter-argument: the MTP control's small gap shows greedy drafting is
not a large penalty *for MTP*. That does not transfer — see above. This is a
hypothesis, not a proven divergence in a single line; but it is a proven
difference in what the two stacks compute.

### (d) Cheap decisive test

One arm, one node, no new harness code: rerun
`experiments/2026-08-28-nvfp4-tiered/replicate_dflash2_eval.py` unchanged except
for `draft_sample_method: "probabilistic"` in the speculative config
(`DraftSampleMethod = Literal["greedy", "probabilistic"]`,
`vllm/config/speculative.py:77`). Gate against 3.938 (the NeoX arm) and 5.94.

**Port finding 2 below before running this arm, or the result is not
interpretable.**

Falsifier: if probabilistic drafting does not move acceptance, `q` is not the
gap and the lattice is behaving like a point mass — which would itself be a
finding worth chasing (it would point at the selector codebooks or
`output_multiplier`/softcap handling).

---

## 2. Draft and target share one Gumbel noise stream (upstream #54282)

**Rank: 2. Inert today; becomes a correctness bug the moment finding 1 is acted on.**

- **Upstream** (`fe755c8899`, #54282): `gumbel.py` defines
  `_DRAFT_NOISE_SALT = 1 << 30` and `gumbel_noised_argmax(..., IS_DRAFTING)`
  offsets `pos` by the salt when drafting
  (`origin/main:vllm/v1/worker/gpu/sample/gumbel.py:16-20, 118-119`). The DFlash2
  walk passes `IS_DRAFTING=True`
  (`origin/main:vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py:64`).
  Rationale in the upstream comment: "Verification is a probability-ratio test,
  not a Gumbel coupling, so a proposal and the residual it is resampled from
  must not share a noise vector."
- **This fork**: no salt, no `IS_DRAFTING`
  (`vllm/v1/worker/gpu/sample/gumbel.py:92-130`;
  `vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py:55-64`). The draft's
  draw is `tl.randint(seed, sample_pos - 1)` and the verifier's uniform is
  `tl_rand32(seed, pos)` at the same `(seed, pos)`
  (`rejection_sampler_utils.py:534`).
- **Why it matters for acceptance**: under probabilistic drafting the accept/
  reject uniform is a deterministic function of the same stream that chose the
  proposal, so accept decisions are correlated with the proposal — the measured
  acceptance is then neither the greedy number nor the correct probabilistic
  number, and the output distribution is biased.
- **Cheap decisive test**: unit-level. `tests/v1/spec_decode/test_rejection_sampler_utils.py`
  in `fe755c8899` already contains the coverage; cherry-pick the test alone and
  run it against HEAD — it should fail.

Bundle `d4f4d3f40f` (#53017, draft-logits-cache column stride, plus its
`logits_cache.size(-1) >= vocab_size` assert) with this port. #53017 is *not*
reachable from DFlash2 — `_cache_draft_logits`
(`dflash2/speculator.py:161-177`) computes its own strides correctly — but the
shared `gumbel_sample` path the fork still carries has the
`col * vocab_size` form (`vllm/v1/worker/gpu/sample/gumbel.py:174-180`) rather
than `col * logits_cache.stride(1)`, and the missing assert is the only thing
that would catch a padded-vocab cache silently truncating.

---

## 3. Draft `seq_lens` is not clamped to `max_model_len`

- **Upstream**: `tl.store(out_seq_lens_ptr + req_idx, tl.minimum(last_valid_pos + 1 + num_query_per_req, max_model_len))`
  (`origin/main:vllm/v1/worker/gpu/spec_decode/dflash/speculator.py`, the
  `_prepare_dflash_inputs_kernel` tail).
- **Fork**: `vllm/v1/worker/gpu/spec_decode/dflash/speculator.py:602` —
  `tl.store(out_seq_lens_ptr + req_idx, last_valid_pos + 1 + num_query_per_req)`,
  unclamped. `max_model_len` is already a kernel argument; only the clamp is
  missing. The query *positions* are clamped one line earlier
  (`speculator.py:580-581`), so the two disagree at the boundary.
- **Why it would depress acceptance**: only within `num_query_per_req` tokens of
  `max_model_len`. Attention reads `seq_lens` past the block table's last valid
  entry. Cannot touch GSM8K.
- **Cheap decisive test**: drive one request to `max_model_len - 3` and assert
  `input_buffers.seq_lens[0] <= max_model_len`.

---

## 4. DCP-local slot math and stale DCP sequence lengths (c4 only)

Three separate upstream hunks the fork lacks, all gated on `cp_size > 1`:

| Upstream | Fork |
| --- | --- |
| `cp_local_slot(...)` and `pos // (block_size * CP_SIZE)` in the DFlash ctx/query slot math (`origin/main` `dflash/speculator.py`, from #52188 `d1e3eee6fb`) | plain `pos // block_size`, `block_id * block_size + pos % block_size` (`vllm/v1/worker/gpu/spec_decode/dflash/speculator.py:538-552, 566-577`) |
| `prepare_dcp_local_seq_lens` re-run inside `_build_draft_attn_metadata` because "draft steps advance and rewind their own global sequence lengths" (`origin/main` `spec_decode/speculator.py:296-306`) | the fork computes it unconditionally but from the *target's* buffers before the draft rewinds (`vllm/v1/worker/gpu/spec_decode/speculator.py:224-234`) |
| `dcp_local_seq_lens` threaded through `_prepare_dflash_inputs_to_capture` (#52188, `origin/main` `dflash/cudagraph.py:45-64`) | absent (`vllm/v1/worker/gpu/spec_decode/dflash/cudagraph.py`) |

Also separate: `_compute_slot_mappings_kernel` in
`vllm/v1/worker/gpu/block_table.py:270-330` collapses the KV-manager block size
and the kernel block size into one tensor (`block_sizes_tensor` is populated
from `kernel_block_sizes`, `block_table.py:97`). Upstream #51031
(`0ecc284790`) separates them. **At `CP_SIZE == 1` the two are algebraically
identical** — I checked this line by line — so this is a c4/DCP4 correctness
item only, not a DCP1 acceptance item.

Cheap decisive test: none needed at DCP1. Before quoting any c4 DFlash2 number,
assert `block_tables.cp_size == 1` in the launch script or port all four hunks.

---

## 5. `sample_from_anchor` is silently accepted rather than rejected

- **Upstream** raises `ValueError` if `dflash_config.sample_from_anchor` is true
  (`origin/main:vllm/v1/worker/gpu/spec_decode/dflash/speculator.py:63-71`).
- **Fork** hard-codes `self.sample_from_anchor = False`
  (`vllm/v1/worker/gpu/spec_decode/dflash/speculator.py:62`) with no guard.
- Inert for this checkpoint (`config.json` has no `sample_from_anchor`), but a
  future checkpoint that sets it would silently drop a prediction slot — i.e. a
  pure acceptance loss with no error. Two lines; port with anything else.

---

## 6. Aux-layer id resolution has no `dflash_config.layer_ids` fallback

- **Upstream** `get_eagle3_aux_layers_from_config` falls back to
  `dflash_config["layer_ids"]` / `eagle_config["layer_ids"]`
  (`origin/main:vllm/v1/worker/gpu/spec_decode/eagle/eagle3_utils.py:56-62`).
- **Fork** stops at `target_layer_ids` / `dspark_target_layer_ids`
  (`vllm/v1/worker/gpu/spec_decode/eagle/eagle3_utils.py:34-58`).
- **Inert here**: this checkpoint carries `dflash_config.target_layer_ids`, which
  both resolve. A checkpoint using the `layer_ids` spelling would fall through to
  `get_eagle3_default_aux_hidden_state_layers()` and tap the *wrong six layers*
  with no warning — a silent 2-token acceptance loss of exactly the shape being
  chased here. Worth porting purely as a landmine removal.

---

## 7. `hc_mult` widening is ungated

- **Upstream** widens the drafter's hidden buffer by `hc_mult` only when the
  target implements `get_mtp_target_hidden_states()`
  (`origin/main:vllm/v1/worker/gpu/spec_decode/speculator.py:34-46, 107-116`).
- **Fork** widens unconditionally from the *draft's* `hf_config.hc_mult`
  (`vllm/v1/worker/gpu/spec_decode/speculator.py:105-110`).
- Inert: this draft config has no `hc_mult` → 1. Latent for any future target.

---

# Subsystems audited and found clean

Explicit negative results, so none of this is re-audited.

## Cross-checked against SGLang (the 5.94 stack), not just upstream vLLM

**Aux hidden-state tap point — CLEAN. This retires the README's ranked-#2
hypothesis ("pre-norm vs post-residual, never diffed against training").**

- SGLang resolves `target_layer_ids` verbatim for a non-`muse_glimmer` target
  (`spec_aux_hidden_state.py::_map_muse_target_layer_ids` applies `+1` *only*
  for `model_type == "muse_glimmer"`), then `set_dflash_layers_to_capture` adds
  `+1` (`sglang .../models/deepseek_v2.py:3235`), and the capture site is
  `prepare_attn_and_capture_last_layer_outputs`
  (`sglang .../layers/communicator.py:509-539`), which appends the `residual`
  returned by `prepare_attn` — i.e. **the residual stream at the *input* of
  layer `id+1`, which is the output of layer `id`**, for
  `id ∈ {5,19,33,47,61,75}`.
- vLLM resolves `[i+1 for i in target_layer_ids]` = `{6,20,34,48,62,76}`
  (`vllm/v1/worker/gpu/spec_decode/eagle/eagle3_utils.py:44-46`) and captures
  `hidden_states + residual` **before** calling `layer(idx)`
  (`vllm/model_executor/models/deepseek_v2.py:1490-1497`, note the capture is
  *above* the `hidden_states, residual = layer(...)` call on line 1498) — i.e.
  the residual stream at the input of layer 6.
- Same tensor, same six boundaries. I initially read the vLLM capture as
  post-layer; it is pre-layer, and the two `+1`s cancel exactly as they do in
  SGLang. `deepseek_v2.py`'s aux block is byte-identical to upstream apart from
  a `residual.contiguous()` the fork adds.

**Non-causal sliding-window mask — CLEAN.**
SGLang: `causal = attn_type not in (ENCODER_ONLY, DECODER_BIDIRECTIONAL)`, then
`window_size = (sw, 0 if causal else sw)`
(`sglang .../attention/flashattention_backend.py:1331-1336`), and
`_get_dflash_attention_type` maps the config's `"is_causal": false` to
`ENCODER_ONLY` (`sglang .../models/dflash.py:104-111`). So SGLang runs
`(2047, 2047)`. vLLM reaches the same value by a different route:
`FlashAttentionImpl.__init__` stores `(2047, 0)`
(`vllm/v1/attention/backends/flash_attn.py:758-764` (`self.sliding_window = (sliding_window - 1, 0)` at :764)) and
`_maybe_symmetrize_window(window, causal=False)` widens it to `(2047, 2047)`
(`flash_attn.py:311-322`). Windows agree.

**Selector lattice and path walk — CLEAN.**
`_score_edges` is algebraically identical
(`vllm/model_executor/models/qwen3_dflash2.py:164-185` vs
`sglang .../models/dflash.py:910-931`), including the anchor expansion into slot
0's predecessor axis. The fork's Triton walk (`previous = 0`;
`score_base = (flat*top_k + previous)*top_k`;
`vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py:38-73`) reproduces
SGLang's `initial_indices = scores[:,0,0].argmax` +
`_follow_maps(scores[:,1:].argmax(-1), ...)` exactly.

**Decoder-layer structure, conv placement, projection order — CLEAN.**
`DFlash2Qwen3DecoderLayer.forward` (`qwen3_dflash2.py:141-161`) matches
`DFlashDecoderLayer.forward` (`sglang .../models/dflash.py:502-543`) hunk for
hunk: fused add-norm → `attention_conv.prepare` → attn →
`attention_conv.finish` → post-attn norm → `mlp_conv.prepare` → MLP →
`mlp_conv.finish`. Projection order is `hidden_norm(fc(concat))` on both sides
(SGLang `DFlashDraftModel.project_target_hidden`, `dflash.py:663-681`; the
per-tap `aux_hidden_norms` variant at `dflash.py:893-906` belongs to
`DFlashLagunaForCausalLM`, not DFlash2). Query tokens are embedded from
`input_ids` only and the projected target hidden feeds the context K/V — same
design in both.

## Fork vs upstream

**Rejection sampler / acceptance logic — CLEAN under greedy.**
`vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py:515-628`. Upstream's
refactor (`accepted`/`verifying` split, explicit `-1` placeholder handling) is
semantically identical in the greedy branch: `accepted &= target_argmax == draft_sampled`,
and `-1` can never equal an argmax so placeholders reject either way. Bonus-token
handling, `accepted_length` accumulation and the resample-index math all match.

**`sample_pos` / `sample_indices` semantics — CLEAN.**
The fork passes `self.sample_pos - 2` and `sample_draft` adds `+1`
(`dflash/speculator.py:265` and `speculator.py:357-368`); upstream passes
`- 1` and `sample_draft` adds nothing. Net-identical. Not a divergence, despite
looking like one in the diff.

**FlashAttention group geometry (#53002 / #53336) — ALREADY PORTED.**
`vllm/v1/attention/backends/flash_attn.py:373-378` uses
`get_num_attention_heads_from_layers(...)`, `kv_cache_spec.num_kv_heads`,
`kv_cache_spec.head_size`. `dflash/speculator.py:92-99` has the
`copy.copy(super().attn_vllm_config)` form. Both hunks present.

**Sliding-window source in `FlashAttentionImpl.forward` — DIFFERENT, EQUIVALENT
HERE.** Fork reads `attn_metadata.sliding_window`, derived from
`kv_cache_spec.sliding_window` (`flash_attn.py:658-662, 938-944`); upstream reads
the layer's own `self.sliding_window` (`origin/main` `flash_attn.py`, from the
"layer's own window wins over the group's" hunk). For an all-SWA draft with one
window both resolve to `(2047, 2047)`. Upstream's form is correct for mixed
windowed/global layers in one group; port it if a mixed DFlash draft is ever
served.

**Rejected-context-row KV writes — DIFFERENT, PROVEN BENIGN.**
Upstream masks context loads to `num_valid_ctx` and stores `PAD_SLOT_ID` for the
rejected suffix; the fork stores real positions and real slots for the full
`num_ctx` (`dflash/speculator.py:524-581`). The rejected positions are exactly
`last_valid_pos+1 .. last_valid_pos+num_rejected`, and the query rows write
`last_valid_pos+1 .. last_valid_pos+num_query_per_req` at the *same* block-table
lookups; `num_rejected <= num_speculative_tokens < num_query_per_req`, and
`precompute_and_store_context_kv` (`dflash/speculator.py:409-413`) runs strictly
before the query forward. Every stale write is overwritten in the same step.
The `is_query = (j >= num_ctx)` vs `(j >= num_valid_ctx)` lane offset likewise
produces the identical set of `query_off` values, and the launch grid
(`num_blocks = cdiv(max_target_query_len + num_query_per_req, BLOCK_SIZE)`)
covers both layouts.

**Draft KV-cache-group annotation / `use_eagle` lookahead — CLEAN.**
Upstream #52047 (`93ab92be0c`) generalises `_annotate_eagle_groups` to key off
`non_causal_multi_token_decode` on `MLAAttentionSpec`. Our draft group is a
plain `SlidingWindowSpec`, and the target is not `model_version == "deepseek_v4"`,
so **both** stacks annotate nothing and fall through to the identical
"flag every group" fallback (`vllm/v1/core/kv_cache_coordinator.py:99-105`,
byte-identical to upstream). `use_eagle()` includes `"dflash"` on both sides
(`vllm/config/speculative.py:1314`).

**`parallel_drafting_token_id` / mask token — CLEAN.**
`vllm/v1/worker/gpu/spec_decode/utils.py` is byte-identical to upstream. The
checkpoint puts `mask_token_id: 154856` inside `dflash_config`, which the fork's
`drafter_config.get("mask_token_id")` path resolves
(`qwen3_dflash.py:410`); upstream's extra top-level fallback is unreachable here.
#51602 touches only the V1 proposer, which this configuration never constructs
(`VLLM_USE_V2_MODEL_RUNNER=1`, and `_is_dflash2_draft()` forces V2 anyway,
`vllm/config/vllm.py:557-561`).

**Weight loading — CLEAN.** I dumped the safetensors header: 96 tensors, no
`midlayer.` prefix, no `encoder.` aliases, no `embed_tokens`/`lm_head`, no
`mask_embedding`, no `attention_sink_bias`. Every name maps to a parameter under
the fork's hand-rolled `DFlashQwen3Model.load_weights`
(`qwen3_dflash.py:668-712`) exactly as it does under upstream's
`AutoWeightsLoader` + `hf_to_vllm_mapper`; the mapper's two extra rewrites
(`midlayer.` → `layers.0.`, Muse `encoder.*`) are no-ops for this file. Shapes
confirm the geometry: `fc.weight [6144, 36864]` = 6 taps × 6144,
`o_proj [6144, 8192]` = 64×128, conv `base_kernel [2, 2, 6144]` and
`kernel_projection [1536, 6144]` = 2·2·(6144/16).
`#53435` (`decoder_layer_cls`) is present (`qwen3_dflash.py:375`).

**`get_top_k_tokens` (vocab-parallel candidate extraction) — BYTE-IDENTICAL** to
upstream (`vllm/model_executor/layers/logits_processor.py:241-286`), including
the org-vocab padding mask, the shard-offset conversion and the TP all-gather +
global top-k. SGLang's `compute_candidates` does the same thing
(`sglang .../models/dflash.py:1095-1140`).

**Draft causality resolution — CLEAN.** `"is_causal": false` at config top level
is honoured by `_dflash_layer_causal` (`qwen3_dflash.py:58-67`), so
`dflash_has_any_non_causal` is True and `attn_vllm_config` sets
`use_non_causal=True` (`dflash/speculator.py:92-99`). Identical to upstream and
to SGLang's `ENCODER_ONLY` mapping.

**`build_attn_metadata` / `attn_utils.py` — CLEAN.** The 462-line diff is the
`allocate_kv_cache`/`_reshape_kv_cache` move to `worker/utils.py` plus
encoder-only and `is_prefilling` plumbing. No semantic change reachable from the
draft's metadata build.

**`_prepare_dflash_inputs_kernel` sampling/indexing — CLEAN.**
`sample_idx = req_idx*num_speculative_steps + (query_off - 1)`,
`sample_indices[sample_idx] = query_idx`, padded rows written with
`sample_idx_mapping = -1` (`dflash/speculator.py:585-600, 619-627`). Upstream's
`-1` init in `__init__`/`capture()` (vs the fork's `0`) only affects values
observed *during* capture; the walk kernel's `valid = req_state >= 0` predicate
is evaluated at replay from data the runtime kernel just wrote, so replay
behaviour is identical.

**Draft logits spec plumbing — CLEAN.** The fork's `draft_logits_spec` override
in DFlash2 (fp32, `-inf` fill, `dflash2/speculator.py:131-135`) does override the
base method — same name on both sides (`speculator.py:276`). The base's fp32
default (vs upstream's `head_dtype`) is a deliberate, documented revert of
#50910 and is inert under greedy.

**Not audited / out of scope**: MLA sparse indexer, tiered-MoE dispatch,
Marlin, the KV-connector and offloading stack, mamba/hybrid managers,
multimodal, and the multi-module-MTP speculator — none is on DFlash2's path.

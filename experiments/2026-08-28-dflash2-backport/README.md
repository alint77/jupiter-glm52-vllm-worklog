# DFlash2 on the 5.2 target: reopening the speculator width question

`incoai/GLM-5.3-DFlash2` — 4.58 GiB, **6 layers**, sliding attention (window
2048), GQA with 8 KV heads, hidden 6144, `num_target_layers: 78`, vocab 154880.
Native DFlash2 checkpoint (`architectures: ["DFlash2DraftModel"]`).

Phase 28 refuted DFlash and recorded the lever as closed. This phase reopens it
for one specific reason, stated up front so the reopening is auditable: **the
Phase 28 refutation was about verify-batch width, and DFlash2 is half the width
of the DFlash that was refuted.**

| | MTP3 | DSpark t=8 | DFlash1 t=15 | **DFlash2 t=7** |
| --- | ---: | ---: | ---: | ---: |
| verify batch | 4 | 9 | 16 | **8** |
| draft layers | — | 3 SWA | 5 full-attn, 64 KV heads | **6 SWA, 8 KV heads** |
| status | shipped | refuted (28) | refuted (28) | **this phase** |

## Why Phase 28 does not already settle it

Phase 28's own numbers give a predictive cost model. Decode throughput is
accepted tokens per unit time, and the three measured points fit
`tok/s = acceptance_length / step_time` to within 1.4%:

| verify width | step time | acceptance | predicted | measured |
| ---: | ---: | ---: | ---: | ---: |
| 4 | ~27 ms | ~2.9 | 107.4 | 106.08 |
| 9 | ~42 ms | 3.98 | 94.8 | ~93 |
| 16 | ~97 ms | 6.84 | 70.5 | ~70 |

Width 8 was never measured. Two bounds bracket it:

- **Pessimistic** — linear interpolation on the measured curve between width 4
  and width 9: `27 + 15 × 4/5` = **39.0 ms**.
- **Optimistic** — the pure routed-MoE slope of 1.06 ms/token measured at c4,
  i.e. holding draft cost constant: `27 + 4 × 1.06` = **31.2 ms**.

The optimistic bound is defensible *for DFlash2 specifically*. Phase 28 noted
its curve is steeper than the routed-MoE slope "because the draft models
themselves also differ", and DFlash2's draft is far cheaper than DFlash1's:
6 sliding-window layers with 8 KV heads against 5 full-attention layers with 64.

Break-even acceptance length against MTP3's 107.4 tok/s:

| assumed step time | break-even acceptance |
| --- | ---: |
| 39.0 ms (pessimistic) | **4.19** |
| 31.2 ms (optimistic) | **3.35** |

Published DFlash2 acceptance lengths, block 8, against a GLM-5.3 target on
GB300/TP4: GSM8K 5.94, MATH-500 6.02, HumanEval 5.48, MBPP 4.95, MT-Bench 4.19.
Their worst published task sits exactly on the pessimistic break-even; every
other task clears both bounds. At HumanEval's 5.48 the prediction is 141 tok/s
(pessimistic) to 176 tok/s (optimistic), against MTP3's 106.08.

**So the entire question reduces to one measurable quantity: does a drafter
trained on GLM-5.3 hidden states retain acceptance against this 5.2 W4G64
target?** Everything else is arithmetic already in hand.

Decision rule fixed before the run:

| measured acceptance | verdict |
| --- | --- |
| < 3.35 | refuted — cannot beat MTP3 under any step-time assumption; stop |
| 3.35 – 4.19 | ambiguous — the measured step time decides it |
| > 4.19 | beats MTP3 under both bounds; proceed to graph/tile retuning |

## Compatibility audit: a 5.3 drafter on a 5.2 target

GLM-5.3 is the same base model as GLM-5.2 with different post-training; the two
`config.json` files are identical apart from `transformers_version` and the FP8
`quantization_config`. Checked before committing any time to the port:

| property | DFlash2 | 5.2 target | verdict |
| --- | --- | --- | --- |
| `fc.weight` | `[6144, 36864]` = 6 × 6144 | hidden 6144 | ok |
| `target_layer_ids` | `[5,19,33,47,61,75]` | 78 layers | ok |
| selector codebooks | `[154880, 256]` ×2 | vocab 154880 | ok |
| `embed_tokens`, `lm_head` | **absent from checkpoint** | borrowed from target | ok |
| `mask_token_id` | 154856 | see below | ok |
| `is_causal` | `false`, top level | needs #52816's resolver | included in port |

The mask token looked like a trap and is not one. DFlash1 uses 154821
(`[MASK]`); DFlash2 uses 154856. Both the 5.2 and 5.3 tokenizers stop at 154855
(`<|video|>`), so 154856 is a reserved embedding row in **both** — and reserved
rows take no gradient, so post-training leaves them identical. The checkpoint
ships no `mask_embedding.pt` override, so the drafter reads the target's
`embed_tokens[154856]` either way, and that row is the same tensor on 5.2 and
5.3.

The genuine risk is the one that cannot be settled by inspection: the drafter
reads target hidden states at six layers, and post-training moved every weight
that produces them. W4G64 quantization perturbs them further, though MTP3
already tolerates that class of shift at 79.8% acceptance. Speculative decoding
is lossless regardless — the failure mode is slow, never wrong — which is what
makes this cheap to test.

## Port: Option B, targeted backport

Branch `dflash2-backport` off `cec73c66b3` (`known-good-1238882`). The fork base
is untouched.

Upstream carries full DFlash2 support in exactly two commits, both after this
fork's base of 2026-08-01:

- `b389ac2946` — #52816, 2026-08-20, "DFlash2: local convolution + candidate
  selector", 866 lines across 14 files
- `a9a17e7095` — #53435, 2026-08-25, "Dflash2 load fix", 115 lines across 2

Cherry-picking both produced 7 conflicts, but **6 of the 7 are upstream drift,
not fork conflict**. Against the merge-base `d08eebad16` (2026-07-16) the fork
changes none of those files:

| conflicting file | lines the fork owns |
| --- | ---: |
| `models/qwen3_dflash.py` | 0 |
| `models/registry.py` | 0 |
| `gpu/sample/gumbel.py` | 0 |
| `gpu/spec_decode/__init__.py` | 0 |
| `gpu/spec_decode/speculator.py` | 14 |
| `tests/test_config.py`, `tests/v1/spec_decode/test_dflash_causality.py` | 0 |

The 14 fork-owned lines in `speculator.py` are this branch's DCP support for
draft attention (`prepare_dcp_local_seq_lens`), which is upstream PR #48392 —
still open there. That answers, in the affirmative, whether the draft path
works under DCP4 on this branch.

Taken from the two commits: the four new files (`qwen3_dflash2.py`,
`gpu/spec_decode/dflash2/`, `tests/v1/spec_decode/test_dflash2.py`), the
`decoder_layer_cls` / `model_cls` indirection, the top-level `is_causal`
resolver, `LogitsProcessor.get_top_k_tokens`, the `gumbel_noised_argmax`
extraction, `_is_dflash2_draft`, and the DFlash2 speculator dispatch.

Deliberately dropped as unrelated upstream work that happened to share a
conflict region:

- Muse-Glimmer registry aliases and the `hf_to_vllm_mapper` in
  `qwen3_dflash.py`. This branch's `load_weights` does its own
  `stacked_params_mapping`, which is why DFlash1 loads today without a mapper.
- the `extract_hidden_states` speculator dispatch
- `gumbel.py`'s `processed_logits` → `logits_cache` rename, kept at the fork's
  version
- the unrelated DSA / ROCm / breakable-cudagraph tests in `test_config.py`

Two deliberate deviations from upstream, both to keep the MTP3 control valid in
the same binary:

- `DraftModelSpeculator.draft_logits_spec` defaults to `torch.float32` rather
  than upstream's `head_dtype`, which is what this branch already allocated.
  DFlash2 overrides it; MTP3 is bit-identical to the known-good binary.
- `test_dflash2.py`'s mock `cache_config` gains `calculate_kv_scales=False`.
  This branch's `attention.py:271` still reads it; upstream dropped the read
  after the test was written.

## Status

- [x] Checkpoint pulled to `models/GLM-5.3-DFlash2`, 4,918,859,112 bytes,
      safetensors header verified against file size, 96 tensors
- [x] Backport applied and committed as `2a26f151ac`; imports clean, registry
      resolves `DFlash2DraftModel`, all pre-commit hooks pass including mypy
- [x] 17 backported unit tests pass
- [x] **`tests/v1/spec_decode/` shows zero regression.** Branch and base each
      fail exactly the same 121 tests, with no test failing on one and not the
      other. Those 121 are pre-existing on `known-good-1238882` and are the
      network-dependent suites (`test_speculators_*`, `test_vocab_mapping`)
      that cannot run under `HF_HUB_OFFLINE=1`. Lists kept as `fail-branch.txt`
      and `fail-base.txt`.
- [ ] **Blocked: a draft-aware placement profile.** See below.
- [ ] c1 server bring-up at `num_speculative_tokens: 7`
- [ ] Acceptance length against the matched MTP3 control

### The one blocker

The tiered planner budgets draft weights only when `method == "mtp"`, so the
4.58 GiB DFlash2 draft plus its sliding-window KV is invisible to it and must
be freed from the hot tier by hand. Phase 28 established that raising
`TIERED_MOE_HBM_RESERVE_GB` cannot substitute, because the fail-closed audit's
`required_free` scales 1:1 with the planned reserve; the profile has to be
trimmed instead.

Production is `hybrid-p0.5-replicas-985.json` at **2,870 hot slots/rank**
(11,480 across 4 ranks, 75 routed layers, 19,200 secondary). The 2026-08-28
VRAM breakdown put safe headroom at 1.50 GiB and the exchange rate at ~51
experts/GiB, so covering ~4.7 GiB of draft needs roughly **163 fewer hot
slots/rank, i.e. ~2,700** — close to the 2,720 Phase 28 used for the larger
DFlash1 draft, though against a different target so the numbers do not
transfer directly. That estimate needs the fail-closed audit to confirm it.

Two ways to produce it, and they are not equivalent:

1. Regenerate with `benchmarks/optimize_routing_profile.py
   --hot-slots-per-rank 2700`, which needs the routing `--trace-dir` the 985
   profile was fitted on. That trace is not in this directory.
2. Trim the existing profile in place, dropping the lowest-value hot experts
   per rank. Cheaper, but `hot_experts` and `secondary_ranks` must stay
   mutually consistent or the loader fails closed, and the file carries no
   explicit priority order to trim along.

**Neither affects the number this phase exists to measure.** Acceptance length
is a property of the drafter and the target's weights, not of expert
residency: a lower-residency configuration computes identical logits and
accepts identically, only slower. So the acceptance arm can run on whatever
trimmed profile loads, and residency only has to be right for a later
throughput arm — which the decision rule says not to start until acceptance
clears.

## Result: the control arm, and what it does to the thresholds

Job 1524929, `2a26f151ac`, node jpbo-026-48, trimmed profile, c1.

Two independent replicates on different nodes and different filesystems:

| MTP3 control | 1524929, GPFS | 1525205, fscratch | pooled | Phase 28 assumed |
| --- | ---: | ---: | ---: | ---: |
| acceptance length | 3.0783 | 3.0659 | **3.0721** | ~2.9 |
| step time | 27.52 ms | 27.86 ms | **27.69 ms** | ~27 ms |
| implied tok/s | 111.85 | 110.06 | **110.95** | 107.41 |
| draft acceptance | 69.28% (345/498) | 68.86% (345/501) | — | — |
| per-position accepted | 141/114/90 of 166 | 145/114/86 of 167 | — | — |
| semantic smoke | exact | exact | — | ` Paris. Distance from Paris to Lyon is` |

The replicates agree to **0.40% on acceptance and 1.21% on step time**, on
different nodes, so the baseline is not a single-node artefact.

**The harness is validated.** Step time reproduces Phase 28's ~27 ms to within
2%, on a different target and a trimmed profile, and the deterministic smoke is
byte-exact. The control is doing its job.

The measured baseline is slightly stronger than the one the decision rule was
written against, so the break-even moves up. Calibrating from the **control
arm** rather than from Phase 28's approximation is legitimate — it is the
baseline being measured more precisely, not the candidate moving a goalpost —
but both are recorded so the substitution is visible:

| step time for width 8 | pre-registered (2.9 @ 27 ms) | calibrated (pooled 3.0721 @ 27.69 ms) |
| --- | ---: | ---: |
| optimistic 31.2 ms | 3.35 | **3.46** |
| pessimistic 39.0 ms | 4.19 | **4.33** |

DFlash2 will be judged against both. Its published acceptance on a GLM-5.3
target ranges 4.19 (MT-Bench) to 6.02 (MATH-500); against the calibrated
pessimistic bound of 4.36, even its worst published task no longer clears, so
the outcome now hinges more tightly than before on how much acceptance
survives the move to a 5.2 target.

One presentation defect found and fixed here: `capture_acceptance.py` printed
the candidate verdict for the control arm too, so the run log shows
`3.0783 -> REFUTED` against MTP3 itself. That line is meaningless — the
control cannot fail a rule about beating itself — and the script now suppresses
it for control labels. The number it reports is correct.

## Scripts

| | |
| --- | --- |
| `run-server-dflash2.sh` | launcher; carries the Phase 28 operational lessons forward, requires `TIERED_MOE_PLACEMENT_PROFILE` explicitly rather than defaulting to the production one |
| `capture_acceptance.py` | acceptance length from the server's Prometheus counters, as a delta across one fixed generation so warmup cannot contaminate it; prints the Phase 42 verdict directly |

Serving notes for the bring-up, not yet validated:

- `--speculative-config '{"method":"dflash","model":".../GLM-5.3-DFlash2","num_speculative_tokens":7}'`
- `_is_dflash2_draft()` forces the V2 model runner unconditionally
  (`config/vllm.py:560`). This branch already runs V2, so this is aligned, but
  no V1 fallback remains reachable for this draft.
- CUDA graph capture sizes are `[4,8,12,16]` in production, sized for MTP3's
  four query tokens. Width 8 needs `[8,16,24,32]`; the first c1 measurement
  needs only `[8]`.
- The draft is non-causal (`is_causal: false`), so it needs a non-causal-capable
  backend, as DFlash1 did in Phase 28.
- Open upstream bugs against this exact path, not carried here: #54041 (DFlash2
  SWA draft KV groups marked as eagle) and #53978 (spec warmup with unfilled
  draft buffers).

Anything measured beyond acceptance length is premature until the decision rule
above is resolved. The Marlin tile and graph-capture retune that width 8 implies
is real work and is not started.

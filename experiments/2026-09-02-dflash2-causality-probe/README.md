# Phase 52: DFlash2's CUDA graph path costs 43% acceptance

**The fork's DFlash2 port is not broken. Its draft CUDA graph is.**

Running the identical arm with the draft's CUDA graph disabled takes GSM8K
acceptance from **3.9951 to 5.7386** -- which is upstream sglang's number
(5.7236 / 5.7063, Phase 51) to within 0.26%.

| fork DFlash2, GSM8K, card protocol, 64 samples | acceptance | tok/s |
| --- | ---: | ---: |
| CUDA graphs **ON** (`fixedgreedy-gsm8k`, Phase 48) | 3.9951 | 92.1 |
| CUDA graphs **OFF** (eager) | **5.7386** | **116.2** |
| upstream sglang, for reference (Phase 51) | 5.7236 / 5.7063 | - |

**+43.6% acceptance and +26% throughput from one flag.** Phase 51 concluded
there was "a defect in the fork's DFlash2 port worth ~43% acceptance" and
pointed at the drafter's code. The defect is real and the magnitude was right,
but it is not in the drafter's math -- it is in graph capture/replay.

## Why five phases of auditing missed it

Every audit compared the serving path against a reference **given the same
inputs and the same loaded weights** -- the subsystem audit (48), the ops-graph
audit (49d) and the in-serving probe all share that design, as Phase 49f
recorded. Graph replay does not corrupt the math; it corrupts what the math is
fed. An instrument that holds the inputs fixed cannot see it, by construction.

The eager comparison was available the whole time and nobody ran it.

## Where the loss sits

Per-position acceptance, mean over steady-state windows of the server's own
`Per-position acceptance rate` lines:

| run | pos 0 | 1 | 2 | 3 | 4 | 5 | 6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **eager** | **0.911** | 0.810 | 0.720 | 0.637 | 0.549 | 0.481 | 0.420 |
| graph (`lattice-ctrl`) | 0.579 | 0.512 | 0.459 | 0.412 | 0.365 | 0.321 | 0.282 |
| graph (`fixedgreedy`) | 0.569 | 0.513 | 0.459 | 0.405 | 0.360 | 0.317 | 0.277 |
| MTP7 on this stack, for scale | 0.899 | - | - | - | - | - | - |

Two things to read off this:

1. **Eager position 0 is 0.911 -- MTP-class.** MTP7 measures 0.899 at position
   0 on this same stack. Phase 49f established position-0 acceptance is recall
   times the target's mode-agreement rate to first order, with recall measured
   at 64.1%. At 0.911 against a ~0.9 mode-agreement ceiling, **recall under
   eager is essentially 100%**. The candidate set was never weak; the graph
   path was starving it.
2. **The loss is a uniform ~0.63x at every position**, not concentrated at
   depth. That is not the lattice walk (49b already cleared it, and a walk
   fault is position-dependent). It points at the candidate set or the hidden
   states feeding it.

This also retires, for the second time and now conclusively, the
untrained-mask-embedding anomaly (49d) and the "drafter is weak on math" reading
(49e): the same weights, same target and same protocol produce 5.74 when the
draft runs eagerly.

## How it was found

By accident, and the accident is worth recording. The
[causality probe](#the-causality-probe-refuted) forces `need_eager` so its
Python body executes at all -- and that flag disables the draft's CUDA graph
for the whole process. The probe run was a full 64-sample benchmark that
happened to come back at 5.7386.

A subagent audit of the probe had flagged exactly this as a control worth
exploiting: *"arming the env forces every propose call eager for the life of
the process... run the benchmark with the probe index set beyond the run. If
eager acceptance still lands at 3.995, the probe's verdict describes the path
that produced the number. If it does not, the graph path is implicated."* It
did not.

## The causality probe: refuted

The probe was built to test a different hypothesis -- that the fork's draft
block attention is silently causal, because it hardcodes
`attn_type = AttentionType.DECODER` (`qwen3_dflash.py:301`) and threads
non-causality through attention *metadata*, where upstream sglang derives
`attn_type = ENCODER_ONLY` from `is_causal:false` (`dflash.py:104-135`).

Method: perturb the token ids at the late query slots (2..7), re-run the same
forward eagerly, and compare slot 1 (the first mask, which predicts position
0). Backend-agnostic on purpose -- no mask to dump, no FA internals to trust.

```
max|d| at slot 0 (anchor)=1.859e+00, slot 1 (first mask)=1.675e+01,
slot 2 (perturbed)=8.562e+00  ->  NON-CAUSAL (early slots see later ones)
```

Slot 1 moves, so the fork's metadata route works. The anchor moved too, which
is the cross-check for a genuinely symmetric window rather than something else
leaking. **`attn_type = DECODER` is a cosmetic divergence from sglang, not a
defect.**

## Audit of the graph path: what is ruled out

Read against `origin/main` and the live buffers:

| checked | verdict |
| --- | --- |
| `get_dummy_slot_mappings` / `get_dummy_block_tables` | **persistent views** (`self.slot_mappings[:, :num_tokens]`), documented as required for capture -- slot mappings are live at replay |
| `InputBatch.make_dummy` | writes into `input_buffers.seq_lens` and returns a view -- seq_lens are live |
| `sample_indices`/`sample_pos`/`sample_idx_mapping` | zeroed before capture, read as device buffers at replay; `sample_indices` stores `query_idx < num_query_tokens`, in bounds |
| `dflash2/speculator.py` vs upstream | diagnostics only; the Gumbel keying is renamed, logic identical |
| `cudagraph.py` vs upstream | sole deletion is DCP code, deliberate (fork fails closed at `cp_size > 1`) |
| `_topk_override` | `VLLM_DFLASH2_SELECTOR_TOPK` defaults to 0, falsy, never fires |
| TP all-gather in `get_top_k_tokens` | allocates from the graph pool; addresses stable across replays |

**Refuted suspect, recorded so it is not re-raised:** capture builds attention
metadata with `max_seq_len=max_model_len` (400000) while the real path uses
`draft_max_seq_len` (a few hundred). This is *not* the cause: `seqused_k` is a
live tensor, FA treats `max_seqlen_k` as an upper bound, and nothing in
400000-vs-300 changes *which* keys slot 1 attends to.

## Next

1. **Confirm** (`eager-clean` 1619723 / `graph-baseline` 1619724, running):
   probe index beyond the run so `need_eager` is forced but the probe never
   fires, against a no-env graph run on identical source. Gate: ~5.74 and
   ~4.0.
2. **Bisect the graph** with `cudagraph_mode: PIECEWISE` instead of
   `FULL_AND_PIECEWISE`. PIECEWISE graphs the transformer body and leaves the
   DFlash2 head -- `compute_candidates`, the selector, the walk -- eager. If
   PIECEWISE gives ~5.7 the fault is in the head under capture; if ~4.0 it is
   in the body. One arm.
3. The uniform-ratio loss with position 0 hit hardest points at the candidate
   set: the `compute_candidates` -> `get_top_k_tokens` chain, or the hidden
   states feeding it. Audit found nothing there by reading, so step 2 is the
   discriminator.

**Do not ship the eager workaround as the fix without step 2.** It costs the
graph's launch savings on every draft step, and on this stack DFlash2 already
matches MTP7's step time (Phase 51); the 116.2 tok/s measured here is *with*
that cost, so the real fix should be worth more.

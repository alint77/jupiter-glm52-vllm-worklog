# Phase 53 — Fork/upstream divergence in the DFlash base speculator

**Date:** 2026-09-02
**Branch:** `dflash2-backport`
**Trigger:** scoping the #52188 DCP port (user: "option B"). The fork-vs-upstream
diff surfaced three divergences unrelated to DCP, two of which are documented
upstream as CUDA-graph correctness fixes.

## Why this matters

Phase 52 established the fork's DFlash2 draft CUDA graph costs 44% acceptance
(eager 5.7190 AL / 114.1 tok/s vs graph 3.9626 / 91.8), with eager matching
upstream sglang (5.7236) to 0.08%. The root cause was never found. Two of the
divergences below are graph-specific by construction.

## Provenance — these are missed upstream fixes, not fork decisions

Merge-base `d08eebad16` ([Perf][MoE] Write FlashInfer combine into final output
#47156) already contains the fork's values:

- `sample_idx_mapping = torch.zeros(` (merge-base:77)  — same as fork
- `sample_pos[:num_sample] - 2` (merge-base:256)       — same as fork
- no `is_valid_ctx` / `num_valid_ctx`, no `cp_local_slot`

So the fork **predates** all three; it did not deliberately change them.
Adopting upstream's versions is not reverting a fork decision.

## The divergences

### 1. `sample_idx_mapping` init: fork `0`, upstream `-1`  [latent, greedy-inert]

Upstream's rationale (`speculator.py:92`):
> `-1` marks an inert sampling row. CUDA graph capture can execute the full
> buffer before a real batch has populated it, so zero would make every padding
> row scatter into request slot 0.

Also `reset()`: fork `.zero_()` vs upstream `.fill_(-1)`.

Caveat (advisor): both kernels already store `-1` for padding *inside*
`prepare_dflash_inputs` (fork:723, upstream:673), so the init difference only
bites at capture time. That cannot by itself explain a probe result of
112/112 candidate entries differing — a slot-0 scatter corrupts request 0 only
(1/8 of the batch at bs=8). Test it, but it is not the lead hypothesis.

### 2. `is_valid_ctx` compaction: fork absent  [live, both modes]

Upstream:
```
num_valid_ctx = valid_ctx_end - ctx_start
is_valid_ctx  = j < num_valid_ctx
is_query      = (j >= num_valid_ctx) & (j < num_valid_ctx + num_query_per_req)
query_off     = j - num_valid_ctx
```
Fork uses untrimmed `num_ctx` for all three. The fork *computes*
`valid_ctx_end` (line 603) but uses it only for `last_valid_pos` (612).

Consequence: with `num_rejected > 0` the fork writes **real positions and real
KV slots** for the rejected suffix rows; upstream gives them `pos=0` and
`PAD_SLOT_ID`.

Corrected framing (an earlier draft of this file overstated it):

- The `is_query`/`query_off` shift is **lane assignment, not layout**.
  `query_idx = query_base + query_off` targets a separate buffer and
  `query_off` spans `[0, nq)` either way, so query outputs are byte-identical
  between the two kernels. Only the ctx rows differ.
- Upstream's "so a replayed graph cannot observe a stale value" comment covers
  rows left *unwritten*. The fork **does** write them (`mask=is_ctx` spans the
  full range), just with real values. That graph rationale does not apply to
  the fork's divergence.
- The real defect: rejected positions are `last_valid_pos+1 .. +num_rejected`,
  and the query rows write `last_valid_pos+1+off` — **the same slots**. So the
  fork has a duplicate-slot write race inside a single forward, in both eager
  and graph. Real, but nondeterministic and small, which is consistent with
  eager still matching sglang at 5.7190.

Verified live, not a no-op:
- `num_rejected` is loaded **inside** the kernel (line 602) from
  `num_rejected_ptr`; `ctx_end` comes from `target_query_start_loc` — the
  caller does **not** pre-trim, so `num_rejected` is genuinely nonzero.
- `DFlash2Speculator(DFlashSpeculator)` overrides only `_generate_draft`; it
  inherits `propose` and the base `prepare_dflash_inputs`. So this kernel **is**
  on the path that measured 5.72 eager / 3.96 graph.

### 3. Gumbel offset `-2` vs `-1` — NOT A BUG, parked

Fork passes `sample_pos - 2`; its `sample_draft` then does `positions + 1`
(`speculator.py:341`, NOTE by woosuk). Upstream passes `sample_pos - 1` and its
`sample_draft` passes `sample_src_positions` straight through with no `+1`.
Both land on `Q-1`. Pure refactor. Affects eager and graph identically anyway,
and is inert under greedy.

## Neither A nor B is the graph defect

`prepare_dflash_inputs` runs **outside** the captured region. In `propose()`:

- line 81  `prepare_dflash_inputs(...)`
- line 146 `dispatch_cg_and_sync_dp(...)`   <- graph mode decided here
- line 206/221 `self._generate_draft(...)`  <- the only captured callee

and DFlash2's `_generate_draft` override contains no `prepare_dflash_inputs`
call. So the kernel's outputs are **inputs** to the graph, recomputed eagerly
every step in both modes; they cannot differ between eager and replay.

That rules out both A and B as explanations for the Phase 52 gap
(eager 5.7190 vs graph 3.9626) and narrows the remaining suspect to what
actually executes differently under replay: DFlash2's `_generate_draft` body,
and the padding rows a replay processes that eager does not.

One padding lead checked and dropped: `out_query_slot_mapping_ptr` padding is
filled with `PAD_SLOT_ID` identically in fork (line 733) and upstream (line
681), so a stale slot overwriting valid KV on replay is not a fork defect.

## Plan

Keep these **unbundled** from the DCP port so each number is attributable:

- commit A: `sample_idx_mapping` `-1` sentinel
- commit B: `is_valid_ctx` compaction
- commit C: DCP port (#52188) — `cp_local_slot`, kernel CP params,
  `cudagraph.py` DCP block, drop the `NotImplementedError` guard, drop the now
  redundant `GUARD_NULL_BLOCK` env (upstream's `ctx_resident`/`q_resident` is
  the same guard, always on), plus the `dp_sync`/`query_start_loc_np`/
  `dcp_local_seq_lens` signature work and the post-rewind
  `prepare_dcp_local_seq_lens` re-run.

Isolation arms A and B run against the Phase 52 eager 5.7190 baseline in
parallel with C.

### Correctness gate, corrected

The originally planned gate ("byte-identical slot mappings at cp_size=1") is
**wrong for commit B**: `is_valid_ctx` changes the mapping by design on any
batch with rejections. Byte-identity applies only to commit C at `cp_size=1`,
where `cp_local_slot` provably reduces to the current formula — upstream's
helper is mathematically identical to the fork's own inline CP math at
`block_table.py:289-300`, which is already proven under DCP4 in production.

## Arms

- **B**: eager-with-B vs the eager 5.7190 baseline. Not a graph arm — B is
  mode-independent. Expected >= baseline, slightly above if the duplicate-slot
  race was costing anything. If eager gets *worse*, revert: upstream being
  upstream does not make it right for this drafter.
- **A**: no arm. Provably inert under `draft_sample_method: greedy`, which is
  what every arm so far has run.

## Commit C — what the port actually needed

Narrower than first scoped, because the fork already had several pieces:

| piece | state |
|---|---|
| `cp_local_slot` | **added** to `cp_utils.py`, byte-identical to upstream |
| kernel CP params + `// (block_size * CP_SIZE)` | **added** |
| `GUARD_NULL_BLOCK` env | **removed**; upstream's `ctx_resident`/`q_resident` is the same guard, always on |
| `cudagraph.py` DCP block | **restored** from `origin/main` (was a pure 13-line deletion, no fork additions) |
| `NotImplementedError` DCP guard | **removed** |
| post-rewind `prepare_dcp_local_seq_lens` | **already present** in the fork's base speculator |
| `query_start_loc_np` | not needed — multi-module MTP's non-uniform query layout, unrelated to DCP |
| `dp_sync` / `DPSyncState` | not needed — a DP refactor, unrelated to DCP |

## Gates (both PASS, on-GPU, no allocation)

See `gates/`. `gate_cp1.py`: at `cp_size=1` the ported kernel is bit-identical
to the pre-port kernel across all ten outputs, 72 configs. `gate_cp4.py`: at
`cp_size>1` ctx and query slots match an independent torch reference of the
DCP round-robin mapping, 72 configs across every rank of cp_size 2/4/8.

## Status

- commit A landed: `85defa32c2`
- commit B landed: `5c5dc1ac54`
- commit C applied, gates pass, pending commit

## Next

1. Arm B+C at `cp_size=1`, eager — must reproduce the 5.7190 baseline.
   This is the regression check that the gates cannot cover (they test the
   input kernel, not end-to-end acceptance).
2. Arm the prod config: DCP4, c=4, 400K, DFlash2 — the configuration the
   `NotImplementedError` used to refuse.
3. Graph defect still open. Suspect narrowed to DFlash2's `_generate_draft`
   body and replay-only padding rows; `need_eager` remains the interim
   mitigation worth +44% acceptance.

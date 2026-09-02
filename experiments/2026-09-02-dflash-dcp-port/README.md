# Phase 53 arms — the #52188 DCP port

**Date:** 2026-09-02  **Jobs:** 1621605-1621608  **Branch:** `dflash2-backport`

The code change and its rationale live in
[2026-09-02-dflash-upstream-divergence](../2026-09-02-dflash-upstream-divergence/README.md).
This directory is the measurement.

## Why these four

The two on-GPU gates prove the *prepare kernel* is bit-identical at
`cp_size=1` and correct at `cp_size>1`, but they test that kernel in
isolation. Only an end-to-end arm shows commit B -- masking rejected suffix
rows to `PAD_SLOT_ID` -- did not move acceptance, since B deliberately changes
the kernel's output on any batch with rejections.

| arm | DCP | seqs | draft graph | compare against |
| --- | --- | --- | --- | --- |
| `bc-eager-dcp1`   | 1 | 1 | off | Phase 52 eager **5.7190** |
| `bc-graph-dcp1`   | 1 | 1 | on  | Phase 52 graph **3.9626** |
| `prod-eager-dcp4` | 4 | 4 | off | (new — previously refused) |
| `prod-graph-dcp4` | 4 | 4 | on  | (new — previously refused) |

Both prod arms run because the graph defect is still open and worth ~44%
acceptance, so the eager number is the one to quote for DCP4 until it is
fixed.

The draft graph is disabled the way Phase 52 did it: `VLLM_DFLASH2_PROBE`
set past the end of the run, which forces `need_eager` without the probe ever
firing.

## Protocol

Identical to Phase 51/52 so the numbers are comparable: GSM8K, 64 samples,
`replicate_dflash2_eval.py`, GLM-5.3-NVFP4 target with the GLM-5.3-DFlash2
drafter, `num_speculative_tokens: 7`, greedy draft sampling, one node, TP4.

```bash
bash agent_space/experiments/2026-09-02-dflash-dcp-port/submit.sh
```

`arm-dcp.sh` is `arm-replicate.sh` with `DCP` and `SEQS` lifted to env vars.
`submit.sh` freezes a snapshot per submission — editing a live arm script has
destroyed running arms before.

## Results

### DCP1 — the port is clean, and B is not a regression

| arm | AL mean | median | tok/s | Phase 52 baseline | delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `bc-eager-dcp1` | **5.7675** | 5.862 | 117.6 | 5.7190 | **+0.85%** |
| `bc-graph-dcp1` | **3.9699** | 4.0329 | 90.8 | 3.9626 | +0.18% |

Eager came in slightly *above* baseline (and 117.6 vs 114.1 tok/s), which is
the direction expected if the duplicate-slot write race commit B removed was
costing anything; it is small enough to be noise either way. What matters is
the negative result: **neither B nor C regressed CP1**, which the gates could
not show for B, since B is present on both sides of the `gate_cp1` diff.

The graph arm reproduces Phase 52 to 0.18%, so the 44% graph-vs-eager gap is
untouched by any of A/B/C — as predicted once `prepare_dflash_inputs` was
found to run outside the captured region.

### DCP4 — blocked below the port, in FlashAttention

Both prod arms died at engine init, eager and graph alike:

```
RuntimeError: scheduler_metadata must have shape (metadata_size)
  qwen3_dflash.py:647  in forward
  flash_attn.py:924    in forward
  flash_attn.py:1223   in _forward_with_dcp
  flash_attn_interface.py:347 in flash_attn_varlen_func
```

Read this carefully, because it is a *good* sign for the port: the draft is
now reaching `_forward_with_dcp` at all. Before commit C it could not run
under DCP; the slot math is being exercised. The failure is one layer down,
in the FA backend's DCP context path, which rejects the draft's shape.

The distinguishing feature is almost certainly the draft's multi-token query.
The same DCP4 topology runs in production under MTP
(`2026-08-29-glm53-c4/server.sbatch`), where `max_query_len == 1`; the DFlash
draft presents `max_query_len == 8`, which is what gates
`split_dcp_context_queries` at `flash_attn.py:574` and feeds
`should_split_fa2_dcp_context_attention` at 1170. The `scheduler_metadata`
built at 601 (`batch_size=num_reqs`, `causal=False`,
`seqlens=dcp_context_kv_lens`) is then handed to the plain varlen call at
1239 alongside `num_splits=attn_metadata.max_num_splits`.

**Not attributable to this port**, and the backend bisect (1621784-1621788)
now says there is no way around it:

| arm | DCP | outcome |
| --- | --- | --- |
| `tri-dcp1` | 1 | **5.7650** AL, 117.39 tok/s -- works |
| `tri-dcp4` | 4 | `AssertionError: DCP requires attention implementations to return the softmax LSE during decode, but TritonAttentionImpl does not` |
| `fi-dcp1` | 1 | `NotImplementedError: FlashInfer backend on SM90 currently crashes with sliding-window attention layers` |
| `fi-dcp4` | 4 | same NotImplementedError |

The DCP1 controls earned their place: FlashInfer fails **identically at DCP1
and DCP4**, so its failure has nothing to do with DCP -- it cannot serve this
all-SWA draft on SM90 at all. Triton serves the draft fine at DCP1 but is
structurally incapable of DCP, since the DCP combine needs the softmax LSE
and `TritonAttentionImpl` does not return it.

**So FlashAttention is the only backend that can serve DFlash2 under DCP on
GH200.** There is no backend workaround; the fix has to land in FA's
`_forward_with_dcp` or in the draft's attention-metadata build.

`tri-dcp1` is also a free cross-check on commits B and C: 5.7650 / 117.39 on
a completely independent draft-attention backend, against `bc-eager-dcp1`'s
5.7675 / 117.6 -- agreement to 0.04%. The eager acceptance is real and
backend-independent.

## Root-cause investigation (in progress)

### Correction: `model.layers.80/81` are DRAFT layers

The GLM-5.3 target has **78** hidden layers (0-77) and is **MLA**
(`kv_lora_rank: 512`, `fp8_ds_mla`), so it never instantiates
`FlashAttentionMetadataBuilder` at all. The draft has 6 layers, occupying
indices 78-83. So every line in the instrumented log -- build *and* call --
belongs to the draft. An earlier reading of this file claimed the builds were
target layers and that the draft's metadata was "never built by the DCP
branch". That was wrong.

### What the instrumented arm showed

```
build: layers=['model.layers.80.self_attn.attn'] batch=4 causal=False num_splits=0 shape=(5,)
build: layers=['model.layers.81.self_attn.attn'] batch=4 causal=False num_splits=0 shape=(5,)
call:  batch=4 causal=False num_splits=0 max_q=8 q_shape=(32,64,128) shape=(5,)
```

Build and call agree on batch, `causal`, `num_splits` **and** the passed
shape. FA3 still rejects it.

### Why that leaves exactly one explanation

`metadata_size` also depends on the `cache_seqlens` **values**, which the
first probe missed by holding them constant:

| cache_seqlens | size |
| --- | ---: |
| all 0 / 1 / 64 | 5 |
| all 4096 | 9 |
| mixed [0,0,0,4096] | 9 |

and `dcp_context_kv_lens` reaches the forward as a **live view of a mutable
persistent buffer** -- `self._dcp_context_kv_lens[:num_reqs]`
(`flash_attn.py:558-560`), stored into the metadata at 686. `scheduler_metadata`
is baked at build time from those values; `seqused_k` is read at call time from
the same view. If the values change in between, the two desync -- built for
near-empty context (5), called against real lengths (FA3 wants 9).

This is DFlash-specific for a reason the fork already documents: the draft
"advance[s] and rewind[s] its own global sequence lengths", and the draft
builds metadata per layer/group, each build overwriting the shared buffer that
earlier layers' metadata still points at.

Not yet confirmed end-to-end: an arm logging the actual `kvlens` values and
`id()` of the buffer at both ends is running (1623176).

### Two candidate fixes, deliberately not chosen yet

1. **Stale view** -- snapshot the lengths into the metadata rather than storing
   a view, or rebuild the schedule at call time. Small, in the FA builder, and
   would mean the draft *can* be DCP-sharded.
2. **The GQA constraint** -- upstream's own rule is
   `tensor_parallel_size // total_num_kv_heads >= dcp_size`; the draft is
   64 heads / **8 KV heads** at TP4, giving `4 // 8 = 0`. `model.py:1247`
   never checks it here: the check is gated on `not self.use_mla` and the
   target *is* MLA, and it validates the target's config, while the draft's is
   verified separately against `draft_parallel_config`. If this is operative,
   the draft cannot shard at TP4 and needs `cp_size=1` block tables of its own
   -- a far larger change, because `cp_local_slot` now PADs 3/4 of the draft's
   context slots per rank, so an unsharded draft would read unwritten entries.

The vLLM DCP blog also lists "better support for MTP and speculative decoding"
as **future work**, so incomplete draft support here is expected rather than
surprising.

## The crash is fixed; a second, structural problem remains

**Fixed (`4bfece3f59`).** `_forward_with_dcp` passed the layer's *raw* window
while the builder plans the AOT `scheduler_metadata` against a *symmetrized*
one -- `_maybe_symmetrize_window` turns a causal `(w, 0)` into `(w, w)` for a
non-causal layer. FA3's `metadata_size` depends on the window ((2047,0) -> 13,
(2047,2047) -> 9), so build and call disagreed. The non-DCP path already
prefers `attn_metadata.sliding_window` (`flash_attn.py:956`); the DCP path did
not. Reproduced and fixed in isolation by `gates/gate_window.py`.

It needs DCP **and** a non-causal layer **and** a sliding window at once, which
is why only the DFlash2 draft hits it -- the GLM-5.3 target is MLA and never
builds FlashAttention metadata. Beyond the crash it was a correctness bug: the
raw causal window would make a bidirectional layer attend backwards only. This
is a genuine upstream bug and worth a PR.

**Not fixed: the draft cannot legally be DCP-sharded at TP4.**

`get_num_kv_heads` is `max(1, total_num_kv_heads // tensor_parallel_size)` and
**DCP does not enter it**. The draft has 8 KV heads at TP4, so each rank holds
**2 distinct** KV heads *and* a context shard. `_forward_with_dcp` all-gathers
the query to 64 heads and pairs them against those 2 kv heads -- a GQA ratio of
32, where the true ratio is 64/8 = 8. Each rank can only correctly serve the 16
q heads that map to its own kv heads, so the LSE combine is wrong.

That is precisely what upstream's rule guards:
`tensor_parallel_size > total_num_kv_heads` forces the `max(1, ...)`
*replication* branch, where every rank holds the same kv head and context
sharding is coherent. `4 > 8` is false, so `max_dcp_size = 4 // 8 = 0`.
`model.py:1247` never checks it here -- the check is gated on `not use_mla`,
our target *is* MLA, and it validates the target's config while the draft's is
verified separately against `draft_parallel_config`.

### Prediction, recorded before the arms land

- `fix-eager-dcp1` should sit within noise of 5.7675 (the fix must not perturb
  the unsharded path).
- `fix-eager-dcp4` should be **materially below** 5.7 without crashing.

  **Two independent defects both push it down, and the number cannot
  distinguish them.** Besides the head math, `cp_local_slot` PADs 3/4 of the
  draft's context slots on each rank, so the draft attends over a quarter of
  its own sliding window regardless of whether the head math is sound. A low
  number therefore confirms **that per-model block tables are needed** -- which
  fixes both -- and not the head-math story specifically.
- `fix-graph-dcp4` will be ~40% below its eager twin regardless, because the
  Phase 52 graph defect is still open. Compare eager-to-eager only.

Realistic outcomes, stated so the result is not read to fit a preferred story:

| outcome | reading |
| --- | --- |
| **low, no crash** | per-model block tables are the fix; do not attribute it to the head math alone |
| **crash at a new site** | progress; report the new site |
| **~5.7** | *both* defects would have to be benign -- very unlikely, so treat it as evidence this analysis is wrong and stop to re-examine, not as a clean falsification of one hypothesis |

Either way the fix for the prod config is to keep the draft at `cp_size=1`
(replicated draft KV is cheap -- 6 layers, SWA 2048) while the target runs
DCP4. That needs per-model block tables, a real chunk of work in
`BlockTables`/`InputBuffers`, and it addresses both defects at once. Whether
that is worth doing versus DCP4-with-MTP (which already works today) plus the
qrep cherry-pick is a call for the user, not an obvious yes.

## Replicated drafter KV — where it stands

Five defects fixed; DCP4 runs end to end for the first time; acceptance is
still 39% short of DCP1 and the cause is not yet found.

| commit | defect |
| --- | --- |
| `4bfece3f59` | `_forward_with_dcp` passed the raw, not the planned, sliding window |
| `796ac4a99f` | AOT metadata scheduled with the global KV dtype, not the group's |
| `e796779f6c` | drafter's KV sharded under DCP despite invalid GQA head math |
| `2b08ee712e` | `dcp_shardable` dropped by spec merge / promotion / projection |
| `b3985ea97a` | tests for the above (74 pass in the file) |

### Results

| arm | AL | median | tok/s | pos0 | pos1 | pos2 | pos3 | pos6 |
| --- | ---: | ---: | ---: | --- | --- | --- | --- | --- |
| `repl2-eager-dcp1` | 5.6265 | 5.6557 | 116.7 | 77% | 67% | 59% | 50% | 31% |
| `repl2-eager-dcp4` | **3.4434** | 2.5458 | 43.9 | 62% | 47% | 36% | 30% | 16% |
| `fix-eager-dcp4` (before) | 1.7630 | 1.7451 | 35.8 | — | 48% | 20% | 8% | 0% |
| `mtp-dcp1` | 4.8762 | 4.8877 | 92.6 | 92% | 70% | 56% | 47% | 19% |
| `mtp-dcp4` | 4.9920 | 5.0211 | 82.6 | 86% | 72% | 56% | 48% | 22% |

The failure signature changed: from a collapse past position 1 (wrong context
slots, deepening with position) to a roughly uniform ~20% deficit at *every*
position including position 0.

### Two hypotheses raised and killed

- **The target's hidden states under DCP.** Refuted by the MTP control: MTP
  shares the target and its hidden states but none of the DFlash KV plumbing,
  and its acceptance is *flat* at DCP4 (4.9920 vs 4.8762), 11% slower where
  DFlash2 is 62% slower. The target's DCP path costs nothing.
- **The drafter overwriting the target's `dcp_local_seq_lens`.** Refuted by
  construction: the speculator builds its own `InputBuffers`
  (`speculator.py:126`), separate from the model runner's
  (`model_runner.py:221`).

### A real cost of replication, which is *not* the acceptance cause

`max_concurrency` falls from **4.00x to 1.00x** (and "GPU KV cache size"
with it -- the log line is literally `max_concurrency * max_model_len`, so
they are one number). DCP4 normally quadruples effective capacity by giving
each rank a quarter of the context; an unsharded drafter group forfeits that.

Do not over-explain this. Per-token arithmetic says unsharding the drafter
should cost ~1.4x, not 4x: the target's MLA is ~45 KB/token at fp8 (~11 KB
sharded 4 ways) against the drafter's ~6 KB/token replicated. Promotion to
`FullAttentionSpec` changes how many blocks are *held*, not their size, so it
is at most a minor contributor. The remaining ~3x is unexplained and lives in
`get_kv_cache_capacity`. An earlier note in this file blamed the promotion;
that was wrong and the arithmetic is above.

It also cannot explain the acceptance deficit: GSM8K prompts are a few hundred
tokens against a 400K limit, and all arms logged **zero preemptions**.

### Next

Test whether the drafter's KV is actually dense at DCP4: count `PAD_SLOT_ID`
entries among valid rows of `_context_slot_mappings` after
`prepare_dflash_inputs`. At `CP_SIZE=1` it should be zero; 3/4 PAD would mean
the CP1 slot math is not reaching the context path. That is the one direct
measurement of the drafter's own KV that has not been taken.

## Status

Port validated at DCP1 on two independent draft-attention backends and
shipped. DCP4 blocked in FlashAttention, which the bisect establishes is the
only backend capable of DCP here -- so that is where the next fix goes.

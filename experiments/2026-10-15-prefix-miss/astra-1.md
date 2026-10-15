[codex] Starting Codex task thread.
[codex] Thread ready (01a13f68-baf6-71c1-9bc4-1ea7613bb6fd).
[codex] Turn started (01a13f68-cab0-7b11-8e16-6c97676385d1).
[codex] Assistant message captured: **The measurements show an early lookup failure, not necessarily loss of most cached blocks.*...
[codex] Turn completed.
**The measurements show an early lookup failure, not necessarily loss of most cached blocks.** A single missing required block at token 37,376 can hide the remaining ~203K. The varying boundaries—including 24,320 → 22,784—fit moving holes better than a fixed retention limit.

### 1. Ranked hypotheses

1. **Incorrect mutation of reused blocks during #113’s allocation, commit, or cleanup.** This best fits “cold fill works; first reuse works; next reuse fails.” Audit `allocate_new_computed_blocks` → manager cached-block cursor → `cache_full_blocks`, plus explicit invalidation/movement and deferred cleanup. A wrongly positioned reused block could enter the “partial→full promotion” branch and have its existing hash removed. Incorrect duplicate-key removal could similarly orphan another block’s mapping. Either needs only one early damaged entry to produce the observed cap; subsequent damage earlier in the prefix explains #160→#161. **The excerpts establish these candidate paths, not that they execute incorrectly.**

2. **Actual group layout exposes a lookup/adoption inconsistency absent from the guessed CPU layouts.** Two concrete concerns:
   - `find_longest_cache_hit` trims only the **first full-attention spec group**. Other full-attention groups can retain lists longer than the reconciled hit after another group reduces it. Whether this corrupts subsequent bookkeeping depends on the unshown adoption code.
   - Production lookup passes DCP=4 to every `FullAttentionSpec` finder. Verify that the finder respects `dcp_shardable=False` for replicated drafter groups and agrees with each manager’s effective block size.

   A purely static lookup mismatch should generally also affect #113. A mismatch that corrupts state **when adopting its hit** fits better.

3. **Production hash identity changes despite identical token prefixes.** Token equality excludes tokenizer divergence, but not differing hash extra keys, mutable hash lists, or incorrect hash updates around speculative output. A first differing hash near 35K also terminates lookup there. Stable initial prefixes make a request-wide salt change less plausible.

4. **Retention/masking or an erroneous cache-progress cursor prevents recaching.** This fits #16→#17’s unchanged hit. However, the shown mask only skips insertions: it cannot explain #112’s already-hittable blocks becoming unavailable without an additional removal, mutation, or lookup problem. Likewise, genuine capacity eviction remains unsupported; verify actual physical-block accounting before diagnosing an accounting bug.

### 2. What serving adds

- **MRv2, async placeholders, MTP:** relevant through finalized-token counts, rollback/invalidation ranges, and cache-commit timing. The cap at `request.num_tokens` is safe only if that quantity excludes noncommittable placeholders. Ordinary three-token lookahead cannot directly explain a ~200K reduction.
- **Deferred free/interleave scheduling:** reproduce the actual ordering of adoption, commit, rollback, and deferred release. Ordinary reference-count release should preserve cached hashes. A late operation targeting reused blocks could violate that invariant.
- **DCP and group specs:** capture actual manager classes, effective block sizes, group IDs, `dcp_shardable`, and propagated eagle flags. Spec equality propagates eagle behavior to every group in that spec bucket.
- **Host UVA:** storage placement alone should not change CPU hash-table membership. Its relevance is custom allocation, grouping, migration, or invalidation.
- **Eagle/alignment:** ordinary boundary rounding/drop costs roughly boundary-sized amounts, not this collapse. All hits being multiples of 256 is expected and does not identify the cause.

**Instrumentation caveat:** the shown `find_longest_cache_hit_per_group` omits `dcp_world_size`; it is not a faithful diagnostic substitute for production lookup under DCP4.

### 3. Best single probe

**Snapshot #113’s successfully reused prefix as resolved `(group, hash) → block_id` mappings, then compare that immutable snapshot at #114 admission.** Stop at the first discrepancy:

- Different request key: hashing/extra-key problem.
- Same key, mapping gone: removal or map corruption.
- Same key and valid mapping, finder stops: lookup/alignment/group reconciliation.
- Mapping present but block metadata disagrees: bookkeeping corruption.

This is more discriminating than an un-cache stack Counter alone: it identifies the exact first obstruction and separates key changes, state destruction, and lookup failure.

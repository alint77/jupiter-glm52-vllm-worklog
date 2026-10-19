For the stated configuration, the main → map stream → main ordering looks sound. I don’t see a demonstrated hot-buffer race in that sequence. The remaining correctness risks are invalidation outside the normal KV-update path, stream lifetime assumptions around the stager, and eligibility predicates that are less identical than the description suggests.

This is a diff-only review; the omitted CUDA kernel, FlashMLA implementation, and runner paths prevent unconditional sign-off on those contracts.

1. **Ordering and allocation lifetime**

   The intended dependency chain is correct:

   ```
   Grace/reserved writes → side map → main FlashMLA → next same-layer update/map
   ```

   `side.wait_stream(main)` includes the preceding writes and top-k production, provided their producers have already joined main. `main.wait_event(ready)` protects both the mapped indices and hot-buffer contents. Later same-layer mutations on main, or maps that fork from main, therefore cannot overtake that attention read.

   The stager now receives the correct pair: **the original Grace pointer and original converted slots**. It gathers the skip-layer caches, so it does not race with replacement of the anchor’s hot rows.

   **`record_stream(main)` is appropriate for the conversion outputs allocated on the side stream.** The event wait establishes execution ordering; `record_stream` establishes allocator lifetime. Keep both. Persistent `hot.out` and cache tables do not need per-call lifetime recording while their owner remains alive.

   There is a third consumer, however: `SkipKVStager.stream` reads `indices`. The shown code does not record that use. In the stated pipeline, the next skip attention waits for staging, and subsequent map-stream allocations occur after that wait; this can provide the necessary transitive protection. Thus this is **not a proven use-after-free in this configuration**. I would nevertheless add:

   ```python
   indices.record_stream(self.stream)
   ```

   inside `SkipKVStager.stage`, covering its enqueued reads. That makes the lifetime contract explicit instead of depending on subsequent layer execution.

   During CUDA-graph capture, the fork and join should become graph dependencies. `record_stream` is allocator bookkeeping, not replay synchronization. Across graph replays, safety requires serialized launches or an explicit dependency between launch streams. Merely reusing `hot.ready` does **not** serialize separately launched graphs. Shared graph memory pools also retain their usual replay-order requirements.

   The drafter is harmless to the hot state **if it only executes layer 78 and does not write target-layer caches**. Excluding layer 78 from `_ANCHORS` establishes the former exclusion here; the drafter’s actual write behavior is outside this diff.

2. **Invalidation and block reuse**

   Your freshness rule is sufficient, with a precise formulation:

   > Every change to a physical slot’s KV bytes must invalidate or refresh its hot entry before any subsequent read through that entry.

   Freeing or reassigning a block alone need not invalidate anything. If its bytes remain unchanged, a hot hit still returns the same bytes as Grace. New ownership becomes relevant when bytes change. Attention must also exclude unwritten portions of newly assigned blocks.

   This covers ordinary preemption/recompute, speculative rollback, and block reuse **when the subsequent writes pass through this hook**. Retained prefix-cache blocks need no invalidation if their contents remain unchanged.

   **Potential P1: external writers bypass this hook.** KV transfers, swap-in/offload restoration, block copies/COW, and direct cache population can change a destination without invoking `do_kv_cache_update`. Such paths need destination invalidation, or a conservative hot-cache reset, ordered after the external write completes and before attention. If those features are disabled, enforce that restriction. This diff alone does not demonstrate support.

   **Potential P1: the predicates are not literally identical.**

   - `kv_update` resolves metadata through `md[0]` and a layer-name lookup.
   - Attention receives its metadata directly.
   - `decodes()` checks `num_prefills` and `num_actual_tokens`.
   - `start_mqa()` additionally checks `fp8_use_mixed_batch`.
   - Actual mapping occurs below earlier attention branches.
   - KV update considers the whole `slot_mapping`, whereas map invalidates the prefix passed as `new_slots`.

   For your uniform 4/8/12/16 decode graphs, these may agree perfectly. But the required invariant is stronger than the current helper:

   > If KV update skips invalidation, this invocation must subsequently map, and its `new_slots` must include every nonnegative slot written by that update.

   I would enforce that contract using the actual execution metadata and padded token count. In particular, do not silently use `md[0]` if multiple metadata entries can represent different invocations.

   Padding with `-1` is fine: `_invalidate_kernel` ignores it, and the reserved write masks it. Positive slots outside the mapped prefix would not be fine.

   Finally, verify against `anchor_hot.cu` that clearing `where` and `slot_of` while leaving `status` untouched is a supported invalidation transition. The integration relies on that kernel contract.

3. **FlashMLA equivalence**

   At the data-model level, this transformation is correct:

   - Preserve index position, multiplicity, and `-1`.
   - Substitute another physical address containing identical packed KV bytes.
   - Preserve the query, scales, valid counts, and split schedule.

   RoPE information already encoded in KV does not need adjustment merely because the physical row changes. Computing valid counts and the schedule from the original indices is appropriate if mapping preserves validity and order.

   The `[257, 64, 656]` allocation is correct for 16,384 hot rows plus 64 reserved rows.

   **What I cannot certify from this diff:** whether the exact FlashMLA implementation uses cache dimensions or physical indices for anything beyond addressing and bounds. The scheduled path needs those properties verified in its wrapper and device code.

   The non-scheduled `_fp8_flash_mla_kernel` path needs a separate audit of `dummy_block_table`, `cache_lens`, shape-derived metadata, and kernel dispatch. Success of the explicitly scheduled call does not establish that wrapper’s equivalence.

   Also, the reserved-row write re-quantizes KV instead of copying the packed Grace row. Using the same `concat_and_cache_mla` inputs, format, and scale should reproduce the bytes, but byte identity is another contract to validate. Comparing the two packed writes directly would isolate it from attention arithmetic.

4. **Planner accounting**

   Given your stated counts and uniform main-cache geometry, the main adjustment is correct:

   ```text
   moved = 22 indexer layers − 1 MTP layer = 21
   per_layer = total main-cache bytes / main-layer count
   ```

   Both modes add `21 × per_layer` to host usage. `"free"` subtracts that amount from main HBM usage and adds the hot allocations. The MTP layer remains charged in HBM.

   The slot-count calculation is correct if `main_page_bytes_per_layer` is the physical byte size of one main-cache block and the division is exact.

   **`table_bytes()` slightly overcounts the shown persistent tensors overall.** The exact per-layer tensor payload is:

   ```text
   656 × (rows + 64)
   + 4 × (num_slots + 2 × rows + 32 + 3 × max_entries)
   ```

   The two reserved-map arrays are shared across layers and together cost `2 × 64 × 8 = 1,024` bytes. The current formula instead charges 512 bytes per layer: 10,752 bytes for 21 layers, an overcount of 9,728 bytes. That is conservative and small.

   These figures exclude allocator rounding, events, conversion/schedule temporaries, and graph-pool effects. They are tensor payload accounting, not an exact CUDA memory footprint.

   **`"same"` needs a capacity assertion:** the retained charge for the moved main caches must cover the actual hot-cache allocation. This is likely true for your production KV pool, but it does not follow for arbitrarily small pools or larger configured hot caches.

5. **Bitwise check mode**

   The integer reinterpretation is the right way to compare bits. It detects signed-zero and NaN-payload differences as well as ordinary numerical differences.

   Bitwise equality is a sound expectation **if this FlashMLA path has a fixed reduction order for the supplied schedule**. Identical schedules alone do not prove that: an implementation could still use nondeterministic accumulation or incompletely initialized outputs. I cannot certify determinism without the omitted implementation.

   Establish a control first: run **Grace versus Grace** with the same schedule, in both eager and graph execution. Then hot-versus-Grace mismatches become much more informative. Keep separate output and LSE mismatch counts.

   The pinned-host reporting is weaker than the GPU comparison:

   - Serialized main-stream `add_` operations can accumulate correctly.
   - Concurrent execution on multiple streams could lose updates because these are not atomic increments.
   - The logging thread has no completion synchronization and reads two separately updated counters. Its snapshot is not guaranteed to be consistent.

   Treat the thread output as progress telemetry. For a definitive result, wait for GPU completion and read the counters afterward, or use an explicitly synchronized snapshot.

   Also, check mode currently **asserts on mapped non-scheduled calls**. Either reject unsupported configurations during initialization or implement the corresponding reference wrapper call.

6. **Other fragile points worth fixing**

   - **Initialize before capture.** Build/load the extension, allocate hot state and check state, and warm all relevant kernels before recording graphs. Lazy construction inside capture can capture initialization writes and introduces avoidable capture/allocation hazards.
   - **Scope `_CACHES` to the runner/cache lifetime.** The singleton has no reset or rebinding mechanism. A partially bound first invocation, cache reallocation, or another model instance can leave stale pointer keys and stale layer references.
   - **Avoid the first-anchor reserved-map dependency.** It is valid when every invocation executes the first anchor and all anchors have identical ownership masks. Recomputing this tiny map per anchor makes partial execution and differing metadata safer.
   - **Validate actual map dimensions.** Check `indices.numel() <= max_entries`, token count `<= 64`, supported width, and slot-mapping coverage before launching the extension. The constructor’s capacity assertion does not validate individual calls.
   - **Fail early on unsupported format/device configurations.** The implementation hardcodes 656-byte FP8 rows, block size 64, target-layer numbering, and an SM90a extension.

   My acceptance conditions for this deployment would be: explicit writer coverage or enforced connector/offload exclusions; enforced map/update eligibility and slot coverage; initialization before capture; and a Grace-versus-Grace determinism control followed by the integrated bitwise check. The proposed stream ordering itself does not appear to require redesign.

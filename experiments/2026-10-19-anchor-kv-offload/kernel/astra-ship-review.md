**The side-stream ordering is sound.** I see no demonstrated new race in the supplied code. Ship approval for the exercised serving configuration remains conditional on the pending CHECK/A/B results; piecewise execution and cache wake/reinitialization are not established safe by this diff.

1. **Reserved writes are correctly ordered.**

   For each anchor, the dependency chain is:

   `Grace write → side wait → reserved write → map → ready → main FlashMLA`

   The next fork’s `side.wait_stream(main)` also includes the preceding FlashMLA read, so it protects both the reserved rows and the ordinary hot rows from premature overwrite. This relies on the main-stream wait on `hot.ready` occurring before FlashMLA, as stated.

   Per-anchor `reserved_map` and the added `record_stream(side)` calls address the relevant storage lifetime concerns. `k_scale` needs no additional lifetime recording if it remains a persistent model tensor.

2. **`expect_map` is an eager/capture diagnostic, not a replay-time guard.**

   A complete eager forward or FULL capture executing both Python calls leaves it balanced. FULL replay executes neither Python mutation, which is fine when the captured graph contains the complete pair.

   Piecewise execution needs verification at the actual compilation boundaries. For example:

   - Captured KV update plus eager mapping: replay does not set the flag, so mapping asserts.
   - Eager KV update plus captured mapping: replay does not clear it, so the next update raises.
   - Separate fragment warmups can execute an update repeatedly before its corresponding map.

   None of those partitions is demonstrated by the excerpt, so I would not claim that piecewise definitely fails—or that it works.

   Metadata that is `None` or has zero actual tokens does not set the flag. A dummy/profile path carrying positive decode metadata **does** set it; skipping attention afterward leaves it set. Clearing it alone would be incorrect: that update also skipped invalidation, so the written slots must be invalidated if mapping is abandoned.

   Taking the earlier disposition as established, MTP metadata sharing alone is harmless: its layer-78 cache is absent from `self.caches`, so the public `kv_update()` wrapper returns.

3. **No additional writer is demonstrated here, but completeness cannot be established from this excerpt.**

   The zeroing and CoW hooks cover those operations provided their writes and invalidations are ordered before the next map. Ordinary prefix-cache lookup/block-table reassignment introduces no KV write. Reuse becomes relevant when the underlying slots are overwritten.

   For DCP prefill, an earlier `kv_update()` invalidation is sufficient **if it covers every slot subsequently written by `forward_mha`, with no intervening map**. Any additional exchange/copy destinations outside that slot set need coverage too. The relevant prefill implementation is not supplied, so that remains the specific unresolved writer question.

4. **Rebinding through `bind_kv_cache()` replaces the caches; pointer identity does not solve other lifecycle transitions.**

   Ordinary allocator reuse is less concerning because each hot cache retains `self.grace`. The important cases are:

   - **Rebuild after capture:** existing graphs still reference the previous hot buffers/tables, while eager invalidation hooks use the replacement `_CACHES`. Rebuilding therefore requires retiring and recapturing those graphs.
   - **Discard/reset at unchanged addresses:** sleep/wake or in-place KV reinitialization can invalidate contents without changing `data_ptr()` or calling `bind_kv_cache()`. These require explicit lifecycle handling.

   If sleep discards the hot allocations, restoration must reinitialize their constants too—including `reserved` and `block_offsets`—not merely clear `where`.

   For any supported sleep/elastic/reinit path, establish either restoration/reset in place or rebuild plus recapture. Otherwise, reject that combination.

5. **There is also a concrete device assumption in the new block hook.**

   In `invalidate_blocks()`:

   ```python
   block_ids.to(torch.int64)
   ```

   preserves the input device. CPU index tensors can work with the existing PyTorch CoW indexing operation, but the new addition to CUDA `self.block_offsets` will fail for them.

   Unless both callers already guarantee CUDA indices, normalize explicitly:

   ```python
   block_ids.to(device=self.block_offsets.device, dtype=torch.int64)
   ```

   The broadcast to every hot cache additionally assumes a common 64-slot block namespace and valid block range for every anchor. `_invalidate_kernel` has no upper-bound protection; establish that invariant from the cache configuration.

   Your reported unhooked-CoW control is useful evidence that the module test detects stale-cache reads. It does not cover the runner’s index-device contract, graph partitioning, or lifecycle transitions. For the original configuration, the remaining acceptance evidence is the pending in-situ CHECK and served A/B; if ship scope includes the additional execution/lifecycle modes above, those need validation or explicit exclusion.

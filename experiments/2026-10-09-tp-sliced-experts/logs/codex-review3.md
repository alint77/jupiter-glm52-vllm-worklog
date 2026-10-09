V22’s barrier protocol looks sound. V23’s ring indexing and S0 accounting also look sound, subject to the launch and helper contracts below. I’m reasoning only from the supplied material.

1. **V22 synchronization**

   The two half barriers protect different boundaries:

   - The opening barrier ensures all four warps have finished reading the **previous flush’s** scratch before any warp overwrites it.
   - The second barrier makes all four warps’ current stores visible before the cross-warp loads.

   No trailing barrier is needed: the next flush’s opening barrier protects reuse, including when intervening stages do not flush. Scratch is separate from the stage ring, so releasing `empty_s` before flushing remains safe; `fs` and `row1` have already been captured.

   R0’s last flush has the required participation. All consumers see the same descriptor and `last`; every thread enters both half barriers, including threads whose output token is inactive. The `k < ntok * 16` predicate occurs afterward.

   Assuming `consumer_sync()` remains barrier 1 with 256 participants, and IDs 2–3 are otherwise unused, I see no barrier cycle. Each half completes its own barriers before joining barrier 1, and all eight consumers traverse the same R0 completion path. The producer participates in neither.

   The output mapping is correct for the existing eight-token layout: `k1 * 32 + lane` covers all 128 float4 positions per half exactly once, and `row1` selects the corresponding token. This replacement requires `ntok <= 8`; it no longer has the old `MAX_TOK / 2` loop.

2. **V23 pre-PDL fill**

   **Transaction accounting:** assuming `mbar_expect_tx` is the existing arrive-and-expect operation, posting the complete byte count before waiting is valid. The arrival alone cannot complete the phase while transactions remain outstanding. For positive T, completion of the weight TMA still leaves `T * XS_BYTES` outstanding until the activation copies finish.

   The essential contract is **`p.T == ws->T`**, with T within the supported one-producer-warp copying range. Expecting more bytes than are copied hangs; expecting fewer can permit premature completion and corrupt subsequent accounting. Using `p.T` consistently for both counts would simplify this invariant.

   **Descriptor publication:** `desc[st]` precedes the barrier arrival, matching the existing dynamic path. With the existing release-arrival/acquire-wait helpers, consumers obtain both the descriptor and completed payload. Moving descriptor writes before PDL introduces no dependency on `route_prep`.

   I would add a producer-warp `__syncwarp()` immediately before the static activation-copy block. `pdl_wait()` is not a warp synchronization primitive; this makes the ordering between lane 0’s setup and the other copying lanes explicit, matching the dynamic paths. That is ordering hardening, not a demonstrated failure of the current code.

   **Ring indexing:** let `m = su1 - su0`. Static stages occupy logical iterations `[0, m)`, so starting dynamic production at `it = m` is correct. On the first wrap, the existing empty-phase expression waits for the static occupants’ release. The end marker also lands correctly when there is no dynamic work, including when `m == STAGES`. Consumers correctly continue starting at iteration zero.

   **Accumulation boundaries:** intersecting the CTA interval with each tile gives the correct runs. Each tile boundary and the CTA interval’s endpoint receive `last`, so `accs` is cleared before another tile or dynamic S1 work. The unused first bit is harmless here because initialization and every preceding last flush establish zero accumulators.

   **Completion accounting:** the CTA intervals partition `[0, UNITS_S0)`, and their tile intersections partition those intervals. Therefore:
   ```
   sum(nch over all S0 last descriptors) == UNITS_S0 == 384
   ```
   Counting units rather than flushes remains correct. V23 also changes shared-expert accumulation grouping, so its numerical checks must cover that additional FP32 ordering change.

   **Early reads:** the newly moved work uses launch parameters, descriptors, weights, and CTA-local storage. Counts, epoch, activation rows, scales, queue counters, and output updates remain after PDL wait. The existing publication of `x13b` to asynchronous readers is still an inherited requirement.

   **Progress:** all local static activation copies are issued before the producer can block on `ready_s`, so there is no local producer/consumer dependency cycle. Static ownership does introduce a grid scheduling obligation: an unstarted CTA’s S0 slice must eventually run while other CTAs wait for shared completion. Whole-grid residency is a sufficient guarantee; `__launch_bounds__(THREADS, 1)` alone does not establish it. If the grid can exceed residency, check that remaining CTAs can retire and admit every outstanding S0 owner.

3. **Why M=8 may show no gain**

   The measured difference, 90.2 versus 90.7 µs, is about 0.55%; without paired variation, it does not establish a regression.

   V22 cuts routed global reduction submissions fourfold, but adds cross-warp synchronization and FP32 additions. At eight active tokens, aggregate scratch traffic is approximately unchanged: previously four warps each read their own contribution; now each output owner reads all four contributions. The main saving is global reductions.

   That saving may be small if weight movement, MMA work, readiness waits, or shared-expert work dominates. The new barriers also expose differences in progress between the four warps.

   Measure paired latency distributions, routed reduction traffic, barrier stalls, memory throughput, and register/spill changes. Include several routed `ntok` distributions: equal total M does not imply equal flush behavior.

   For V23, also measure the interval from PDL completion to the first consumable stage and shared/routed readiness publication. Its `td_p0` excludes the static activation issue and all pre-wait setup, so producer-duration comparisons need aligned instrumentation.

4. **Next three steps**

   1. Add the explicit producer warp sync, verify the T and grid-progress contracts, then check V23 across shared on/off, supported T values, sparse routed token counts, and both tier distributions. Keep `GR1 = 1`.
   2. Run paired timings for V20, V22, **V20 plus static S0**, and V23, using the same fence setting. This separates local aggregation from static assignment/prefill effects.
   3. Profile the winning variants. If aggregation adds barrier cost without relieving reduction pressure, retain V20’s flush. If static S0 helps, investigate its increased flush frequency, delayed routed starts, and actual overlap with `route_prep` before tuning further.

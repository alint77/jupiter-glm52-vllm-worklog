**v44/v45’s handoffs look correct. v46’s cumulative publication also looks correct, but `hq_full` reuse has a PTX phase-ordering hole.** Your ring-distance argument addresses generation mixing; it does not establish that someone has successfully waited on the completed phase before arrivals begin in the next phase.

I’m assuming the helpers implement the stated semantics with appropriate compiler memory clobbers, initialization is synchronized across the CTA, and `flush_rows` ends as described.

1. **The three memory handoffs**

   **(a) Scheduler → producer: yes.** The ordering chain is:

   ```text
   scheduler lanes: shared entry stores
       → scheduler __syncwarp()
       → scheduler lane 0: release-arrive
       → producer lane 0: successful acquire-wait
       → producer __syncwarp()
       → producer lanes: shared entry loads
   ```

   Both warp synchronizations matter: the first gathers the writers; the second distributes the acquired ordering. CTA scope covers the communicating threads. PTX gives successful acquire-waits visibility of accesses preceding release-arrivals, and warp barriers provide memory ordering among their participants. [PTX synchronization instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-test-wait)

   Your reverse `sq_empty` handoff also protects reuse: the producer’s final `__syncwarp()` puts every lane’s entry reads before lane 0 releases the slot.

   One small source correction: **check `e.kind == K_END` before constructing the complete `Group`.** The terminal publication initializes only `kind`; on a slot’s first use, `x/c0/nch` can be uninitialized. Avoid those source-level reads even if optimization would eliminate them.

   **(b) Ready probe → FIFO → async copies: yes.** For the successful-probe path:

   ```text
   remote x2 writes
       → remote ready release
       → scheduler ready acquire
       → FIFO release/acquire
       → producer __syncwarp()
       → each copying lane's fence.proxy.async
       → that lane's bulk copy
   ```

   The intermediate CTA synchronization does not discard the ordering imported by the GPU acquire. The scope needs to cover the endpoints of each synchronization hop. Keep the proxy fence in every issuing lane, after that lane receives the handoff; ensure its state-space coverage includes global memory. PTX specifies that proxy fencing composes with other synchronization. [PTX proxy fences](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-membar)

   This assumes `ready == epoch` remains valid for that entry and `x2` stays unchanged through its reads. A false probe merely chooses the existing fallback.

   **(c) Reductions → finisher → remote winner: yes, once the HQ protocol is repaired. Every consumer does not need its own GPU fence.**

   Each consumer’s ordinary `red.add.global` precedes its release-arrival. The finisher acquires all those contributions, then publishes them through its GPU fence and `done13` RMW. The winning RMW observes the preceding counter updates through the RMW chain; its following GPU fence establishes the acquire side before activation reads. PTX’s cumulative ordering and observation order through RMWs support this relay. [PTX memory model](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#memory-consistency-model)

   Two distinctions matter:

   - `atomicAdd` itself is relaxed. Your comment “lane 0’s acquire (the count)” should refer to **the count RMW plus the following fence**.
   - `__shfl_sync` distributes `done`; the subsequent `__syncwarp()` distributes memory ordering to the activation lanes.

   Freeing `hq_info[k]` before counting/activation is fine: all finisher lanes have loaded their local `h`, and the second `__syncwarp()` precedes the slot release. The queue storage can be reused while activation continues.

2. **Phase generations: SQ is sound; HQ needs an additional gate**

   For each SQ slot, the sequence is:

   ```text
   publish full[g]
       → producer successfully waits full[g]
       → producer releases empty[g]
       → scheduler successfully waits empty[g]
       → publish full[g+1]
   ```

   This protects both payload reuse and barrier phases. Your parity arithmetic matches that sequence. A failed claim followed by stealing leaves `n/k` unchanged, which is correct. The terminal slot needs no empty acknowledgement because it will not be reused.

   For HQ, there are **two separate requirements**:

   - Arrivals for the next logical use must not complete the previous use.
   - A completed barrier phase must have a successful `test_wait`/`try_wait` before any arrival in its successor phase. This rule already exists in PTX 8.7. [PTX mbarrier phase requirements](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier)

   Your four-stage versus twelve-chunk argument supports the first requirement for the stated configuration, assuming the omitted consumer ordering enforces that skew bound. It does **not** satisfy the second.

   A counterexample to the second requirement:

   ```text
   All consumers complete handoff g.
   The finisher has not successfully waited on full[k] for g.

   Consumers continue through enough R0 groups to reuse k.

   Warp 0 lane 0 blocks on empty[k].
   Warps 1–7 arrive on full[k] for g+HQ.
   ```

   Those arrivals enter the new barrier phase before any successful wait observed the old phase. This violates the documented protocol even though warp 0 prevents the new phase from completing.

   **The simple repair is to gate every consumer warp on `hq_empty`:**

   ```cpp
   const auto hand_off = [&](int4 h) {
     if (lane == 0) {
       if (hn >= TD_HQ)
         mbar_wait(&hq_empty[hk], ((hn / TD_HQ) - 1) & 1);
       if (warp == 0)
         hq_info[hk] = h;
     }
     __syncwarp();

     mbar_arrive(&hq_full[hk]);  // still initialized to 256

     if (++hk == TD_HQ) hk = 0;
     ++hn;
   };
   ```

   Now every arrival for the reused slot follows the finisher’s successful old-phase wait and slot release. This also removes the dependence on twelve-chunk groups and handles `K_END` cleanly.

   Parity remains safe with multiple empty-waiters: the finisher cannot release the slot again until every consumer has passed its current empty wait and contributed to the next full phase.

   Independently, the terminal handoff deserves attention in any distance-based proof: `K_END` can immediately follow the final R0 group. With `TD_HQ=1`, “another handoff” need not imply another twelve chunks. The explicit gate avoids that issue entirely.

3. **Deadlock: no additional queue cycle after that repair, subject to work ordering**

   The finisher’s structure is favorable: once it obtains an HQ item, counting and activation do not wait for producer progress or future HQ items. Consequently, HQ backpressure can be relieved independently of a producer spinning on `ready`.

   The remaining global invariant is:

   > Every R0 group required by an R1 group must be claimed before that dependent R1 group, and each CTA must issue its claims in FIFO order.

   Under that invariant, a blocked R1 depends on earlier-claimed R0 work. If that R0 is queued behind another blocked R1 on its owner, the dependency moves to an even earlier claim. Following these dependencies cannot form a cycle. Stealing preserves the argument if it preserves those ordering properties.

   **Claimed prerequisites may still be executing; they do not need to have completed before R1 is claimed.**

   Conversely, if `group_at` permits an R1 ahead of an unclaimed prerequisite—or permits a prerequisite to be queued behind its dependent R1 on the same producer—there is a real cycle:

   ```text
   producer waits ready
       → required R0 cannot be issued
       → consumers cannot submit its HQ completion
       → finisher cannot publish ready
   ```

   `group_at` and the complete consumer loop are absent, so I cannot certify that invariant from these excerpts. The new queues themselves provide no escape from an invalid work order.

   The HQ end entry is otherwise appropriate: all consumers contribute, and the finisher processes preceding local completion records before exiting.

4. **Performance: next steps for M=8**

   **First benchmark corrected v46.** Moving the reported ~6.3 µs of completion work off warp 0 is a plausible improvement, but the kernel saving depends on how much overlaps subsequent consumption. Watch whether that time reappears as HQ backpressure or delayed `ready`.

   Then I would prioritize these changes and measurements:

   - **Reduce HQ arrivals from 256 to 8.** With the repaired helper above, its `__syncwarp()` gathers each warp’s reductions and lane 0’s metadata write. Initialize `hq_full` to `CONSUMER_WARPS`, and replace the arrival with:
     ```cpp
     if (lane == 0) mbar_arrive(&hq_full[hk]);
     ```
     This uses the same relay reasoning as the scheduler FIFO. It reduces logical arrivals; the hardware speedup needs measurement.

   - **Reduce completion-fence overhead separately.** Only finisher lane 0 needs the post-count acquire fence, followed by the existing full-warp synchronization. A further experiment is a GPU `acq_rel` counter RMW replacing the pre-count fence plus relaxed RMW and post-count acquire fence. Preserve the activation publication chain while measuring this change.

   - **Investigate the remaining R0 non-math time.** Your numbers imply approximately:
     ```text
     R0: 34.7 − 31 × (0.61…0.64) ≈ 14.9…15.8 µs
     R1: 19.0 − 15 × (0.90 + 0.22) ≈ 2.2 µs
     ```
     Some R0 remainder is the completion work v46 moves. Re-measure what remains before changing the math pipeline. R1 flushing totals about 3.3 µs, which bounds the direct opportunity there.

   - **Trace the CTA that determines kernel completion.** Separate consumer full waits, producer empty waits, scheduler starvation, HQ reuse waits, and actual ready spins. If your listed buckets are disjoint, they total about 75.3 µs; that leaves roughly 12.7 µs relative to the 88 µs kernel duration to explain through other work or CTA imbalance.

   Also check registers and spills at **320 → 352 threads**, particularly around `activate_route`’s `a[]`. An extra warp can change the block’s register allocation constraint.

   I would give v45 little further attention unless useful-probe timing says otherwise. Count successful probes **actually used when `key != last_ready`**, and measure the exposed wait duration eliminated. A 95% raw hit rate can coexist with negligible savings.

No commands or tests were run.

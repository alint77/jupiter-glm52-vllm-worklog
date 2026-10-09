**I would keep the v26 design, add one warp synchronization before activation, and prioritize GR1=2 plus ready-aware dispatch for M=32.** The handoff itself looks sound under the stated ring invariants.

1. **Correctness: the missing edge is counter acquisition → activation by the other lanes.**

   `bar.arrive 2,256` from warps 1–7 and `bar.sync 2,256` from warp 0 correctly gather all 256 consumers. PTX explicitly includes ordinary `red` operations in the memory accesses ordered by the barrier. Through that synchronization, lane 0’s subsequent GPU fence can publish the other warps’ contributions before its counter increment. This does not require each contributing warp to execute its own GPU fence solely for this handoff. [PTX barrier semantics](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-bar-barrier)

   The counter’s returned value is used, so this is a returning atomic operation. Its read followed by lane 0’s winning-branch `__threadfence()` supplies the acquire side of the completion protocol. The issue is extending that acquisition to lanes 1–31: **`__shfl_sync` transfers the decision but does not guarantee memory ordering.** The other lanes’ fences are not themselves counter acquisitions. The final `__syncwarp()` comes after the activation reads and cannot supply the missing earlier edge. [CUDA shuffle and synchronization guarantees](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html#warp-shuffle-functions)

   The smallest proof-completing change is:

   ```cpp
   if (__shfl_sync(0xffffffffu, done, 0)) {
     __threadfence();
     __syncwarp();  // propagate lane 0's acquisition before any y13 loads

     const Expert& e = experts[ei];
     for (int r = 0; r < e.ntok; ++r)
       activate_route(ws, e.route[r], lane);

     fence_proxy_async();
     __threadfence();
     __syncwarp();
     if (lane == 0) st_release(&ws->ready[q][ei], epoch);
   }
   ```

   This is an ordering-proof gap, not a claim that your 15 passing repetitions exhibited corruption. The release/acquire construction relies on the counter read and subsequent fence; PTX explicitly permits those patterns and their transitive propagation through synchronization. [PTX acquire/release patterns](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#release-and-acquire-patterns)

   The **publication side already has the needed warp rendezvous**: every activation lane performs its stores and fences, then all meet before lane 0 releases `ready`. Preserve the copying lanes’ existing acquire/proxy protocol.

2. **The generation argument works, but state its invariant more precisely.**

   Let flush \(j\) occur after consuming ring unit \(u\). While its handoff generation remains incomplete, warp 0 cannot progress beyond that handoff. Because it releases the current empty slot *before* flushing, the other warps can consume up to `STAGES` further published units. They cannot reach the next routed flush if that requires more than `STAGES` units.

   Thus **`R0CH > STAGES` is sufficient if `R0CH` is the minimum spacing between successive routed handoffs in actual ring publications**, including transitions between groups, tiles, and entries. The strict inequality matters because the current slot has already been released.

   The assertion alone does not establish that spacing. It depends on:

   - Every consumer following the same published descriptor sequence.
   - Every relevant ring slot requiring warp 0’s empty arrival before reuse.
   - No shortened group or special transition producing a closer routed flush.
   - Barrier 2 having no other users.

   What must be excluded is a second arrival into an **unfinished** generation. A new generation beginning after reset while warp 0 is still doing completion work is not inherently an error.

   **Bar 1 interaction:** fast consumers reaching an S0 `consumer_sync()` or the K_END synchronization is safe provided all consumers encounter the same dynamic sequence of bar-1 operations. They have already arrived at the preceding bar 2, so waiting at bar 1 cannot withhold that arrival. Warp 0 finishes its finite count/activation work and catches up. Barrier IDs differ, but this program-order argument—not merely different IDs—establishes progress.

   **Empty-ring interaction:** a long activation eventually stalls the other consumers and producer through ring backpressure. That is a performance stall, not a new deadlock, because warp 0’s completion path does not require another ring publication. Likewise, a producer spinning on this entry’s `ready` does not prevent warp 0 from publishing it.

   Preserve the existing global dispatch progress argument: this change does not fix a schedule that strands unfinished w13 work behind resident w2 waiters. Also, delayed completion code must not retain live references into released ring storage; descriptor values and flush scratch must remain captured or independently owned.

   The most useful stress cases are maximum `ntok`, the smallest legal `R0CH > STAGES`, S0/routed transitions, and immediate K_END, with deliberate skew before handoff and during activation.

3. **One warp activating eight routes can matter, but I would measure before splitting.**

   The total activation work is unchanged, but serializing routes reduces parallelism and delays the single entry-wide `ready`. It also keeps warp 0 away from the ring longer. The ring hides only a bounded amount of that delay.

   Measure these separately, bucketed by `ntok`:

   - Last counter result → `ready` publication.
   - Producer ready-wait overlapping that interval.
   - Empty-ring stalls caused by the completion warp’s absence.

   A long activation interval matters only to the extent that it is exposed on the kernel’s critical path. The supplied aggregate timings do not isolate it.

   If it is exposed for larger `ntok`, **test a two-warp completion team first**. Warps 2–7 arrive and continue; warps 0–1 synchronize at the handoff. Lane 0 acquires completion, a separate 64-thread rendezvous distributes the result, the two warps process alternating routes, and another team rendezvous precedes publication.

   Do not try to reuse warp 1 after letting it advance, or insert `consumer_sync()` into warp 0’s winning branch: the required participants are no longer there. Two completion warps preserve most of the handoff benefit while reducing serial activation.

4. **For M=32, prioritize sustained throughput and exposed dependency stalls.**

   Using the shared-fused replay:

   | Batch | EP slowest | EP mean | Sliced | Reduction vs EP slowest |
   |---|---:|---:|---:|---:|
   | M=8 | 8.74 ms | 7.51 ms | 7.05 ms | 19.3% |
   | M=32 | 22.00 ms | 20.13 ms | 20.53 ms | 6.7% |

   At M=32, the 1.47 ms advantage decomposes into **1.87 ms of EP slowest-minus-mean spread, offset by sliced being 0.40 ms slower than EP’s mean**. The mean is not an attainable EP baseline by itself, but this is strong evidence that widening the margin requires reducing sliced execution time beyond its balancing advantage.

   A **15% latency lead** means reaching **18.70 ms**, saving another **24.4 µs/layer**. A 20% lead requires **39.1 µs/layer**.

   My order of work would be:

   - **Replay v26 GR1=2 first.** At `110/0` and `110/12`, it improves over v25 by 4.3 and 15.8 µs—1.9% and 6.1%. It is the strongest supplied M=32 candidate. Its regression at `38/0` argues for workload-sensitive selection if replay confirms the split. R0S=2 currently has weaker evidence.
   - **Use NOREADY to screen ready-aware dispatch.** Treat its delta as dependency-removal sensitivity, not a guaranteed recoverable saving or rigorous bound: it changes issue timing, contention, and the data consumed. The actionable measurement is how often a blocked producer has independent, dispatchable work available. Select complete work groups, preserve the handoff-spacing invariant, and retain acquisition before copying.
   - **Attack the weighted hot/cold work using replay counts.** For sliced, the fitted variable cost is `1.71H + 5.98C`. Compare prospective percentage reductions against those weighted contributions. At fixed entry count, converting one cold entry to hot predicts a 4.27 µs/layer saving; reducing the cold coefficient by one microsecond saves `C` µs/layer. Use actual tail-layer contributions to choose between hot-path throughput and cold-path latency work.
   - **Treat shared fusion as a secondary target.** Its measured M=32 increment is 0.54 ms, or 7.2 µs/layer. Even recovering all of that would not reach the 15% target. Likewise, eliminating the entire fitted 13.6 µs intercept would still be insufficient alone.

   Finally, do not compare the sliced and EP coefficients as direct speedup ratios: sliced counts and **whole-expert counts per EP GPU** need their actual replay mapping. Refit after GR1=2 and any dispatcher change, then evaluate the same slowest-GPU replay metric.

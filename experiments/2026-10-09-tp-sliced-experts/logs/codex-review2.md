The supplied changes look structurally sound: **v21 fixes the identified stage-lifetime bug, and v20 can safely issue weights before waiting for x2.** I do not see a new deadlock in either diff. The sweeps substantially weaken the forecast of a recoverable 10–16 µs w2 producer loss, though they do not eliminate every producer-side explanation.

This is a source review, without execution. I assume the existing `mbar_expect_tx` helper performs a **release arrive-and-expect**, and the barrier waits provide acquire semantics; their definitions are not supplied.

**Correctness**

- **v21’s destination snapshots fix the use-after-release.** At v21 lines 1185–1190, each lane saves `sd_row[s][lane/16 + 2*j]` before `empty` arrival. That matches the flush reader’s `c = (lane + 32*j) >> 4` exactly. After release, the flush uses register-held rows, scales and accumulators, plus separate scratch—not the recycled stage. This covers both R0 and R1 flushes (v21 hunks at `+1196` and `+1231`).

- **The scratch permutation is consistent.** In `flush_rows`, v21 lines 796–817, each accumulator goes to token `2*tq + (i&1)` and row `16*mb + g + 8*(i>>1)`. Across the warp, these uniquely cover all 64 rows for eight tokens. The first `__syncwarp()` publishes those stores to the reading lanes; the final one protects scratch reuse. Every vector read for a valid token was written, including odd `ntok`. Both scratch and output vector addresses are 16-byte aligned: `SCR_LD=68`, the output strides, and the tile/half offsets preserve that alignment. Hopper supports these vector reductions, with atomicity per float; whole-vector atomicity is unnecessary here. [PTX reduction specification](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-red)

- **v20’s transaction accounting prevents premature stage completion.** At v20 lines 1095–1097, the expected bytes include **weights + weight scales + all `ntok * X2_COPY` activation bytes before either TMA starts**. If both weight transfers finish during the ready spin, the activation credit remains outstanding. Other lanes cannot issue bulk copies before lane 0 establishes that credit: they rendezvous at line 1108 before copying at lines 1113–1115. PTX completes a barrier phase only when both pending arrivals and transaction count reach zero. [PTX barrier completion rules](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-phase-completion)

  The unchanged empty-stage wait and producer `__syncwarp()`—v17 lines 1022–1024—also precede these writes and transfers. Issuing weights early therefore does not overwrite an occupied stage. Do **not** change this to “expect weight bytes now, add activation bytes after ready” with an already-consumed arrival: that could complete the phase between the two operations.

- **The carried x2 scale follows the same publication chain as the activation.** The float store at v20 line 850 occurs inside `activate()`, before the retained writer-side proxy fence, thread fences, consumer rendezvous and release of `ready` (v17 lines 1175–1179). Every copying producer lane then acquires `ready` and executes `fence_proxy_async()` before its bulk copy (v20 lines 1101–1115). The consumer reads the scale after full-barrier completion and before empty arrival (v20 lines 1193–1198).

  Thus the chain is: activation/scale stores → publication → producer acquire/proxy fence → bulk copy → full-barrier observation → consumer LDS. Async-copy completion provides the destination-side proxy ordering. [PTX async-proxy rules](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-operations)

  The layout arithmetic checks out: `520` halves means a **1040-byte stride**, the float starts at byte **1024**, and the **1040-byte copy** fits the **1088-byte stage row**. Sixteen-byte source alignment is preserved. The remaining twelve copied padding bytes are in bounds and unused. All allocations and any unshown x2 readers must use the new workspace layout.

- **The register-carried xs13 is valid under the stated bounds.** The load near v20 line 1003 remains after PDL synchronization. At lines 1046–1056, all producer lanes participate in the shuffle; valid token IDs are below `T <= 32`. Inactive record slots are physically in bounds, and their results are masked before copies or metadata publication. “One round trip” is an optimization hypothesis, however: several independent field loads remain, and source code does not guarantee their scheduling.

- **No new dependency cycle is apparent.** v20 can hold a partially filled R1 stage while its producer spins, but earlier R0 stages can still drain and immediately perform count/activation. With the stated queue ordering, unfinished prerequisite R0 work is either earlier in a consumer FIFO or owned by a producer that has not progressed to R1. Completion does not require consuming the blocked R1 stage. That is the crucial distinction from v19’s deferred completion. The new scratch barriers involve only their own warp. Preserve the tier-switch reset of `last_ready` at v17 line 972.

These changes do **not** establish that the historical GR1/stealing failure is fixed. They also retain the baseline completion-count protocol. Sixty passing no-fence runs are useful evidence, but with no measured speed benefit I would keep fence removal out of this comparison.

One performance concern in v21 is visible directly: the four row snapshots execute on **every R0 chunk**, although only the final chunk flushes. They could be guarded by `kind == K_R1 || (kind == K_R0 && last)`, still before empty arrival. The flush also stages all eight token positions regardless of `ntok` and adds **17 KiB of static shared memory**. Neither vectorization nor the source-level instruction reduction guarantees a win at one or two tokens per entry.

**What the sweeps actually establish**

The **GR1 sweep is meaningful negative evidence against exposed, amortizable w2 producer overhead**. GR1 changes amortize the record reads and dependent xs2 load, as well as claims, because v17 prepares those once per group at lines 995–1015. At 38/0, GR1=2 saves nothing; GR1=4 and 6 regress. If 10–16 µs were predominantly independently removable per-group overhead, this is not the expected result.

That is not a strict upper bound: larger groups reduce scheduling granularity and the number of CTAs distributing an entry’s w2 tiles, potentially cancelling savings. The sweep also excludes cold entries. Nevertheless, I would **withdraw 10–16 µs as a performance forecast until a direct ablation supports it**. v20 remains useful because early weight issue changes overlap, something GR1 alone does not test.

The **cold-CTA sweep rejects “increase the starting cold preference” as an effective remedy**. It does not prove C2C is unimportant or saturated:

- Starting preference is not the number of CTAs actively issuing cold requests; stealing, empty-stage backpressure and ready waits change that.
- The sweep stops at 32, while the cited plain-load saturation requires at least 48 CTAs.
- Those extra cold-preferring CTAs come from the same fixed 132-CTA pool serving hot work.
- The measured 419 GB/s belongs to a different access mechanism and workload.

Likewise, 19 MB divided by 85 µs measures **whole-call average throughput**. Approximately 45 µs of transfers near 419 GB/s plus 40 µs without useful cold traffic would produce a similar average. That is an illustrative possibility, not an inferred timeline.

The current evidence instead supports a **coupled pipeline limitation**: consumer execution and flush/completion work throttle stage recycling; dependency transitions, transfer latency and imbalance intermittently starve consumers. The producer’s 20.8 µs empty wait is evidence of backpressure, not an independently removable producer cost. Its ready waits can overlap older consumer work.

The 0.72 µs isolated consume probe does not isolate “bookkeeping” as the entire gap. It omits production control flow, reductions, activation, synchronization and mixed shared-expert work. Loaded execution additionally introduces shared-memory traffic from TMA and memory-system competition with reductions. Neither sampled instruction shares nor overlapping producer/consumer waits can be added into a latency budget.

**My next three experiments, ranked**

1. **Validate and time a factorial comparison separating flush changes from producer changes.** Compare v17, v21, v17 with only the v20 producer/layout changes, and full v20. Keep fences and GR1=1 for performance attribution. Use repeated per-slice checks covering partial token groups, M=32 cold entries and stealing; separately stress the known GR1>1 failure.

   Measure end-to-end and compute-only time, compiler resource usage, and sparse per-kind timings for consumer execution/flush, empty waits, ready waits and full waits. If the producer change wins, separate early weight issue from the metadata/scale changes. **This directly tests the audit’s prediction without letting scratch overhead hide the answer.**

2. **Implement the originally planned pre-PDL shared-w13 weight fill as a separate change.** The actual supplied v20 does not implement that proposal. Statically assign the relevant S0 groups, issue their weight transfers before v17 line 934, and remove those same groups from subsequent dynamic scheduling.

   Keep the full barrier’s arrival pending until post-PDL metadata and activation-copy credits are established; do not publish a weight-only completed stage. Continue reading route-prep-owned data only after PDL. Measure route-prep duration too, since early transfers could contend with it. The plausible benefit is bounded by the roughly 4–5 µs startup overlap opportunity, and may be smaller.

3. **Measure the cold transfer ceiling using this exact TMA path and stage geometry.** Replay the actual cold weight/scale maps with four stages at 16/32/48/64 issuing CTAs. Compare a minimal consumer that safely drains completed stages with the real consume body, then add concurrent hot traffic. Measure active-transfer intervals as well as whole-run bandwidth.

   If continuous cold TMA approaches 419 GB/s, production needs better request continuity or scheduling. If even the minimal-consumer replay stays near 220 GB/s, compare the transfer mechanism with the proven plain-load path. If adding the real consumer causes the collapse, prioritize stage recycling and consumer/shared-memory contention. This distinguishes three materially different fixes that another cold-preference sweep cannot.

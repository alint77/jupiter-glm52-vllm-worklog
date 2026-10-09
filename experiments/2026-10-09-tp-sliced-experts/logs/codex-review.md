The persistent decomposition is reasonable. I would keep v17 as the baseline, but stop promoting v18/v19 until two concrete problems are fixed: **v18 reads stage metadata after releasing the stage, and v19 can make forward progress depend on a completion action that it has deferred.** The evidence supports a mixed instruction, synchronization, and transport bottleneck—not just per-unit bookkeeping.

1. **The unit shape is defensible; scheduling and completion ownership offer better near-term opportunities.**

   The actual routed shape is 128×512, with eight w13 tiles and twelve chunks per tile (lines 59–65). Both projections use the same consumer mapping (lines 1148 and 1193). This is a sensible compromise: enough independent tiles, reusable decoding code, and twelve chunks of w13 accumulation before flushing.

   I would not start with more warps. The 16-warp probe and v9 already argue against that. Two CTAs per SM would require substantial resource changes: four stages consume about 181 KiB before static shared allocations (lines 82–91), and 288 threads × 160 registers is about 46K registers per CTA. Two such CTAs cannot fit. Two stages could address shared-memory capacity, but registers would also need to fall to roughly 112 per thread, subject to allocation granularity. `setmaxnreg` can redistribute registers; it cannot solve the shared-memory limit.

   The strongest structural opportunities are:

   - **Separate runnable w2 work from unfinished dependencies.** Currently, a producer claims work and then blocks on readiness before issuing its first routed w2 weight load (lines 1007–1018, 1055–1066). A ready-entry queue could expose w2 only after activation publication, allowing other CTAs to continue useful work. The comment claiming weights issue before this wait is stale (lines 757–758).
   - **Use finer w13 groups specifically for cold work.** Splitting a twelve-chunk cold group into two six-chunk groups increases transport concurrency without changing the 128×512 unit. It adds flushes and changes accumulation order, so it needs numerical validation.
   - **Exploit exclusive w13 tile ownership if the numerical contract permits it.** One R0 group already owns every chunk of one entry/tile (lines 898, 988–990). Its four K-partition warps could reduce their already-scaled partials locally and issue stores, eliminating that tile’s global atomic reduction. Shared w13 remains a different case because its groups split K across CTAs.

   That last option is not automatically numerically equivalent. Preserve the group-scale FFMAs and final per-partial scaling in lines 864–867 and 1161; reducing before multiplying changes rounding. Local reductions also change summation order and potentially subnormal handling relative to global atomic adds. If “no numeric changes” means bitwise equivalence, these reductions and split-K changes are excluded. The source itself currently allows summation-order differences (lines 23–25).

2. **“Consumer-side overhead matters” is supported; “bookkeeping explains the gap” is not established.**

   The probe measures `consume_routed` in isolation. The full consumer additionally performs descriptor and destination loads, barrier operations, output atomics, completion synchronization, activation, and shared-expert work (lines 1128–1266). The quoted 1.3 µs/unit also includes effects outside that isolated block.

   For 38/4, there are:

   - 42 × 144 = 6,048 routed units.
   - 384 + 192 = 576 shared units.
   - About 50.2 total units per CTA on average.

   Dividing 65 µs by 50.2 gives approximately 1.3 µs, but that includes route preparation and finalization, and treats shared units as though they had the probe’s cost. It is not a measurement of routed-unit execution time.

   Several other explanations fit the observations:

   - **Low latency tolerance.** Eight consumer warps provide only two warps per scheduler on average. Decode, MMA-result dependencies, FFMAs, address arithmetic, and synchronization can leave few eligible instructions despite unused tensor capacity.
   - **Shared-memory contention with TMA.** The isolated probe has LDS traffic without concurrent stage fills. Real execution adds asynchronous shared-memory writes and activation copies.
   - **L2 and atomic contention.** Different experts and four K-partition warps update overlapping output locations (lines 1197–1205). This costs more than its byte volume suggests.
   - **Synchronization charged outside the full-barrier bucket.** The count fences, consumer barriers, activation, and atomic-result dependency all contribute to apparent consumer time (lines 1165–1179).
   - **Producer starvation and queue effects.** Empty-stage blocking demonstrates backpressure during some intervals; simultaneous full-stage waits demonstrate starvation during others. Neither establishes one uniform bottleneck.

   The additional roughly 12 µs outside full-barrier waiting could therefore include slower instruction execution, other synchronization, and memory-system contention. Also, end-to-end loaded versus compute-only time differs by **25.5 µs**, not 12 µs.

   Treat the SASS percentages as instruction-distribution evidence, not additive time savings. In particular, “MMA blocks” include decoding and scale application (lines 847–867); 51% there does not mean 51% tensor execution.

   I would build a matched progression: consume alone → real ring and descriptors → real flushes → completion/activation → real loads. Use initialized representative data, identical work assignments, and matched clocks. Removing producer loads entirely (lines 160–170, 211–217) changes scheduling and contention, so compute-only is an informative counterfactual, not a directly subtractable component.

3. **Cold concurrency is insufficient early in the schedule, and the quoted transport roofline needs careful byte accounting.**

   Sixteen cold-preferring CTAs are a weak starting point against a measurement requiring at least 48 plain-load CTAs. However, merely setting `TD_COLD_CTAS=48` is insufficient.

   With four cold entries, the current cold w13 queue contains only **4 × 8 = 32 groups** (lines 63, 892–898). Thus there cannot be 48 CTAs independently executing cold w13 groups. One-ahead reservations can concentrate those groups into still fewer producers (lines 977–980). Additional CTAs may claim w2 and wait rather than generate C2C traffic.

   Splitting cold w13 into six-chunk groups creates 64 groups for this case. Pair that experiment with less aggressive reservation on short cold queues and scheduling that selects ready w2 work.

   Also, the plain-load transport result does not establish the concurrency requirement or throughput of this exact 3D-TMA path. Measure the actual tensor maps, activation copies, barriers, and mixed HBM/C2C traffic with a lightweight consumer.

   The supplied dimensions imply **5.308 MB per routed entry including scales**, rather than 4.719 MB of weights alone:

   \[
   144(32768+4096)=5{,}308{,}416\ \text{bytes}.
   \]

   The shared expert contributes another 18.874 MB of bf16 weights. Assuming cold scales reside in Grace too, the weight-and-scale bounds are:

   | Entries hot/cold | HBM bytes, including shared | C2C bytes | HBM time | C2C time |
   |---|---:|---:|---:|---:|
   | 38/0 | 220.59 MB | 0 | 61.3 µs | 0 |
   | 38/4 | 220.59 MB | 21.23 MB | 61.3 µs | 50.7 µs |
   | 110/12 | 602.80 MB | 63.70 MB | 167.4 µs | 152.0 µs |

   These follow from lines 55–79 and exclude activation/workspace traffic. If cold scales are actually in HBM, move their bytes accordingly.

   Consequently, “19 MB in 85 µs” appears to count only weights. Including scales would give about **250 GB/s**, still below the needed rate but materially above 220 GB/s.

   Keeping both interfaces continuously pegged is not necessary for the target. At 38/4, cold traffic needs about **347 GB/s averaged over the 61.3 µs HBM interval**. Peak C2C service could finish earlier, after which CTAs should help hot work. At 110/12, the corresponding requirement is about 380 GB/s.

   Finally, `max(hot/BW, cold/BW)` is an optimistic streaming bound. It assumes independently sustainable bandwidths despite shared SM, TMA, and L2 resources. For current v17, the initial PDL wait precedes all weight loads (line 934), so route preparation and finalization also sit outside that streaming interval. A 61 µs streaming bound is not presently a 61 µs end-to-end bound.

4. **The v17 handoff is defensible; the v18/v19 changes introduce identifiable correctness hazards.**

   **v18 has a stage-metadata use-after-release.** In v17, destinations and scales are copied into registers before consuming/releasing the stage (lines 1142–1145, 1150 and 1195). In the diff, destinations instead get read inside `flush_rows()`:

   ```cpp
   const int row = static_cast<int>(lds_u32(rows_u + c * 4));
   ```

   But callers still execute `mbar_arrive_a(empty_s)` before calling `flush_rows()`—see the diff hunks anchored at original lines 1139 and 1194.

   Once all eight warps have arrived, the producer can reuse that stage and overwrite `sd_row` (lines 1022–1029 and 1045–1047). A consumer can then send a correct accumulated value to the next occupant’s token or route. Correct weight checksums would not detect this.

   **First fix:** move stage release after `flush_rows()`. Once correctness is restored, recover overlap by snapshotting every needed destination into registers before release. The invariant is: *a warp’s empty arrival follows its final read of every object owned by that stage.*

   The scratch transpose otherwise appears consistent with v17’s row mapping. Its 68-float stride preserves 16-byte alignment. However, vector reds still perform four elementwise atomic additions; they primarily reduce instruction/address overhead, not the number of scalar atomic updates. The staging code also writes all eight token positions even when only one or two are valid, adding 16 KiB of shared stores across the eight warps per flush.

   **v19 can deadlock on deferred completion.** Its settlement occurs after another unit has been consumed, while the loop first waits for that unit’s full barrier (diff hunks anchored at original lines 1123, 1130 and 1268).

   A valid failure sequence is:

   1. The consumer finishes an entry’s last w13 contribution but leaves its count or activation pending.
   2. The producer reaches w2 for that entry and waits at line 1009.
   3. The consumer drains available stages, then waits for another full stage.
   4. The pending completion cannot execute because it sits after that wait.

   Draining pending work at `K_END` does not help when the producer cannot reach `K_END`. Deferred completion must progress independently of future stage availability—for example, settle pending actions before blocking when no stage is ready. Ordinary producer lookahead already covers useful overlap; this change requires an explicit forward-progress argument.

   **For v17’s memory ordering:** the shown path has the ingredients of a correct GPU-local publication protocol:

   - Each contributor orders its reds, and the consumer barrier gathers the participating threads (lines 1165–1167).
   - The leader fences and updates the completion counter (lines 1168–1172).
   - The last CTA orders its reads before activation (lines 1173–1175).
   - Activation writers perform proxy and device ordering, rendezvous, and publish readiness with release semantics (lines 1176–1179).
   - Every copying lane acquires readiness and executes the proxy fence before copying x2 (lines 1006–1015, 1064–1066).

   I do not see an obvious missing acquire in that sequence. GPU scope is appropriate when these mutable workspace objects are accessed only by this GPU; reading immutable weights from Grace does not itself require system-scope completion counters.

   There is probably simplification available, but do it independently of v19. An explicit GPU-scope acquire-release completion RMW, with consumer rendezvous before and after it, makes the publication chain easier to audit. A per-thread fence is not automatically necessary merely because another thread increments the counter: the CTA barrier participates in the ordering proof. Conversely, a leader fence alone, without that rendezvous, is insufficient. Establish the exact PTX ordering of the reds and barriers before removing fences.

   The snippets do not show route-prep publication or workspace-reuse ordering, so those remain outside this review’s correctness proof.

   **The older GR1 stealing failure is not explained by an obvious index bug here.** Group indexing accounts for GR1 (lines 992–993), readiness caching is reset on a tier switch (line 972), and R1 accumulators are reset every unit (line 1205). Do not attribute it to missing fences solely from its symptoms.

   Capture the failing unit’s scales, activation bytes, destination rows, flush factors, and `(epoch, tier, entry, tile, stage, generation)` before stage release. The existing checksum covers only weights in one half (lines 1185–1190). First-half-only corruption makes warp-group skew, stage lifetime, and row mapping particularly useful suspects, but does not prove one.

5. **My next steps, in order, would be:**

   1. Identify the exact failing binary and macro configuration. Fix v18’s metadata lifetime; revert or redesign v19’s deferred settlement. Validate these independently against v17.
   2. Stress per-slice correctness with forced stealing, GR1 variations, ring wraparound, and artificial producer/consumer delays. Inspect intermediate y13/x2 values and destinations, not just final relative error.
   3. Correct byte accounting and match clocks. Run the incremental consumer-cost experiment and the actual mixed-tier TMA transport probe.
   4. Sweep cold allocation together with cold group size and reservation policy. Then introduce ready-work scheduling if dependency stalls remain significant.
   5. Try v20 as a bounded startup optimization. Immutable shared weights can be prefetched before line 934, but `T`, counters, and prepared activations cannot be read early. Keep stages unavailable until activation data and metadata are valid, and remove statically assigned work from the dynamic queue to avoid duplicates.
   6. After profiling the corrected implementation, simplify completion fences and consider local K reductions within the accepted numerical contract. Revisit occupancy and unit shape only if those measurements justify the complexity.

   For v20 specifically, preserving twelve-chunk shared groups exposes only 32 initial groups (lines 76, 880–881, 896). Four-stage prefilling across those owners covers roughly 4 MiB of weights. Prefilling across all 132 CTAs requires a different static assignment, potentially increasing shared-expert partial flushes. Measure the overlap achieved; do not assume the whole reported five-microsecond fill disappears.

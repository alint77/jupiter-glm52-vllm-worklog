**(a) I don’t see a GR1-specific race in the current code, including when hot CTAs steal cold work.** The ownership, accumulator, and stage-lifetime rules now support multiple w2 units per group.

The relevant checks are:

- **Each w2 unit has one owner.** A claimed R1 group maps to `ei = gr.x / (TILES1 / GR1)` and units `t0 ... t0 + GR1 - 1`. Those ranges partition the entry’s units. The divisibility assertion prevents a group from crossing an entry boundary.
- **Stealing preserves the tier identity.** The producer selects `tr` and the entry record using `q`, writes `q` into every stage header, and resets `last_ready` when changing tiers. Consumers use the stage’s recorded tier. They can therefore still be consuming hot stages while the producer starts issuing cold stages.
- **R1 flushes every unit.** Lines 1261–1263 are unconditional within the R1 path. They do not depend on the group’s `last` bit. `flush_rows()` resets every accumulator element, so GR1 does not accidentally accumulate adjacent output tiles together.
- **Everything needed after releasing the stage is already private.** Before the empty arrival, consumers have loaded the descriptor, destination rows, route weights, activation scales, and MMA operands. The subsequent flush uses registers and per-warp scratch. In particular, `rows4` is copied before release, and the activation scale comes from the copied x2 row.
- **The producer cannot overwrite a stage prematurely.** Empty completion requires all eight consumer-warp arrivals. Each warp synchronizes after its stage reads and before its arrival.

The readiness path also supports GR1 > 1. Every copying lane acquires the entry’s epoch and executes the proxy fence before its first x2 copy. Later units can reuse that acquisition because the entry’s x2 data remains immutable for the invocation. The full barrier includes the activation bytes, so consumers cannot proceed merely because the weights have arrived. They do not need another global ready acquisition.

For the no-per-thread-flush-fence version, the important distinction is that the leader’s `__threadfence()` is **preceded by the collective consumer barrier**. The intended publication chain is:

`all consumers’ reductions → consumer_sync → leader fence/count → last-CTA synchronization → activate → per-thread proxy/device fences → consumer_sync → ready release`

Thus, a leader fence standing alone would be an inadequate explanation; the collective barrier and completion/publication sequence are essential. With the normal CUDA/PTX contracts for these helpers, I don’t see a reason GR1 requires restoring `TD_FLUSH_FENCE`.

This assumes the usual surrounding invariants: counters are initialized correctly, `UNITS0` matches the total contributed chunks, and invocations sharing the workspace do not overlap. The 120 successful repetitions support the source-level assessment; they are not its sole basis.

For targeted stress verification, I would force delays around stage release and activation publication, use matching numeric entry IDs in both tiers, vary `ntok`, and change activation scales between epochs. Record `(q, ei, t)` ownership to check exactly-once issue. Those cases directly exercise the former classes of stealing, stale-row, and stale-scale failures.

**(b) The measurements indicate roughly a 15% effective throughput deficit in the loads-only path, with evidence of dependency stalls and uneven progress. They do not identify 10–30 µs of pure TMA overhead.**

Using your stated fixed overhead:

| Case | HBM floor + 6.3 µs | Loads-only | Excess |
|---|---:|---:|---:|
| 38/4 | 67.6 | 78.1 | 10.5 |
| 110/12 | 173.7 | 203.6 | 29.9 |

After subtracting the fixed overhead, the measured windows achieve about **85% of the floor-implied throughput in both cases**. The approximately proportional deficit is consistent with recurring issue/retirement gaps, although a workload-dependent tail could also contribute.

There are four important qualifications.

**First, this ablation still executes substantial work.** `TD_ABL_NOMMA` removes routed math, and `TD_ABL_NOFLUSH` removes routed flushes. They do not remove:

- Shared-expert MMA and atomic flushes.
- Routed and shared activation, including reads, nonlinear operations, stores, and y13 clearing.
- Completion counters, collective barriers, and readiness publication.
- Entry-record loads, per-stage metadata, activation copies, and ring management.

Consequently, the pure TMA probe and this ablation have different service demands even when their weight boxes match.

**Second, readiness blocks the entire producer, despite weights-first issue.** For an unready R1 entry, the producer:

1. Reserves a stage and includes activation bytes in its expected transaction count.
2. Issues that stage’s weights and scales.
3. Spins until the entry becomes ready.
4. Only then issues activation copies and advances.

It cannot issue subsequent stages during step 3. The weights-first strategy overlaps readiness with **one unit’s** weight transfer; it does not keep a multi-stage weight pipeline running throughout a long ready wait.

Several CTAs can claim w2 groups for the same unready entry. One delayed w13 contributor can therefore stall many producers simultaneously.

**Third, full-wait and empty-wait totals are not mutually exclusive bottleneck measurements.**

A consumer waiting on full can be waiting for:

- TMA service,
- activation bytes withheld by a ready wait,
- producer scheduling or metadata work,
- or stage reuse delayed by another consumer warp.

The producer waits for **all eight** empty arrivals. Warp 0 can have released an old stage and advanced to waiting for its next generation while another warp still prevents that stage’s reuse. Producer-empty and warp-0-full waits can therefore overlap.

There are also alternating periods: consumers perform an activation or group epilogue while the producer fills the ring; later they drain it and wait for the producer. The timers include wait-instruction/check overhead, and percentile summaries need not describe the same CTA.

So **50.7 + 20.8 µs is not an additive stall budget**, and 50.7 µs of full waits does not establish that memory was continuously saturated.

**Fourth, the w13 tail matters, but its trace meaning is specific.** `td_c0` records completion of the CTA’s last local R0 group, including activation when that CTA wins the completion count. Its distribution is not an entry-ready distribution.

Nevertheless, local R0 completions extending to 67.8 µs, near the 72.2 µs median exit, together with ready waits reaching 22 µs, strongly suggest a dependency tail. Some producers encounter w2 whose prerequisites remain unfinished while much of the grid has already passed its main w13 work.

The current queue also delays eligibility: early entries’ w2 cannot be claimed until all routed w13 groups in that tier have been **claimed**, including prefetched but potentially unissued groups. This is not a global completion barrier, but it restricts opportunities to overlap early w2 with late w13.

The 3.76 TB/s probe demonstrates that the boxes and hardware can sustain the target under its issue pattern. It does not establish that the real kernel maintains the same number of useful outstanding transactions throughout execution.

Finally, the mixed-tier reference should include the cold traffic:

\[
T_{\mathrm{stream,min}}\ \ge\
\max\left(B_{\mathrm{HBM}}/BW_{\mathrm{HBM}},
          B_{\mathrm{cold}}/BW_{\mathrm{C2C}}\right).
\]

The 363–382 GB/s result is useful here. It is an observed cold-only ceiling with 132 cold CTAs, not a guarantee that the smaller cold allocation in a mixed run reaches that rate. Any uncovered cold-service requirement consumes part of the apparent HBM-floor gap.

**Why R0S hurts:** it preserves the number of weight-transfer units while increasing the number of group epilogues.

Each split adds another accumulator flush, completion-count update, collective synchronization sequence, claim, and entry-record handling. With R0S = 4, the same tile’s contributions are flushed four times instead of once. Readiness still requires every contribution to the entire entry.

Adjacent queue groups also do not guarantee assignment to different CTAs; preclaiming can retain adjacent pieces on one producer. Splitting therefore pays a definite overhead for an uncertain reduction in the critical path. Your monotonic regressions are strong evidence to keep **R0S = 1**.

The full-kernel GR1 results similarly make batching a secondary knob. Most improvements are small and the best setting changes by case; 96/8 at GR1 = 3 is about 3% faster, but there is no consistent trend. GR1 reduces claims and record loads while retaining each unit’s stage protocol, copies, math, and flush.

**(c) I would rank the changes below by likely performance value. The gain descriptions are hypotheses, not additive forecasts. The quoted 10.5/29.9 µs gaps are optimistic total budgets, and some of that budget is retained work.**

1. **Ready-aware w2 dispatch, combined with a bounded window of w13 work.**

   This has the largest plausible benefit: several microseconds in the smaller case and potentially low tens in the larger case **if dependency stalls lie on the critical path**. The current trace does not justify a more precise estimate, particularly for 110/12.

   Separate unclaimed w13 work from eligible w2 work. When an entry publishes readiness, make its w2 groups eligible. Producers choose ready w2 or continue issuing w13 for later entries.

   This directly implements “w2 of early entries alongside w13 of late ones,” with eligibility determined by actual completion. Maintain enough w13 work in flight to sustain bandwidth and prevent w2 from indefinitely delaying remaining prerequisites.

   Two reasonable implementations are:

   - A ready-entry bitmap plus an atomic per-entry w2 cursor.
   - A queue publishing multiple independently claimable w2 group descriptors per ready entry.

   **Do not give one CTA exclusive ownership of an entire ready entry’s w2.** That sacrifices distribution and creates a new tail.

   There are three correctness requirements for this redesign:

   - Select ready work before committing a normal FIFO stage to it; otherwise an incomplete stage can still block consumption.
   - Keep each R0 group’s chunks contiguous. The current consumer has only one routed accumulator.
   - Queue emptiness cannot mean termination while entries can still become ready. A queue reservation also cannot be treated as a published descriptor; use committed-slot/sequence semantics if implementing a concurrent FIFO.

   **Verify:** first establish the opportunity with a timing build using frozen, precomputed activation buffers and bypassed readiness waits. Ensure `activate()` does not concurrently overwrite those source buffers. If eliminating readiness barely improves elapsed time, reduce this redesign’s priority. In the real implementation, measure entry-ready-to-first-w2 latency, simultaneous ready-blocked producers, issue gaps, and the final tail.

2. **Pipeline the next entry record, then evaluate two-group lookahead.**

   Today you preclaim one group, but you do not fetch its entry record until the current group has finished issuing. The next group’s record latency can therefore remain exposed, especially for GR1 = 1.

   Start by buffering **one next descriptor and its immutable entry fields** while issuing the current group. Then test two ahead. Simply moving the next claim’s result use earlier can expose its atomic latency; the objective is actual overlap of record fetches with current work.

   Expected benefit is modest: zero to a few microseconds in the smaller case, potentially several in the larger case. The GR1 sweep and mixed no-prefetch results argue against assuming this alone explains the gap.

   Avoid accumulating many reserved, unissued R0 groups. That can make the readiness tail worse. All valid prefetched claims must also be drained before tier switching or termination.

   **Verify:** record the interval from the last issue of one group to the first issue of the next, separately for each kind. Confirm that the implementation hides record latency without spills or a longer w13 tail.

3. **Increase usable stage depth, after or alongside fixing issue gaps.**

   This is cheap enough to test early, despite its lower expected payoff. Your existing stage sweep does not show a decisive improvement, so I would expect zero to a few microseconds unless another change first enables more sustained issue.

   More stages help when transfer latency or short consumer epilogues exhaust the available buffering. They do little when the producer stops issuing at the first unready entry. A larger ring cannot fill itself during that spin.

   Test feasible 3- and 4-stage configurations while checking resource residency and spills.

   **Verify:** compare outstanding stages and inter-issue gaps. A useful stage increase should improve elapsed time and sustained outstanding work; merely shifting time between full and empty waits is insufficient.

4. **Make hot/cold allocation elastic at group boundaries.**

   This addresses mixed-tier imbalance, so it cannot explain or fix the all-hot gap. Expected benefit depends on whether the existing cold allocation saturates C2C and whether one tier finishes significantly earlier.

   Preserve a baseline allocation sufficient to feed cold traffic, then allow additional CTAs to choose eligible work from either tier before their original queue is exhausted. Use remaining bytes, measured service rates, and readiness to guide selection.

   Blind alternation is risky: a slow cold stage at the head of the same FIFO can delay consumption of later hot stages.

   With repeated tier switching, readiness caching must explicitly include `(q, ei)` or reset on **every** tier change. The existing one-time reset is sufficient only for the current switching pattern.

   **Verify:** measure mixed-run C2C throughput, each tier’s remaining work over time, and idle hot capacity. If C2C already sustains the measured ceiling throughout its critical interval, assigning it more CTAs has little bandwidth upside.

5. **Treat static queue interleaving as a limited prototype, not the final scheduling policy.**

   A static order such as “w13 for a window of entries, then w2 for earlier entries” may improve overlap. It may also put many producers onto an unready entry while useful late-entry w13 remains unclaimed.

   Its expected gain is therefore uncertain and can be negative. The useful version is readiness-aware: choose early-entry w2 when eligible, otherwise advance late-entry w13. That becomes the first design above.

For measurement, I would add one shared time axis showing **issued bytes, ready-blocked producers, empty-blocked producers, and entry-ready events**. Sample it lightly and use uninstrumented runs for final timings. That distinguishes insufficient issue during the main streaming interval from a mostly healthy stream followed by a dependency tail.

Keep **R0S = 1**, use GR1 as a minor per-shape tuning choice, and prioritize determining how much elapsed time a producer that always has eligible work can actually recover.

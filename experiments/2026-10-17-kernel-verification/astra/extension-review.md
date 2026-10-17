GRID=1 is a good first milestone. **S2–S4 need revision before their results can be called PTX-sound.** The routed synchronization looks defensible; the shared-expert path depends on stronger fence semantics than the proposed acquire/release approximation captures.

I found no definite intra-launch synchronization bug in the shown handoffs. I did find several ways the proposed verifier could accept an impossible execution, miss an ordering obligation, or misclassify a failure. Line references below use your excerpts. No machine commands or tests were run.

1. **Critical — S2’s hb-versus-modification-order check is necessary but insufficient.**

   A pinned modification order can support a meaningful conditional verdict, but it does not specify a complete execution. You also need reads-from choices, event existence, returned values, control-flow consistency, and any relevant SC-fence order.

   A simple counterexample to the proposed check is two successive acquire probes of one flag, returning `epoch` and then `epoch-1`, with only one publication. There need not be any RMW-order contradiction, yet that observation sequence is forbidden. PTX consistency includes coherence, RMW atomicity, per-location consistency, causality, Fence-SC, and no-thin-air requirements. [PTX memory-model axioms](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#axioms)

   Your particular feedback loop is significant:

   ```text
   claim returns
     → group ownership and execution
     → completion-counter order
     → identity of the last finisher
     → ready publication and successful probes
     → producer progress and later claims
   ```

   Reaching a numerical fixpoint does not establish that this entire execution is feasible. Iteration can also oscillate, change event membership, or stall before discovering the writer needed to unblock the current candidate.

   There is a useful simplification: **the shown completion counters have constant increments per location.** `done13` receives `R0CH` from routed groups; `done_s` receives `SCH0` from shared groups—see L1018–1020, L1446, and L1507. The variable name `nch` does not create a variable-increment problem here. Given a valid ordering of the actual completion events, their counter values are straightforward.

   **Recommendation:** choose one of two explicit products:

   - An executable replay: select among enabled actions, execute RMWs against shared global state, and record the resulting order and reads-from choices.
   - An axiomatic checker: constrain event existence, values, reads-from, coherence, synchronization, and SC-fence order jointly.

   The first is substantially cheaper. It explores a narrower set of executions, but supplies an execution witness. Crucially, **the interpreter’s scheduling order must not become an hb edge** between unrelated threads.

   For either product, distinguish `feasible-and-checked`, `infeasible-candidate`, and `unknown/incomplete`. A fixpoint iteration limit belongs in the last category.

2. **High — Use PTX release/acquire patterns and observation order; retain full SC semantics for `done_s`.**

   PTX’s relevant mechanism is scoped release/acquire patterns connected through observation order, which can traverse RMWs. Importing a C++ “release sequence” implementation wholesale is not a sufficient specification. [PTX release/acquire patterns and observation order](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#release-and-acquire-patterns)

   Applied to this kernel:

   - The `.gpu` operations cover the participating CTAs on one device.
   - Every earlier `done13` release RMW can contribute to the last RMW’s acquire through the counter’s RMW chain.
   - `fence; atom.relaxed` can publish preceding work. A receiving `atom.relaxed; fence` can acquire it through the appropriate observation chain.
   - An intermediate relaxed RMW can carry an earlier publication through that chain; it does not automatically publish its own preceding payload.
   - A `red` must not be treated as an acquiring read merely because it performs a reduction. [PTX reduction exception](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#reductions-do-not-form-acquire-patterns)

   Your `membar.gl = fence.sc.gpu` mapping is correct on sm_90. But implementing that as *only* acquire-plus-release loses additional synchronization from Fence-SC order. CUDA likewise specifies `__threadfence()` as a device-scoped sequentially consistent fence. [PTX fence semantics](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-membar), [CUDA fence functions](https://docs.nvidia.com/cuda/archive/12.9.2/cuda-c-programming-guide/index.html#memory-fence-functions)

   **This distinction matters at L1503–1512.** Thread 0 reads `done_s` at L1507. The other consumer threads do not. Their fences at L1511 therefore cannot simply be paired with thread 0’s counter read as same-thread acquire patterns. Also, L1509 precedes thread 0’s acquiring fence at L1511, so that barrier cannot distribute that later acquire.

   Nevertheless, I believe the shown sequence is ordered under **full PTX SC-fence semantics**. Here is the argument:

   - Let `F_i` be an earlier contributor CTA’s L1506 fence, and `C_i` its counter RMW.
   - Let `L` be the winning counter RMW.
   - Let `G_t` be any winning consumer thread’s L1511 fence.
   - The L1509 barrier orders `L` before `G_t`.
   - If `G_t` preceded `F_i` in SC-fence order, the resulting causality path would order `L` before `C_i`, contradicting their counter coherence order.
   - Thus `F_i` precedes `G_t`; the contributor’s accumulated data reaches that thread’s activation reads.

   That is a stronger argument than the proposed fence-pattern rule implements.

   **Recommendation:** make this a dedicated litmus test before extending SentryWard. Either model SC fences faithfully or explicitly leave this path unsupported. An optional production simplification is an acq_rel `done_s` RMW **before** L1509, allowing that barrier to distribute the acquire. Do not make that strengthening only in the verification build and claim to have checked the original protocol.

3. **High — S3 conflates observing a value, acquiring its payload, and proving progress.**

   `AwaitValue` must resolve to a write supplying the requested value—not necessarily a **release** write.

   For example, replacing L1111 with `st.relaxed.gpu` can still let L1307 terminate. What disappears is payload publication. Reporting `await_no_writer` would misdiagnose precisely one of your intended mutations.

   Similarly:

   - A successful acquire load needs a specific reads-from justification.
   - A false probe needs a legal source value too.
   - “The writer is not hb-after the probe” is insufficient for probe feasibility.
   - Finding a candidate writer does not establish that execution can reach it.
   - Multiple matching writers require disambiguation or a separate uniqueness-invariant failure; they are not inherently a polling deadlock.

   A cycle obtained by inserting an assumed successful wait edge establishes an impossible proposed completion. To establish deadlock, inspect an executable blocked prefix and its dependencies. Otherwise an inconsistent pinned choice can be mislabeled as a kernel deadlock.

   **Recommendation:** represent the successful flag read separately from the synchronization it establishes. Normalize only after establishing that failed iterations have no relevant effects. Keep termination conditional on an explicit progress/fairness assumption; normalization alone does not prove it.

   The one-shot probe deserves its own consistency tests. In particular, “all probes false” is not necessarily feasible once an earlier observation or handoff orders the publication before a later probe.

4. **High — The consuming-side global proxy fence is appropriate; S4’s blanket WAR exemption is not justified.**

   The intended routed chain is:

   ```text
   finisher writes x2
     → warp synchronization
     → release ready
     → scheduler acquire
     → sq release-arrive / acquire-wait
     → producer warp synchronization
     → producer global proxy fence
     → async read of x2
   ```

   The CTA-scoped `sq` handoff does not discard the scheduler’s acquired global-memory ordering. Its endpoints are in the same CTA. The `.gpu` flag operations supply the cross-CTA link.

   **Placing the proxy fence in the consuming producer is sufficient for this chain.** It also places the fence in the CTA performing the non-generic access. A fence only in a remote writer CTA is not a general substitute. NVIDIA’s mixed-proxy analysis explicitly discusses this placement restriction. [Mixed-proxy model, §5.4](https://d1qx31qr3h6wln.cloudfront.net/publications/ISCA_2022_MixedProxy.pdf)

   Consequently:

   - L1309 and L1327 are correctly placed after acquisition and before copying.
   - L1513 should not be used to justify removing the remote producers’ L1327 fences.
   - Requiring a fence in **every issuing thread** is a conservative sufficient rule, not the most general rule: a correctly synchronized fence elsewhere in that CTA may suffice.
   - `.global` identifies the fenced state space; it is not a `.gpu` scope qualifier.

   For stage reuse, the proposed blanket rule—

   > generic reads followed by async writes need only the mbarrier

   —should not enter a sound checker without a separately established exception. The relevant path here is generic shared-memory consumption followed by a TMA overwrite, and L1268 supplies its explicit proxy bridge. NVIDIA’s guidance for pipelined TMA likewise calls for synchronization plus a proxy fence before reuse. [Nsight Compute TMA guidance](https://docs.nvidia.com/nsight-compute/ComputeTriage/#symptom-shared-memory-fills-go-through-the-register-file)

   **Recommendation:** retain L1268 and check the WAR path. CUTLASS examples involving TMA and WGMMA may involve two async-proxy accesses, which does not establish this generic-to-async case.

   S4 should also distinguish:

   - explicit generic-to-async ordering;
   - the implicit async-to-generic bridge associated with observed bulk-copy completion;
   - issue versus completion;
   - generic memory publication versus mbarrier initialization.

   In particular, a bulk operation’s `complete_tx` must not publish arbitrary earlier operations from its issuing thread. [PTX asynchronous-operation ordering](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#asynchronous-operations)

5. **Medium — The main kernel handoffs look correct, but their supporting invariants must be checked explicitly.**

   My assessment of the requested paths is:

   | Path | Assessment and relevant lines |
   |---|---|
   | Routed reductions → finisher | `flush_rows` reductions are gathered by warp synchronization, followed by each consumer warp’s release arrival at `hq_full` (L804–807, L1371–1372). The finisher waits before its RMW (L1083–1099). This supplies the local publication link. |
   | Last RMW → activation | The counter acquire is distributed through L1107 before every lane’s y13 reads. The shuffle at L1105 should not be credited as the memory barrier. |
   | Activation → ready | L1110 gathers all lanes’ x2 writes before lane 0 publishes at L1111. This is essential. |
   | Ready → zeroing | Publishing ready before `zero_routes` is not intrinsically wrong: w2 reads x2, not y13. L1110 also orders activation reads before subsequent zeroing. Safety requires all y13 contributors to have completed and route ownership to be unique. |
   | HQ generations | Every consumer warp waits on the reused slot’s `hq_empty` at L1368. The finisher releases it only after reading the record, L1085–1087. That is the key reuse handshake. |
   | Ready probe handoff | L1182 → L1212 → L1245–1246 supplies the acquire relay. The producer still executes its proxy fence at L1309. |
   | Stage reuse | Metadata is captured before release at L1432–1434 and L1450–1452. The producer waits before refilling at L1265–1268. |
   | `K_END` | Omitting `sq_empty` acknowledgement for the terminal entry is consistent with the scheduler already exiting. Omitting terminal stage `empty` is consistent with no subsequent refill. The final HQ record waits for all consumer warps. |
   | Stealing | With static queues, one steal is sufficient for claiming available work: exhaustion means prior tickets have already been allocated. L1141 also consumes an extra ticket on exhaustion; the claim model must include that discarded ticket. |

   For reductions, the atomic-versus-atomic race exemption is appropriate for these matching device-scoped scalar accesses. **Model vector reductions per element**, and retain conflicts with ordinary activation loads and zeroing stores. [PTX atomic/reduction rules](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-red)

   The remaining progress obligation is more important than the nominal GRID size: **can every claimed w13 group progress to its reduction and completion notification without depending on a blocked w2 producer?** The heavy-group handshake at L1213–1215 and acknowledgement at L1341 appear designed to ensure this. Prove it in a small scheduler model, including stealing with outstanding work.

   Also, the HQ reuse handshake already directly prevents premature reuse of an HQ phase. Do not rely solely on the distance assertion at L1010, especially for terminal handoffs.

6. **Medium — S0 and S1 need tighter contracts.**

   Two concrete corrections:

   - **`INTER` cannot currently shrink independently.** L76–83 force the routed stage and activation row dimensions to match the fixed `CK0=512` layout. Keep `INTER=512` initially.
   - **`TD_COLD_CTAS=16` defeats the intended small-grid split.** With both tiers populated, L1076–1077 place every CTA in GRID=1–3 on the cold tier initially. For mixed-tier GRID=2–3 cases, explicitly choose a split such as one cold CTA.

   `HIDDEN=3072` satisfies the shown routed and shared chunk divisibility requirements. “CHUNKS0 ≥ 5” alone does not capture all those requirements.

   S1 should specify more than values:

   - exact verification-build layout and object extents;
   - valid tensor-map source objects and addressing;
   - route-index bounds and uniqueness;
   - `ntok`, counts, token limits, and queue membership;
   - initial payload visibility to both relevant proxies;
   - exclusive workspace ownership during the checked invocation.

   A dump from a differently sized build cannot be applied by byte offset without conversion. Synthetic contract-checked fixtures are likely cheaper initially; GPU dumps can later validate their correspondence.

   Finally, the GRID=1 statement is true for the shown **control counters**, not every RMW location. Multiple consumer warps still reduce into overlapping y13/y locations. Treating those values as opaque can avoid choosing their floating-point order, but does not make their accesses single-threaded.

7. **Medium — Race freedom does not cover several proposed mutation tests or the most important remaining properties.**

   `zero_routes` before activation can be a completely ordered, race-free computation of the wrong answer. A dropped contribution or incorrect `nch` can likewise break the algorithm without introducing a race.

   Add protocol assertions alongside race checking:

   ```text
   each valid group is processed exactly once
   each completion notification follows its group's reductions
   completion totals match the required groups
   publication occurs exactly once, after complete accumulation
   activation precedes zeroing
   x2 remains unchanged while its consumers may read it
   every claimed group completes before termination
   ```

   Other material omissions from “What this still won’t cover” are:

   - **Progress and starvation:** successful normalized waits do not prove termination.
   - **Abstraction soundness:** opaque arithmetic must not affect addresses, predicates, barrier counts, or synchronization participation. Never silently substitute zero.
   - **Existing ordering assumptions:** audit collective handling. `shfl.sync`, `mma.sync`, and memory barriers must not all acquire identical memory-ordering effects.
   - **Addressing and initialization:** bounds, aliases, alignment, swizzles, padding, and inactive-token reads are separate from race freedom.
   - **Compiler boundary:** your stated PTX→SASS gap should also include CUDA/C++→PTX lowering and inline-assembly contracts.
   - **External interference:** concurrent invocations sharing a workspace are a separate problem from sequential epoch reuse.

   Treat the initialized image and no-op PDL instructions as an explicit external precondition. They do not verify that the actual preceding kernel established that state and visibility.

I recommend this sequence:

| Step | Deliverable | Rough effort |
|---|---|---:|
| 1 | Hand-extracted litmus tests for routed counters, shared SC-fence collection, scheduler acquire relay, and proxy placement/reuse | 2–5 engineer-days |
| 2 | Corrected S0, validated S1 fixtures, and necessary S5 interpreter support | 3–7 days |
| 3 | GRID=1 with executable control-counter semantics, faithful flag reads, early proxy checks, and protocol assertions | 5–10 days |
| 4 | Small group-level scheduler model covering ownership, stealing, outstanding work, and termination; GRID=2–3 replay cases | 3–7 days |
| 5 | Full symbolic/pinned PTX execution-consistency machinery, only if broader coverage warrants it | Several weeks; approximately 3–6+ |

These are implementation estimates for someone familiar with SentryWard, not estimates for a generally complete PTX verifier.

**Yes: an extracted cross-CTA litmus model is the better first investment.** Use SentryWard for the actual instruction-level ring and handoff accesses, and use the extracted model to settle the cross-CTA ordering rules before encoding them.

Dartagnan supports a PTX target, but the selected memory model still needs auditing for the features you use. NVIDIA’s mixed-proxy artifact is useful reference material, although it explicitly describes itself as a research prototype with limited proxy coverage. Neither should be assumed to model Hopper mbarriers and TMA automatically. [Dartagnan documentation](https://github.com/hernanponcedeleon/Dat3M), [NVIDIA mixed-proxy artifact](https://github.com/NVlabs/mixedproxy)

For the first useful milestone, include **GRID=1, both ready paths, HQ/SQ wraparound, terminal handoffs, and explicit proxy checks**. Report its result as:

> For this input image and witnessed control execution, the modeled accesses satisfy the checked ordering and protocol obligations, under the listed initialization, proxy, and progress assumptions.

That gives the result a precise meaning while leaving a clear path toward stronger coverage.

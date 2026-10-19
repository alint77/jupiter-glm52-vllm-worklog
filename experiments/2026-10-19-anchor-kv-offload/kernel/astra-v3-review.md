**The blocking finding is grid-barrier liveness.** The “last block has arrived” argument does not prove that the last block can start. Subject to a valid execution arrangement and serialized cache mutations, I do not find another wrong-row bug in the sweep or copy mapping.

Line numbers below refer to the supplied source. This is a static review; I did not run commands or tests.

1. **[P1] The ordinary kernel launch can deadlock before the last arrival — lines 250–284, 370. Q1: no.**

   Consider the measured 12-block grid running with capacity for eight of these CTAs:

   - Eight blocks execute lookup and increment `ctl[9]`.
   - All eight wait for `ARR == step`, retaining their CTA resources.
   - The remaining four blocks cannot start.
   - Nobody publishes `ARR`.

   `__nanosleep` does not relinquish the CTA’s registers, shared memory, or residency. The statement “every block has started once the last arrives” is true, but reaching that condition is exactly the unresolved problem.

   Your small grid on an otherwise available GH200 explains why testing succeeds. Finite, independent competing kernels can merely delay progress until they finish; their presence does not automatically imply deadlock. However, a restricted allocation, another persistent workload, or another interdependent spinning grid can remove that escape.

   CUDA graphs do not add a residency guarantee to this ordinary launch. Preemption is not a forward-progress guarantee on which this protocol can rely. MPS and green-context resource restrictions must be included in the execution contract; NVIDIA explicitly cautions that even disjoint green-context partitions do not guarantee concurrent forward progress. [CUDA block execution requirements](https://docs.nvidia.com/cuda/archive/13.0.3/cuda-c-programming-guide/index.html#thread-hierarchy), [green-context concurrency](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__EXECUTION__CONTEXT.html).

   **Fix:** use a supported cooperative launch with an appropriately bounded grid, or retain a kernel-boundary fallback. Checking nominal occupancy alone does not turn an ordinary launch into a cooperative one. [Cooperative grid synchronization](https://docs.nvidia.com/cuda/archive/12.9.1/pdf/CUDA_C_Programming_Guide.pdf).

   The earlier last-block design allowed other CTAs to retire. Keeping them alive to copy introduces this liveness requirement.

2. **[P1 integration condition] Two-sided invalidation must also be serialized, and the inverse audit does not establish content freshness. Q4/Q6.**

   For the supplied kernel, unconditional eviction is justified by the invariant:

   - A claim has `where[s] == -2`, so its slot cannot also appear in `slot_of`.
   - Victim slots are distinct from claim slots.
   - Each sweep window contains distinct rows because `N >= 4096`.
   - Assigning a row stamps the current step, preventing selection again after wrap.
   - New-slot invalidation completes before selection, and those slots bypass ordinary lookup.

   Thus I see no reason to restore the conditional `where[old]` check **for this closed set of writers**.

   But “every writer clears both” is insufficient when writers overlap. For example, an external invalidator can read `where[s] == r`, the sweep can reuse `r` for another slot, and the invalidator can subsequently clear `slot_of[r]`. Serialize external invalidation with lookup/sweep, or provide a stronger ownership protocol.

   Separately, a source rewrite can leave `where` and `slot_of` perfectly inverse while making the cached bytes stale. The integration must cover physical-slot reuse, prefill writes, speculative rewrites, and any restore/copy path that changes a row.

   **Prefetch before destination assignment is safe** when `miss[i]` names an immutable, accessible source row throughout the read. Destination selection is independent of those bytes. `miss_row == -1` suppresses the destination store; it does **not** suppress the earlier source read.

   Your current-token argument is valid if `new_slots` covers the complete concurrent write set and those writes touch only their designated rows. The inline source cannot establish that condition for the other integration paths.

   Finally, `REL` publishes mapping metadata **before the copies finish**. FlashMLA, subsequent cache operations, and invalidation must wait for kernel completion, not just `REL`.

3. **[P2] Long-running counters overflow much earlier than the documented epoch bound — lines 174, 186–192.**

   `ctl[7]` and `ctl[8]` accumulate into signed 32-bit integers. At 303 claims/step and 50 steps/s, the claims counter reaches `INT32_MAX` in about **39 hours**. The candidate counter can overflow sooner. Signed `+=` overflow is undefined behavior in the source language.

   Use 64-bit cumulative counters, or explicitly defined unsigned wrapping if wrapping telemetry is acceptable. Profiling accumulators need the same treatment.

   Also, `2^30` steps at 50 steps/s is about **249 days**, rather than decades. The stated epoch restriction can remain a contract, but its operational lifetime needs correcting.

4. **[P2 integration] The wrapper leaves several necessary preconditions unchecked — lines 345–373.**

   The relevant omissions are:

   - CUDA tensors must belong to the intended device, and the launch must use that device’s stream.
   - Nonnegative indices and `new_slots` must be below `nslots`.
   - `src` must be GPU-accessible and remain alive.
   - Both row-buffer pointers must be 16-byte aligned. Contiguity alone permits a misaligned byte-offset view.
   - Buffers must satisfy the required non-aliasing relationships.

   These can be established upstream rather than checked on every decode step, but the current wrapper does not establish them. In particular, the device/stream assumption deserves an explicit contract in the TP integration.

For the remaining correctness questions:

**Q2: thread 0’s acquire followed by `__syncthreads()` is sufficient; every reader does not need its own acquire.** The publication chain is:

```text
sweep writer
  → CTA barrier
  → thread 0 release
  → reader thread 0 acquire
  → reader CTA barrier
  → other reader threads
```

The arrival side is subtler. On Hopper/PTX, the existing all-thread SC fences support the ordering: reversing a producer fence and a last-block consumer fence would order the last counter RMW before an earlier arrival, contradicting counter coherence. This relies on SC-fence semantics; relaxed `atomicAdd` alone would not suffice. [PTX synchronization and ordering rules](https://docs.nvidia.com/cuda/parallel-thread-execution/#memory-consistency-model).

I would nevertheless simplify arrival to an explicit, readily auditable protocol:

```cuda
__syncthreads();
if (threadIdx.x == 0) {
  cuda::atomic_ref<int32_t, cuda::thread_scope_device> arrived(st.ctl[9]);
  last = arrived.fetch_add(1, cuda::memory_order_acq_rel)
         == int(gridDim.x) - 1;
}
__syncthreads();
```

Each leader publishes its CTA’s work; the RMW chain accumulates earlier arrivals; the final CTA barrier distributes that visibility. This permits removing the surrounding all-thread device fences. Likewise, the sweep’s CTA barrier followed by a release store provides the publication path without every writer independently executing `__threadfence`. This changes ordering overhead, **not** the liveness finding.

**Q3: I do not see an L1-staleness defect in the sweep.** `__ldcg` bypasses L1. With the synchronization above and no concurrent writers, the earlier lookup accesses on that SM do not require reader-side atomics. Cache policy itself does not establish synchronization. [PTX cache operators](https://docs.nvidia.com/cuda/parallel-thread-execution/#cache-operators).

**Q5: the parity counters and plain initial `step`/`hand` loads are sound under the stated serialization.** Different shapes do not matter:

- The current parity was cleared by the preceding invocation.
- The other parity has no remaining readers from that completed invocation.
- Every CTA has consumed this invocation’s initial state before the sweep updates it.
- The next ordinary same-stream invocation starts after the complete copy finishes.

Initialize both parity counters and the arrival counter consistently; do not capture a per-shape state reset into replay graphs. Separate graph executables sharing this state must remain serialized, including their internal accesses. Ordinary stream and graph dependency edges provide the necessary ordering. [CUDA execution ordering](https://docs.nvidia.com/cuda/parallel-thread-execution/#memory-synchronization).

For Q7, there is one measurement issue to fix before choosing among optimizations:

**`gtime_after` does not establish the claimed hardware dependency — lines 83–85.** The assembly never references its input operand. The resulting timer instruction has no dependency on `dep`; an unused inline-assembly operand is not evidence that preceding loads completed. Verify the emitted instruction sequence with a real dependency before treating “candidate loads” versus “stores” as a settled attribution. The end-to-end timings remain useful. Also, the zero-miss path leaves `t_cand == 0` but still accumulates `t_cand - t_pre`.

My performance priorities would be these. The ranges are **experiment targets inferred from your timings**, not predicted or measured gains, and are not additive.

| Priority | Change | Plausible net target |
|---|---|---:|
| 1 | Distribute eviction and `slot_of` updates after release | 0.5–1.2 µs |
| 2 | Replace all-thread device fences with leader publication protocols | 0.2–0.6 µs |
| 3 | Try 512/256-thread blocks while preserving sweep coverage | 0.5–1.5 µs, with regression risk |
| 4 | Replace runtime sweep modulo with bounded conditional subtraction | 0.1–0.4 µs |

The first option has a useful **multi-round-safe subset**:

- Keep current-step `status` stamps and reference-bit clears inside the sweep.
- Publish `where[new]` and `miss_row` before release.
- After release, assign one thread per claim to read `slot_of[r]`, clear `where[old]`, and write `slot_of[r] = new`.

This also removes **all `slot_of` candidate loads from the sweep**, rather than merely distributing its stores. No saved-old-slot descriptor is required: the unique owner can read it after release.

It remains correct across multiple rounds because the in-sweep status stamps prevent row reselection. Pending outputs still read the already-published `where[new]`; no entry this step can require an evicted old slot. The inverse mapping becomes temporarily inconsistent until commit completes, which is acceptable only under the existing kernel-completion boundary for external observers.

You can additionally defer status updates and reference clears **in the final round**, once `total >= need` is known. Earlier rounds must retain the state needed by subsequent candidate reads. Moving `where[new]` after release would require another remapping mechanism or synchronization.

For smaller blocks, preserve the candidate-window size initially: 512 threads × 8 candidates or 256 × 16. Simply reducing `THREADS` with `SWEEP_V == 4` can turn the measured one-round case into several rounds. Check register pressure and sweep latency alongside lookup gains; the grid’s residency requirement also changes.

The modulo removal is straightforward: with `hand < N` and candidate offset `< 4096 <= N`, `hand + offset < 2N`. One conditional subtraction suffices. The same bound applies to advancing the hand.

I would put a distributed two-level sweep behind these changes: its extra coordination may consume much of the saving on your one-round workload.

PDL belongs to the **pipeline** experiment. It can hide launch latency and independent preparation behind the producer, but producer-dependent reads still need synchronization. The current lookup consumes `idx` almost immediately, limiting the independently overlapable work. PDL also supplies no remedy for the grid barrier. [NVIDIA PDL requirements](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html).

Finally, 303 and 235 copied rows are only approximately **0.47 µs and 0.37 µs** of payload transfer at your measured 421 GB/s. Those are bandwidth lower bounds, not latency predictions, but they point toward sweep coordination, scattered metadata operations, and lookup scheduling as the better places to seek another 2 µs.

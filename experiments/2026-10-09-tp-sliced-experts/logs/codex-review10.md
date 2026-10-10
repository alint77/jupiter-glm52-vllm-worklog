**I would try C plus the inexpensive parts of B for M=8. My implementation ranking is C > B > corrected A > D.** A offers useful concurrency, but its reservation protocol is incorrect as written, and repairing it adds scheduling machinery that C largely avoids.

Keep the existing 288-thread, GR1=2 path for M=16/32 and select the new variant by shape. That preserves their measured performance without calibration-based balancing.

Your producer’s intrinsic service time is approximately **2.0 us**, after excluding the 0.17 us empty wait, which is backpressure. Reaching the consumer’s 1.4 us requires removing or overlapping roughly **0.6 us/unit**. That is a plausible target for separating scheduling from issuing.

**C directly separates the expensive chains.** Give the scheduler ownership of `q`, stealing, claims, group decoding, and record loading. It publishes complete group records through a small FIFO; the existing producer remains the sole owner of ring position `it`.

Use two queue slots initially, and keep a slot occupied until its group has been issued. Wait for queue capacity before claiming more work. This bounds outstanding claims to approximately the current group plus one lookahead, avoiding a larger tail-hoarding problem.

An additional useful optimization: the scheduler can **test readiness once** while preparing an R1 record.

- If ready, perform the acquire and publish an `already_ready` bit with the record.
- Otherwise, publish the record immediately; the issuing warp waits after launching weights.

A properly synchronized release/acquire queue handoff can carry the scheduler’s acquired visibility to the issuing warp. Every copying lane must participate in that handoff, directly or through a correctly placed warp synchronization, and still execute the required async-proxy fence. This can move successful ready-check latency off the issuing warp without delaying weights behind an unsuccessful check.

Do budget the scheduler using **atomic return latency**, not the 0.06 us atomic issue timestamp. C isolates the SB5 problem; it does not eliminate atomic contention. Its benefit depends on the scheduler sustaining a group every ≤1.4 us, including publication. The supplied numbers make that plausible, especially after specializing the division.

All scheduler workspace accesses remain after the PDL wait under your stated rule.

For **A**, there are two separate correctness problems.

First, **a reservation ticket is not sufficient permission to use a parity barrier**. For example:

```text
P0 reserves positions  0..11
P1 reserves position 12

P1 waits on empty[0], parity ((12 / 4) - 1) & 1 = 0.
```

That parity also describes the completion associated with position 0. It does not identify position 8 uniquely. P1 can therefore mistake an older completion for the one it needs. Two producers can then interfere with the same physical slot and `full` barrier generation.

One repair is a monotonic **producer turn per slot**, in addition to the existing empty barrier:

```cpp
// Initialized once, before producers start:
issue_turn[s] = s;

// Each producer, for its reserved position pos:
int s = pos % STAGES;

if (lane == 0) {
  while (load_acquire_cta(&issue_turn[s]) != pos)
    __nanosleep(32);

  if (pos >= STAGES)
    mbar_wait(&empty[s], ((pos / STAGES) - 1) & 1);
}
__syncwarp();

issue_entire_position(pos);  // metadata, barrier setup, all copy issues

__syncwarp();
if (lane == 0)
  store_release_cta(&issue_turn[s], pos + STAGES);
```

These are semantic helper names, not literal PTX declarations. The turn counter establishes that the preceding producer generation has been issued; the empty barrier establishes that its consumers have finished. Together they make the parity check meaningful. No extra consumer-side counter is required.

Second, **claim order and reservation order can create a dependency deadlock**:

```text
P0 claims a required R0 group, then gets delayed before reserving.
P1 claims an R1 group and reserves earlier ring positions.
P0 reserves its R0 positions behind that R1 group.
```

If that R1 entry needs P0’s R0 contribution, P1 waits for readiness while consumers cannot get past its R1 slot to finish the required R0 work.

Thus, merely fixing slot generations is insufficient. You must preserve the relevant w13-before-dependent-w2 ordering when assigning positions. A CTA-local serialized claim-and-reserve operation is one solution, provided it also covers prefetched claims: **an already-claimed but unreserved group cannot be allowed to disappear from this ordering**. Shared S0/S1 dependencies need the same treatment.

Your concern about holding a claim behind another producer’s 12-chunk group is otherwise a performance issue: it increases outstanding work and can worsen the tail. The dependency inversion above is the correctness issue.

With those repairs, the **two-END scheme is sound** under these conditions:

- `full[s]` retains one arrival per position. Two producers do **not** imply an arrival count of two.
- `empty[s]` retains the existing eight consumer-warp arrivals.
- Each producer emits END only after draining all its claimed work and never reserves anything afterward.
- Every consumer releases the first END slot and advances its normal stage/parity state.
- The second END is terminal. Releasing its empty barrier is optional if nothing subsequently uses it.

The first END must also advance the producer-side slot turn, like any other issued position. Consumer position counts include END slots.

On scheduling: under the usual modulo-four warp placement, warp 8 already shares SMSP0 with consumer warps 0 and 4. Adding warp 9 similarly puts three warps on SMSP1. That creates additional instruction competition, but permanent producer starvation is not the expected consequence. Two producers help principally by overlapping stalls; they do not double TMA or memory bandwidth. Check actual register allocation after the change, including allocation granularity.

For **B**, I would prioritize the edits as follows.

1. **Specialize `g1` and expose a dedicated R1 path.**

   This is the cleanest attack on the measured 0.22 us decode portion. Constant division should become a much shorter sequence; an R1-specific record path can also avoid irrelevant w13 scale selection/shuffling if those instructions survive in the current SASS.

   “Incremental decoding” needs caution: successive atomic claims are not consecutive indices. You cannot advance `(ei, t)` by one across claims. Within a claimed range, incrementing works, but your `t0 + ci` already does that.

2. **Move the next-claim atomic past consumption of the current record loads.**

   A useful placement to test is after the current record has been consumed, with substantial current-stage issuing work still remaining to overlap the atomic.

   Moving it to the very end avoids the record dependency but can simply transfer the latency into the next `shfl`. Also, if you add an early ready probe, do not accidentally leave its result pending on the atomic’s scoreboard.

   The v40 result demonstrates a potential regression to avoid; it does not establish that removing this interaction will recover 0.33 us from v41.

3. **Launch weights before row metadata stores—but preserve barrier publication.**

   This gives weights an earlier start, although it does not remove the stores’ instruction cost. `local` still comes from the record, and registering the complete transaction count normally also requires `ntok`.

   Do **not** simply move an `arrive.expect_tx` and the TMAs ahead of metadata stores. Metadata written after the release arrival is not correctly published merely because transaction completion occurs later.

   A safe structure is:

   ```cpp
   wait_for_slot();

   if (lane == 0) {
     expect_tx_without_arrival(full[s], total_bytes);
     issue_weight_tma();
     issue_scale_tma();
   }

   write_desc_and_row_metadata();
   __syncwarp();

   if (lane == 0)
     arrive_release(full[s]);

   acquire_ready_if_needed();
   fence_proxy_async();
   __syncwarp();
   issue_x2_copies();
   ```

   Keeping the arrival outstanding prevents the phase from completing before metadata publication, even if the weight transfers finish quickly. This adds a barrier instruction, so measure the resulting issue interval as well as the earlier weight launch.

4. **Hoist a nonblocking ready probe, not the blocking ready loop.**

   Once `ei` is known, an early relaxed load can overlap record/setup work. Inspect its result later, after launching weights. If it failed, continue polling there.

   Hoisting the entire blocking wait ahead of weights sacrifices the overlap that your current ordering deliberately provides.

The likely B gains are concentrated in group decoding, dependency placement, and earlier weight launch. I would not assume those changes alone reliably bring 2.0 us below 1.4 us; that is why I prefer C as the structural change.

**Yes: a relaxed flag load followed by `fence.acq_rel.gpu` after success is a valid acquire pattern**, assuming the successful load reads the matching `st.release.gpu` publication and the existing epoch/lifetime protocol remains valid:

```cpp
uint32_t seen;
do {
  seen = ld_relaxed_gpu(flag);
  if (seen != epoch)
    __nanosleep(32);
} while (seen != epoch);

asm volatile("fence.acq_rel.gpu;" ::: "memory");
fence_proxy_async();
```

Retaining this sequence in every copying lane preserves your current acquisition structure. The GPU fence supplies generic-memory acquisition; the proxy fence supplies the required ordering into the async proxy. One does not replace the other.

However, **this mainly saves acquire work on failed polls**. If most entries are ready on the first load, it replaces one acquire load with one relaxed load plus one acquire fence. The necessary cache-ordering work may still appear once, and the replacement may be neutral or slower. It is not a reason to subtract the whole 0.22–0.30 us from every unit.

Finally, **there is no supported CUDA/PTX mechanism to request a particular SASS dependency scoreboard**.

- `asm volatile` and a `"memory"` clobber cannot demand “ATOMG uses a different scoreboard from these LDGs.” They can constrain compiler motion, sometimes at the expense of overlap.
- Moving the operation into another warp gives it independent warp dependency state. This is a real isolation mechanism, though atomic and memory-system contention remain shared.
- Issuing from another lane of the same warp does not provide that isolation.
- `red` plus a separate counter load cannot replace a ticket-returning atomic. Concurrent claimers can observe the same counter value or a value advanced by other claimers.

For D, a compact predecoded R1 work table could reduce decoding and load instruction count, but it introduces construction and traffic costs. I would put that behind the queue split and the targeted B edits given the evidence here.

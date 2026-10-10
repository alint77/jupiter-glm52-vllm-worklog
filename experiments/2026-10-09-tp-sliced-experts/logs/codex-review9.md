**I would cut the repeated loop metadata first, pipeline two row blocks inside `consume_routed`, and replace the completion fences with explicit release/acquire ordering.** Those offer a credible combined path from 1.38 to 1.2 µs. I would not make finding a shorter INT4 decode the main bet.

The 61 µs bandwidth floor is a separate, harder target: a 1.2 µs cycle across roughly 50 units already spends 60 µs before the exposed head and tail.

| Rank | Change | Main opportunity |
|---|---|---|
| 1 | Group-aware consumer loop; pipeline flush metadata | Remove repeated work from the 13.5 µs loop-head bucket |
| 2 | Two-row-block software pipeline inside routed math | Hide decode → HMMA → FFMA dependencies without doubling unit state |
| 3 | Release/acquire completion; defer use of the returned count | Reduce the 4.6 µs handoff/activation burden |
| 4 | Assign activation to the producer owning the first R1 group | Remove returned-count and activation work from consumers |
| 5 | CTA-local R0 reduction followed by ordinary stores | Remove unnecessary global reductions when `R0S == 1` |

As a **budget, not a prediction**, cutting 0.10 µs of head, 0.06 µs of math, and 0.02 µs of amortized handoff gets your R0 cycle to 1.20 µs.

For the loop head, the largest concrete redundancy is that **R0 has one group’s metadata but pays to rediscover it 12 times**.

After acquiring the first stage of a group, retain `kind/q/ei/tile/ntok/nch`. Run a counted inner loop over its chunks. For intermediate chunks, the consumer needs only:

```text
acquire full stage
consume_routed
release empty stage
advance stage and phase
```

It does not need another descriptor load, kind dispatch, `fs`, or `rows4`. Derive `last` from the group loop. Apply the same principle to shared groups.

For R0, load `fs` and `rows4` only on the final chunk. For R1, load the flush metadata during the final part of routed computation, and defer the `fs *= x2_scale` operations until the flush. Ideally, issue those LDS instructions while the last independent decode/HMMA work can cover their latency.

**All stage-resident metadata must be captured before the empty arrival.** Your current placement correctly enforces that lifetime; preserve it when moving the loads.

Two smaller changes fit here:

- Fetch the adjacent `fs[0:2]` with an aligned 64-bit shared load.
- Consider storing rows as `[stage][token_parity][j]`, so the flush’s row list becomes a contiguous vector load for each half-warp.

Moving loads alone does not save instructions, and merely moving their stalls to the tail achieves little. The group loop actually removes work; scheduling the remaining loads hides latency.

I would postpone an independent metadata-publication barrier. It can let consumers acquire descriptors before TMA completion, but introduces another protocol. Never speculatively read a reusable stage descriptor before its publishing acquire.

Also, the 62% long-scoreboard attribution at `try_wait` is partly the **consequence of consumer skew**. It is not all removable barrier overhead.

For routed math, **interleave two `mb` values before interleaving two units**. You already have four independent row blocks and independent accumulators available.

The current logical chain is approximately:

```text
decode w0[mb] → MMA0 → decode w1[mb] → MMA1 → scale FFMA
```

Try an explicit two-block schedule:

```text
decode w0[0], decode w0[1]
MMA0[0], MMA0[1]

decode w1[0], decode w1[1]
MMA1[0], MMA1[1]

decode first halves for blocks 2 and 3
scale/accumulate completed blocks 0 and 1
...
```

Use two temporary `d[4]` sets and a small number of decoded fragments. Preserve the two-HMMA K32 group and each accumulator’s sequence of group-scale FFMAs.

This gives independent instructions around both the half-arithmetic dependencies and the HMMA-result dependencies. It needs much less additional state than two complete units.

Check these details in the resulting SASS:

- Whether the compiler already performs this interleaving.
- Whether the second MMA uses its temporary accumulator in place.
- Whether zero-initialized MMA inputs become zero operands rather than register initialization/copies.
- Whether pure arithmetic wrappers have unnecessary `volatile` or `"memory"` constraints.
- Whether the extra temporary state creates spills or additional register copies.

The 0.84 IMAD/HMMA attributed to moves deserves attention alongside decode. A source-level instruction saving that causes more allocation copies can lose immediately.

There is an **exact alternative decode**, although it still has nine core instructions. Change the packing magic to `0x2c00`, use two half adds, and reuse the magic as the multiplier for the other two outputs:

```cuda
__device__ __forceinline__
void decode_int4_add2(uint32_t w, uint32_t* a) {
  constexpr uint32_t magic = 0x2c002c00u;  // half2(2^-4)
  constexpr uint32_t blo   = 0xac08ac08u;
  constexpr uint32_t bhi   = 0x9c809c80u;

  const uint32_t w8 = w >> 8;
  const uint32_t l0 = lop3_and_or(w,  0x000f000fu, magic);
  const uint32_t l1 = lop3_and_or(w8, 0x000f000fu, magic);
  const uint32_t h0 = lop3_and_or(w,  0x00f000f0u, magic);
  const uint32_t h1 = lop3_and_or(w8, 0x00f000f0u, magic);

  asm("add.rn.f16x2 %0, %1, %2;"
      : "=r"(a[0]) : "r"(l0), "r"(blo));
  asm("add.rn.f16x2 %0, %1, %2;"
      : "=r"(a[1]) : "r"(l1), "r"(blo));
  asm("fma.rn.f16x2 %0, %1, %2, %3;"
      : "=r"(a[2]) : "r"(h0), "r"(magic), "r"(bhi));
  asm("fma.rn.f16x2 %0, %1, %2, %3;"
      : "=r"(a[3]) : "r"(h1), "r"(magic), "r"(bhi));
}
```

The identities, for every code \(c\in[0,15]\), are:

\[
\begin{aligned}
L(c)&=2^{-4}+c\,2^{-14},\\
L(c)-(2^{-4}+8\,2^{-14})&=(c-8)2^{-14},\\
H(c)&=2^{-4}+16c\,2^{-14},\\
H(c)2^{-4}-(2^{-8}+8\,2^{-14})&=(c-8)2^{-14}.
\end{aligned}
\]

All results are exactly representable, including positive zero at \(c=8\) under round-to-nearest. This preserves your fragment order.

The potential benefit is **fewer distinct constants and simpler operand requirements**, not a proven throughput advantage of HADD2. Inspect whether it reduces the surrounding moves/register pressure. I do not know a reliable sub-nine-instruction exact decode for this packing on sm_90a.

I would keep the group-scale FFMAs in FP32 after the two MMAs. Folding arbitrary bf16 scales into fp16 weights or activations changes rounding and potentially range. Instead, prepare `s0/s1` early and schedule their FFMAs against another block’s decode/MMA work.

`ldmatrix` is also not an obvious improvement here: the routed layout already produces four useful weight words with one conflict-free LDS.128, and each activation fragment feeds four MMAs. Replacing that load mechanism does not remove decoding. Expanding weights into shared memory adds traffic and substantially increases the footprint.

The 25% math slowdown with TMA enabled has several plausible causes; the measurements do not isolate one:

| Candidate | Why it fits | Useful discriminator |
|---|---|---|
| Producer instruction issue | TMA avoids moving each byte through registers, but its issuing warp still executes queue, metadata, synchronization, and copy instructions | Preserve producer control flow while suppressing transfers |
| Shared-memory contention | The measured math includes LDS operations while TMA fills other stages | Compare LDS latency/stalls under controlled TMA traffic and different issue timing |
| Changed warp eligibility | Full execution changes the alignment of producer, consumer, and handoff work | Compare issue/eligibility by scheduler and warp |
| Clock differences | Base-clock NCU measurements do not automatically establish identical clocks in every timing experiment | Compare SM cycles as well as microseconds |

TMA removes the instruction cost of transporting individual bytes; it does not make concurrent traffic free. NVIDIA documents that distinction in its [Hopper tuning guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/#tensor-memory-accelerator).

Your extra-stage and prefetch results argue against insufficient lookahead. I would test **when transfers overlap arithmetic**, rather than increasing the amount outstanding again.

For completion, start with a smaller, reviewable change: **retain the existing consumer rendezvous, replace the threadfence/relaxed-atomic pattern with an explicitly ordered atomic**.

The required dependency chain is:

```text
all consumer y13 reductions
    → consumer handoff barrier
    → lane-0 release on done13
    → completing thread's acquire
    → warp synchronization
    → activation reads
```

The CTA barrier establishes communication from the participating writers to warp 0. Release/acquire communication through the counter then carries that dependency between CTAs. These are the relevant [PTX synchronization rules](https://docs.nvidia.com/cuda/parallel-thread-execution/#memory-synchronization).

A straightforward first version is:

```cuda
// Other consumer warps execute handoff_arrive().
handoff_sync();  // warp 0

unsigned last = 0;
if (lane == 0) {
  unsigned old;
  asm volatile(
      "atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
      : "=r"(old)
      : "l"(&ws->done13[q][ei]), "r"(nch)
      : "memory");
  last = old == UNITS0 - nch;
}

last = __shfl_sync(0xffffffffu, last, 0);
if (last) {
  __syncwarp();  // carry lane 0's acquire to activation lanes

  // Existing activate_route loop, executed by the whole warp.
  ...

  fence_proxy_async();  // retain on every writing lane
  __syncwarp();
  if (lane == 0)
    st_release(&ws->ready[q][ei], epoch);
}
```

With this chain, the explicit `__threadfence()` before the count, before activation, and before the ready release are redundant. `__threadfence()` provides sequentially consistent device-scope fencing; this protocol needs narrower ordering. [CUDA memory-fence documentation](https://docs.nvidia.com/cuda/archive/13.0.2/cuda-c-programming-guide/index.html#memory-fence-functions)

Keep the proxy fences and the copying lanes’ ready acquires initially. They address generic/async visibility, which is a different obligation from ordering the counter. Keep compiler memory clobbers on the synchronization wrappers as well.

A subsequent refinement is `atom.release` plus the existing `fence_acq_rel_gpu()` **only on the winner**, before the warp synchronization. This avoids acquiring on every non-completing count.

To overlap the returned atomic’s latency, issue it and postpone the first use of `old` until after independent work. But there is a critical progress constraint:

**Do not enter a blocking full-stage wait while holding an unresolved completion that might make that stage ready.**

A safe bounded experiment is:

1. Issue the completion atomic.
2. Nonblocking-test the next full stage.
3. If ready, perform independent computation before consuming `old`.
4. Otherwise, resolve the completion immediately.
5. Resolve pending work before exit or another potentially dependent blocking operation.

This can hide return latency; it cannot eliminate the release’s obligation to order prior writes. Inspect the SASS to ensure the first scoreboard-dependent use actually moved.

The stronger asynchronous design is to **make the producer owning R1 tile zero the entry’s activator**.

Because each R1 group is claimed uniquely, the group with `t0 == 0` provides an owner without another election:

```text
R0 consumers:
    flush
    existing handoff join
    lane 0: red.release.gpu.global.add.u32 done13, nch
    continue

R1 tile-zero producer:
    issue its weight TMA
    acquire-wait until done13 == UNITS0
    activate the entry using the producer warp
    proxy fence + warp join + release ready
    continue activation-row copies
```

Other R1 producers retain their ready wait. A release reduction can publish completion; the activator must acquire through a load, since a reduction itself does not establish an acquire pattern. [PTX release/acquire patterns](https://docs.nvidia.com/cuda/parallel-thread-execution/#release-and-acquire-patterns)

This removes the returned count and activation from the consumer path, using an existing warp. The cost is delayed producer service during activation. Your tenth-warp result does not settle this tradeoff, but it makes producer scheduling and ring starvation essential measurements.

Preserve the property that outstanding R0 work can finish independently of R1 readiness, including across stealing and prefetched claims.

There is also a useful structural observation in this particular TP slice: **with `R0S == 1`, a CTA owns the complete K reduction for its w13 output tile**. The only remaining partition is between its four K-slice warps.

That permits:

```text
four warp partials
    → shared-memory reduction inside the CTA
    → ordinary global stores to the owned y13 tile
    → completion publication
```

Keep each partial’s existing FP32 `acc * fs` operation before combining partials. Then this changes FP32 summation order without moving scale arithmetic across the reduction.

Once every participating `y13` element is overwritten before activation, its zeroing in `activate_route` can also disappear.

The tradeoff is extra local synchronization and scratch reads. Since R0 flush itself is only 0.7 µs/CTA, the attraction includes reducing the global atomic traffic that the completion path must order. This overwrite scheme does not apply unchanged to split R0 groups or shared work with multiple CTA contributors.

For R1, local reduction could similarly reduce four global updates to one, but updates from different experts still require global combination. Its per-unit synchronization makes it a less attractive first experiment.

If these changes stall, the larger architectural experiment would be two warpgroups using `wgmma.mma_async.m64n8k16`, decoded weights in registers and activations in shared memory, with each warpgroup owning rows across K. That operand form exists on Hopper. [PTX WGMMA documentation](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-multiply-accumulate-instructions-wgmma-mma)

However, group-32 scaling still requires separate K32 partial results before applying FP32 scales. You would need pipelined partial accumulators, appropriate activation layout, and correctly placed WGMMA fences/waits. Decode remains. I would rank this below the changes above.

Finally, the ablations support **consumer limitation in the full kernel**, but they do not establish a 61 µs delivery path. Loads-only still takes 78 µs and has much larger completion costs. Those ablation times cannot simply be combined as independent resource bounds.

Likewise, removing the entire 4.6 µs handoff bucket from your approximately 68 µs of consumer work still leaves about 63.4 µs, before exposed startup and drain. Reaching the bandwidth floor requires arithmetic/control savings as well as communication overlap.

My first three isolated variants would therefore be the group-aware loop, the two-block arithmetic pipeline, and the ordered-atomic handoff. Measure their combination before committing to the producer-activation or WGMMA rewrite.

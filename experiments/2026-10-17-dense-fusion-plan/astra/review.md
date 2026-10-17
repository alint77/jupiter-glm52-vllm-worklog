**Your o-chain correction is right. The q-chain remains plausible, but neither the persistent-kernel argument nor the gain estimates are complete enough to justify the claimed 0.6–0.8 ms/step.** I would first remove unnecessary glue and test PDL across the norm boundary.

This review uses only the supplied material; I ran no commands.

1. **Critical — The proposed W_UV prologue does not match o_proj’s ownership.**

   Each o_proj CTA owns 16 output channels and consumes the entire K=4096. Therefore, each needs W_UV’s output for all 16 heads. Recomputing W_UV in each of 384 CTAs means approximately **1.5 GiB of W_UV weight reads** and 384 copies of its arithmetic. Cache residency does not make that inexpensive.

   Splitting o_proj over heads changes the ownership, but **does not by itself eliminate W_UV duplication**: if every `(output tile, head)` CTA computes its own head’s W_UV, each head is still recomputed 384 times.

   **Recommendation:** Drop the stated prologue. If pursuing fusion, compute W_UV once using producer tasks, materialize its bf16 output, and let o_proj consumers read it.

   Of the proposed structures, prefer a producer/consumer persistent kernel that retains **full-K ownership for each o_proj output tile**. Start with one W_UV-complete condition. Per-head readiness can subsequently permit an o_proj CTA to process completed heads while retaining its own accumulators; this does **not** require split-K across CTAs.

   Head-split o_proj is a lower-priority alternative. Sixteen fp32 partials require about **3 MiB of workspace at M=8**, plus reads, reduction, synchronization, and substantially more CTAs. Previous split-K results argue against it, although they do not conclusively test readiness-driven overlap with W_UV producers.

2. **High — 1b’s queue ordering can prevent producer starvation, but “claimed” is not “completed.”**

   The local deadlock argument is valid under these conditions:

   - Claims occur only when a running CTA can execute the item.
   - A producer finishes its GEMM, stores its bf16 output, and publishes completion before claiming another item.
   - All required producer items precede consumer items in the queue.
   - Producers never wait for consumers or for unclaimed producer work.

   Under those conditions, a waiting q_b consumer cannot depend on an **unscheduled producer item**: its producers have already been claimed by executing CTAs. The entire grid need not be simultaneously resident.

   However, this argument does not establish data visibility. Completion must use a correct device-scope release/acquire protocol. An atomic count or volatile ready flag alone is not a sufficient specification.

   There are 128 q-producing tiles and 36 remaining qkv_a tiles. A q-ready condition can cover the 128 q tiles, but kv readiness must be tracked separately if anything consumes kv before the fused kernel completes.

   **PDL, side streams, and graphs add separate requirements:**

   - Keep upstream `griddepcontrol.wait` semantics before consuming predecessor output.
   - PDL does not replace synchronization between phases within this kernel.
   - Do not create a cycle where an indexer stream waits for fused-kernel completion while that kernel waits for indexer progress.
   - A side-stream consumer needs an explicit dependency that protects normalized q_c. A local flag does not automatically become a graph or stream dependency.
   - Queue counters and readiness state need correct initialization on every graph replay, including any concurrent use of scratch buffers.

   **Recommendation:** Write down the producer/consumer state machine and memory-ordering protocol before implementation. Keep producer tasks free of global waits. Do not introduce an ordinary-grid barrier based merely on an estimated occupancy bound.

3. **High — “16.8 MB fits in aggregate shared memory” does not establish a useful prefetch scheme.**

   One q_b output tile has **64 KiB of weights**. Staging all 256 tiles requires 16 MiB distributed across their owners. Dividing that by 132 SMs gives roughly 124 KiB per SM, before activation buffers, reductions, and other shared storage.

   Shared memory belongs to individual resident CTAs. Actual feasibility depends on the persistent worker count, per-CTA storage, registers, and which workers still execute qkv_a. With fewer workers, only a subset of q_b tiles can be staged.

   More fundamentally, qkv_a and q_b weights share the same HBM bandwidth. Prefetching q_b while qkv_a still streams weights can simply redistribute bandwidth. The useful opportunity is the producer’s **underutilized tail and normalization interval**, whose size is unmeasured here.

   **Recommendation:** Specify a bounded staging scheme and measure its incremental benefit. Do not make full q_b-weight staging a premise of the design.

4. **High — The gain estimates combine different effects and overstate what the trace establishes.**

   A fitted streaming intercept of approximately 3.3 µs is **not evidence of a removable 3.3 µs launch cost at every boundary**. It can include memory latency, bandwidth ramp-up, occupancy effects, and the kernel’s tail. A persistent kernel can still pay these costs when changing phases.

   Conversely, a 0.1–0.2 µs inter-kernel gap does not prove the next kernel has no startup cost: that cost occurs inside its duration.

   The supplied trace shows:

   | Boundary | Producer end | Consumer start | Observed overlap |
   |---|---:|---:|---|
   | norm → q_b | 220.3 µs | 220.4 µs | None |
   | W_UV → o_proj | 28.9 µs | 29.1 µs | None |

   Thus, this trace does **not** show q_b or o_proj preloading weights during those immediate predecessors.

   The source establishes that decode_gemm enables PDL and loads its first weight batch before waiting. It does not establish that the preceding Triton or nvjet kernels release dependents early, or that the captured graph permits early launch on those edges. Kernel names cannot answer whether these Triton instances emit `launch_dependents`.

   At M=8, the first weight batch covers, per CTA:

   - q_b: one quarter of its K range.
   - o_proj: one eighth.
   - qkv_a: one twelfth.

   That is a bounded preload, not the complete weight stream.

   **Recommendation:** Inspect the actual generated kernels and graph dependencies, then measure the overlap. If the norm producer has no early-release mechanism, q_b’s PDL prologue cannot overlap that norm merely because q_b enables PDL.

   The savings ledger should also change:

   - The two presumed glue kernels total **3.7 µs**, but their attribution is unverified and kv normalization must survive somewhere.
   - Removing a kernel replaces its necessary work; it does not automatically save its entire duration.
   - Count 1b only as an incremental improvement over 1a if 1a is already included. Both can target the same normalization boundary and startup interval.
   - W_UV’s 3.9 µs contains required arithmetic and weight traffic. It is not an elimination budget.
   - With both weights streamed from HBM, the o-chain’s weight-only floor is approximately **54.5 MB / 3.6 TB/s = 15.1 µs**, rather than 14.4 µs. A lower estimate needs an explicit cache assumption.

   Saving 4 µs across 78 invocations would indeed save about 0.31 ms, but that multiplication is not evidence that 4 µs is recoverable. Treat **0.6–0.8 ms as an unvalidated target**, not an expected result.

5. **High — Norm into q_b is feasible, but its duplication and remaining consumers undermine the “cheap 4 µs” claim.**

   You are right about q_c’s layout: its logical width is 2048, but its row stride is 2624. The supplied dispatcher rejects that noncontiguous view.

   However, a separate contiguous copy is not inherently necessary. A norm kernel can read the strided slice and directly write contiguous normalized q_c. The alleged copy must first be attributed; the supplied Python split operation itself only creates views.

   Recomputing RMSNorm in 256 q_b CTAs duplicates reductions over approximately **4.2 million elements at M=8**. The aggregate input is about 8 MiB per pass. Although each CTA already reads the activation for GEMM, that does not make the norm free: the GEMM partitions K across warps and streams fragments, while RMSNorm requires a full-row reduction before normalization.

   Retaining activations, performing another pass, or writing normalized values into shared memory each introduces a cost. Matching the original reduction tree can further constrain the implementation.

   The designated normalized-q_c writer also needs a dependency contract. Waiting for q_b completion is straightforward, but could delay the indexer **if its current dependency allows earlier execution**. The supplied forward function does not establish the actual side-stream event placement.

   **Recommendation:** First test a single-pass kernel that reads strided q_c and kv_c and produces their required normalized outputs. Then compare replicated norm inside q_b against that improved baseline.

6. **High — Preserving bf16 boundaries is necessary; matching the norm formula is insufficient.**

   For RMSNorm, establish the actual generated implementation’s:

   - Epsilon and placement.
   - Input conversion and fp32 reduction tree.
   - Mean calculation and reciprocal-square-root behavior.
   - Scale multiplication, intermediate casts, and compiler contraction.
   - Final bf16 rounding before q_b and indexer wq_b.

   Matching reduction order alone does not establish bitwise equivalence. The supplied material does not contain enough of the Triton implementation to certify it.

   Likewise, W_UV must produce a **bf16-rounded intermediate** before o_proj consumes it. Keeping that intermediate in registers or shared memory is fine; feeding its fp32 accumulator directly into the next GEMM is not. If using split-K, accumulate and reduce partials in fp32 and convert to bf16 at the original final-output boundary.

   **Recommendation:** Test normalization bitwise using identical bf16 inputs to the reference and candidate implementations. Test GEMMs under your separate allowance for changed fp32 accumulation order. A whole-chain bitwise comparison would otherwise confuse permitted GEMM differences with forbidden normalization changes.

   Keep kv normalization, indexer output, and all existing rope/cache rounding points explicit in the design.

7. **Medium — Several cheaper alternatives deserve testing before two-GEMM persistence.**

   **Merge copy/split preparation and normalization.** A strided-input norm producing contiguous outputs is the most direct candidate. Alternatively, qkv_a’s epilogue can write separate packed outputs while preserving its bf16 conversion. Its 2048/512/64 output partitions align with the current 16-row tiles.

   **Make the relevant glue boundary PDL-aware.** A norm kernel could release q_b early, allowing independent weight loads while normalization completes, with q_b waiting before activation loads. This needs compatible launch/graph behavior and correct synchronization. Making a producer PDL-aware alone does not enable an ordinary dependent kernel to launch early.

   This is particularly worth testing because the norm has relatively little weight traffic. The previous 0.2–0.5 µs PDL gains are useful evidence, but do not directly measure a newly enabled norm→q_b overlap.

   **Merge kv norm with rope/cache preparation where dependencies permit.** This can consolidate small kernels, but verify attribution and preserve operation order. Combining work can also delay an otherwise independent output, so compare the resulting schedule.

   **Concatenate q_b and indexer wq_b only as an anchor-layer experiment.** The common K makes this mathematically straightforward, preserving bf16 outputs. It still streams both weight matrices. A combined kernel can delay main-path q consumption until the indexer work completes, losing existing side-stream scheduling freedom. Its benefit is also limited to 21 layers.

   **Consider W_UK output-layout and gather-preparation fusion before owning the collective.** Writing directly into the required packed buffer may remove local work. Fusing the actual DCP gather introduces communication and readiness obligations; its duration is not removable merely by combining kernels.

   **Recommendation:** Prioritize local preparation and layout changes. Keep communication fusion behind demonstrated local wins.

8. **Medium — Extending decode_gemm is a separate opportunity, not a prerequisite for every fusion.**

   The dispatch is more selective than “these GEMMs use cuBLAS at M=32–64”:

   - q_b falls back above M=16.
   - qkv_a falls back at M=32.
   - o_proj still uses decode_gemm at M=32.
   - All fall back above M=32 through `MAX_M`.

   Extending to M=64 requires more than changing that constant: the compiled specializations stop at NT=4, and additional token tiles increase activation and accumulator register pressure.

   **Recommendation:** For higher-concurrency workloads, the reported 50–63% SOL makes shape-specific GEMM tuning a strong independent candidate. For a c=1 objective, it need not precede glue optimization.

   I agree with deferring AR+residual+RMSNorm→qkv_a fusion. It is feasible in principle with the normalized bf16 boundary preserved, but the communication ownership cost and your failed prefetch experiments make it unattractive. The old rank-0 wait is not available slack on the rank that determines step completion.

Recommended plan, in order:

1. **Rebaseline on a Booster node.** Use the current sliced MoE and fused DCP tail. Attribute both Triton kernels, inspect tensor strides and side-stream dependencies, and verify PDL behavior on the captured edges. Measure whole-step performance across ranks alongside local chains; the older rank-0 trace cannot establish current savings.

2. **Build the smallest q-chain improvement.** Remove any confirmed redundant copy by normalizing directly from the strided qkv output. Merge compatible q/kv preparation. Test PDL from that producer into q_b. Preserve a contiguous normalized q_c for the indexer.

3. **Choose the next experiment by the target workload.** For c=1, benchmark norm-inside-q_b against the improved baseline, checking registers, occupancy, repeated norm cost, and indexer timing. For larger M, prioritize the specific GEMM shapes still falling back to cuBLAS. Neither needs to wait for the other architecturally.

4. **Prototype persistence only if a measurable residual opportunity remains.** Start with q-chain producer tasks, one normalization step after q completion, and q_b consumers. Use bounded weight staging. Include counter initialization, graph replay behavior, and side-stream consumers in both correctness and timing measurements.

5. **Defer the o-chain persistent experiment; drop the replicated prologue.** If attempted, compute W_UV once and retain full-K o_proj ownership. Begin with phase completion; add per-head readiness only if profiling shows a useful tail to overlap. Keep head-split reduction and collective fusion lower priority.

Report each saving as an **incremental whole-step measurement against the preceding version**. That will resolve both double-counting and optimizations that merely move time into neighboring kernels.

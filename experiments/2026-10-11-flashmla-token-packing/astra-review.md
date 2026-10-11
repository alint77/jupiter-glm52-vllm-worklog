**I would prioritize balanced early stop and one producer-ablation experiment before implementing packing.** Variant B is technically plausible, but the supplied measurements do not yet support a win. Variant A does not fit this pipeline’s register and shared-memory budgets. Two effects make B less attractive than the proposed arithmetic suggests: producer work already overlaps consumption, and union lengths round into whole blocks and splits.

Everything below follows from the supplied source and measurements unless explicitly marked as an estimate or inference.

1. **The source establishes a substantial consumer critical path, but does not establish which side dominates.**

   The per-block accounting is:

   | Work per CTA, per 64-key block | Amount |
   |---|---:|
   | QK, WG0 | 36 WGMMA instructions; 4.72 MFLOP |
   | PV, WG0 | 4 WGMMA instructions; 2.10 MFLOP |
   | PV, WG1 | 4 WGMMA instructions; 2.10 MFLOP |
   | Unique fp8 KV record bytes | 41,984 B |
   | Dequantized K written to shared memory | 73,728 B |
   | FP8 elements converted | 32,768 |
   | BF16-pair scale multiplications | 16,384 |
   | Softmax probability exponentials | 4,096, plus running-max rescaling |

   One correction to the load accounting: four threads cooperate on each key, and each loads the same 16-byte scale vector. Thus the source issues **45,056 bytes of load payload** per block: 32,768 nope + 8,192 rope + 4,096 scale bytes. Cache/coalescing can eliminate much of that scale duplication downstream.

   Each producer thread executes two rounds, each containing 11 vector loads and 18 vector shared stores. It converts 256 FP8 elements across both rounds. This is substantial instruction work, independently of HBM bandwidth.

   Using your achieved dense throughput only as a reference, \(630/132=4.77\) TFLOP/s per SM gives throughput-equivalent times of approximately:

   | Tensor work | Reference time |
   |---|---:|
   | QK | 0.99 µs |
   | Each half-PV | 0.44 µs |
   | All tensor work | 1.87 µs |

   These are **not predictions for these WGMMA shapes**, nor rigorous latency bounds. They show that tensor work already occupies a substantial fraction of 4.3 µs before softmax, accumulator rescaling, fences, and synchronization. Being below dense throughput does not rule out a consumer bottleneck.

   WG0 waits for QK, performs softmax and rescales its output, issues PV, and waits for PV before the next iteration. WG1 overlaps its PV with part of this work, but both warpgroups share the SM’s tensor resources. The single shared P/scale buffer also creates a dependency: WG0 cannot overwrite it until WG1 finishes using it.

   Meanwhile, the producer overlaps work on the other K buffer. For cluster size one, a buffer becomes reusable after all 256 consumer threads release it. The producer prefetches indices and loads scales before the availability wait, but the main FP8 loads and conversions occur afterward. This is double buffering, not a deep global-memory prefetch pipeline.

   Therefore, the useful first model is approximately

   \[
   t_{\rm block}\approx \max(P,C)+\text{pipeline/synchronization effects},
   \]

   not \(P+C\).

   **My inference:** consumer time is likely a substantial floor, and a purely producer-dominated interpretation is not justified. Exact producer and consumer microseconds cannot be extracted from this source. Quoting a narrow range would invent information that requires measurement.

   The HBM estimate also needs qualification. At 128 CTAs, 42 KB per block corresponds to 5.37 MB and about 1.49 µs at 3.6 TB/s—**if every record misses cache**. Overlapping tokens can already share data through L2, and padding repeatedly addresses the same record. Four gathers do not imply four HBM fetches. Packing reliably removes repeated load instructions and conversion work; its HBM savings may be much smaller.

   The fitted 13 µs intercept is not a measured startup phase. At 128 CTAs, Q loads total 9.44 MB and split-output stores total 16.78 MB. Their combined HBM-throughput equivalent is 7.28 µs, although actual traffic may involve L2 and overlap. Pipeline startup/drain, output rescaling and staging, barriers, descriptor handling, and launch overhead plausibly explain the rest. The fit changes configurations, so it cannot attribute those costs individually.

   **Choose experiment (b): replace gathered/dequantized K with a finite constant fill, retaining the shared stores, masks, barriers, GEMMs, resources, and launch geometry.** Compare slopes over several block counts at the same CTA count and split count; avoid interpreting only a one-block timing.

   If the slope barely changes, expensive packing has little opportunity: consumption or the retained store/handshake work dominates. A large reduction identifies exposed producer cost or producer interference with consumers. For a continuous \(r=1.3\), an ablated slope still above \(4.3/1.3=3.31\) µs already rules out a win under an optimistic model that makes B’s producer free. Block rounding can make that threshold stricter.

   Experiment (a) is useful corroboration, but **the same `s_q` is not a controlled comparison**: changing 64 to 128 heads halves the planner’s split count and doubles per-CTA key work. Forcing the old split count instead doubles CTAs. A cleaner cluster-2 comparison uses 32 tokens × 128 heads versus 64 tokens × 64 heads, with matched indices and two splits.

   Removing GEMMs radically changes buffer lifetimes and resource pressure. NCU is valuable afterward, particularly for distinguishing waits on K readiness from producer waits on K availability and consumer WGMMA/P dependencies; stall percentages alone are not a producer-time fraction.

2. **A does not fit as proposed; B fits the existing per-CTA storage model but requires more than generalized peer stores.**

   The current register budgets sum to

   \[
   128(192+160+152)=64{,}512
   \]

   32-bit registers, against a 65,536-register SM.

   Output accumulators alone require:

   | Rows held concurrently | FP32 output registers |
   |---|---:|
   | 64 | 32,768 |
   | 128 | 65,536 |
   | 256 | 131,072 |

   Thus M=128 consumes the entire register file for output alone. More warpgroups redistribute those registers; they do not create capacity.

   Shared memory independently excludes a straightforward M=128 extension: Q would consume 144 KiB, the existing two K buffers 144 KiB, and P 16 KiB—304 KiB before metadata. M=256’s Q alone is 288 KiB. Serializing tokens, spilling accumulators, or changing output tiling would constitute a substantially different pipeline.

   For B, per union block each CTA would produce 18,432 bytes locally and send 55,296 bytes to peers. Across the cluster, that is **216 KiB of remote traffic**, in addition to 72 KiB of local writes. Every destination still receives a full 72 KiB K tile.

   At 4.3 µs/block this is about 12.9 GB/s of outgoing remote data per CTA, or 1.65 TB/s summed across 32 clusters. These are required rates, not a claim about achievable DSMEM bandwidth. The source supplies no basis for asserting that the broadcast is cheaper than the eliminated conversions.

   More importantly, `/4` key work does not automatically mean `/4` producer latency. Today’s mapping handles 32 keys per round using four threads per key. Handling 16 keys with that mapping leaves half the producer threads inactive. Using all 128 threads requires a new dimension assignment and attention to redundant scale loads.

   The barrier protocol must also change precisely. The existing cluster-2 path initializes `bar_k_avail` with **four arrivals**, representing two consumer warpgroups from each of two CTAs. A corresponding cluster-4 protocol needs eight release notifications per destination buffer, assuming one elected notification per consumer warpgroup per CTA.

   Each destination remote-ready barrier could expect 55,296 incoming bytes with one local `arrive_and_expect_tx`, or use separate per-peer barriers. Its arrival count and transaction-byte count are different quantities. Every remote store must complete the correct destination barrier and phase.

   Other required changes include replacing the XOR-based single-peer addressing with rank-aware addressing, ensuring all producers observe safe buffer reuse, maintaining asynchronous shared-memory visibility, and ensuring no CTA exits while peers can still access its shared memory. A locally all-masked block cannot simply bypass this protocol.

3. **132 SMs do not establish that 32 cluster-4s can reside simultaneously.**

   With one CTA per SM, a topology-level capacity bound is

   \[
   C_{\rm resident}\leq
   \sum_{\rm GPC}\left\lfloor\frac{S_{\rm GPC}}4\right\rfloor,
   \]

   before any additional placement/resource restrictions.

   For illustration, both of these hypothetical eight-GPC distributions total 132 SMs:

   | SMs per GPC | Cluster-4 capacity |
   |---|---:|
   | 18, 18, 16, 16, 16, 16, 16, 16 | 32 clusters / 128 SMs |
   | 18, 18, 18, 18, 18, 18, 12, 12 | 30 clusters / 120 SMs |

   The actual distribution is not supplied, so an exact residency answer would be speculation. The occupancy query with the actual kernel configuration is essential.

   This is a sharp threshold: your launch needs 32 clusters. If only 30 fit, two clusters start after an earlier cluster finishes. With similar cluster durations, this creates a substantial second-wave tail—not merely a \(32/30\) slowdown. Reducing split counts to fit introduces longer-running clusters and needs a separate scheduling analysis.

   **I would make residency of 32 clusters a prerequisite for the straightforward B design.**

4. **There are two useful gain formulas; only one represents the existing overlap.**

   If \(f\) means an *additive, exposed, reducible producer fraction*, the optimistic Amdahl-style model is

   \[
   q\equiv\frac{\text{packed variable time}}{\text{original variable time}}
     =r\left[(1-f)+\frac f4\right]
     =r(1-0.75f).
   \]

   Ignoring overhead, its break-even ratio is

   \[
   r_{\rm break}=\frac1{1-0.75f}.
   \]

   | \(f\) | Break-even \(r\) | Variable-time ratio at \(r=1.3\) |
   |---:|---:|---:|
   | 0.25 | 1.23 | 1.056 |
   | 0.50 | 1.60 | 0.813 |
   | 0.75 | 2.29 | 0.569 |

   For \(r=1.3\), this requires \(f>30.8\%\), before broadcast, synchronization, planning, or residency costs.

   But if \(P,C\) are overlapping producer and consumer service times, the more appropriate idealization is

   \[
   q=r\frac{\max(P/4,C)}{\max(P,C)}.
   \]

   If \(C\geq P\), packing cannot improve steady-state block time for \(r\geq1\). If \(P>C\),

   \[
   r_{\rm break}=\min(4,P/C).
   \]

   Thus \(r=1.3\) requires the original producer service time to exceed consumer time by at least 30%, even before additional overhead.

   A practical extension is

   \[
   T_B\approx F_B+
   N_U\{\max(P/4+D_{\rm producer},C)+H\}
   +T_{\rm combine},
   \]

   with a separate scheduling penalty if clusters need another wave. Here \(N_U\) is the actual critical split’s number of union blocks, not a fractional \(rN\).

   **Block rounding materially changes your example.** For 512 live keys:

   | Case | Total blocks | Longest of two balanced splits |
   |---|---:|---:|
   | Own live set | 8 | 4 |
   | Union, \(r=1.2\), about 614 keys | 10 | 5 |
   | Union, \(r=1.3\), about 666 keys | 11 | 6 |
   | Current width 768 | 12 | 6 |

   Consequently, \(r=1.3\) can mean an effective critical-path expansion of **1.5× against early stop**, while retaining today’s six-block consumer path. Under the overlapping model, that requires \(P/C>1.5\), not merely 1.3.

   For scale, using your ideal four-block early-stop baseline gives \(13+17.2+7.7=37.9\) µs/layer. The additive model at \(r=1.3\) predicts incremental packing savings of approximately:

   | Assumed \(f\) | Saving over early stop, 78 layers |
   |---:|---:|
   | 0.25 | −0.08 ms |
   | 0.50 | +0.25 ms |
   | 0.75 | +0.58 ms |

   These omit rounding and all new overhead. **My planning estimate would be zero to a few tenths of a millisecond beyond early stop, with a credible regression risk.** This is a judgment, not a measurement-derived forecast. A strongly producer-bound ablation could justify revising it upward.

   Measure the distribution of union block counts and slowest split lengths across sequences, not just mean \(r\). Reusing an index set across layers says nothing by itself about overlap between the four query rows.

5. **Early stop has the best expected return; the remaining alternatives depend on the ablation.**

   My ordering by expected gain per implementation effort is:

   | Priority | Change | Assessment |
   |---:|---|---|
   | 1 | Balanced live-length planning / early stop | Removes both producer and consumer work |
   | 2 | Targeted producer prefetch or instruction improvements | Attractive if the ablation shows exposed producer cost |
   | 3 | Overlap next QK with outstanding PV | Plausible consumer improvement, but needs careful scheduling |
   | 4 | Union dequantization scratch prototype | Avoids cluster placement risk; adds a pass and substantial traffic |
   | 5 | Cluster packing | High complexity, conditional benefit |
   | — | Third full K buffer | Does not fit current shared memory |
   | — | One split at c16 | Cheap experiment, unfavorable current model |

   Early stop needs **repartitioning**, not just terminating each existing split at its live length. Today’s split boundary is at 384 keys. With 512 live keys, clipping produces six blocks in split zero and two in split one; the main-kernel critical path remains six blocks. Rebalancing gives four plus four.

   Your 0.67 ms gross estimate corresponds exactly to saving two blocks:
   \(78\times2\times4.3\) µs. It is reasonable as an idealized target, but rows above 512 require nine or ten blocks, giving a five-block longest split under simple per-row splitting. Planner behavior and length tails determine the realized gain.

   A third full K buffer adds 72 KiB to an already roughly 226 KiB allocation. It does not fit. A deeper producer pipeline would need register prefetching, smaller tiles, or a storage redesign; the register budget also limits prefetch depth.

   WG0’s next QK could potentially be issued before waiting for its current PV, because its score accumulator and output accumulator are distinct. However, P operands must remain valid until PV completes, output rescaling must wait for the old PV, and K-buffer releases must remain correct. WG1’s PV also competes for tensor execution. This may reduce bubbles; it does not remove the tensor work.

   The scratch idea has an important TMA limitation: **a contiguous union buffer does not make each token’s selected rows a contiguous TMA tile.** Loading contiguous union tiles retains the extra masked GEMMs. Keeping each token’s original key count requires BF16 gathers or scattering dequantized records into separate per-token contiguous buffers.

   At 700 union keys, your scratch is 12.9 MB. Writing it once and reading it four times moves about 64.5 MB of BF16 scratch data per layer, before the original FP8 reads. Fitting the allocation in L2 does not make this traffic free, and Q/output buffers compete for capacity.

   Also, union indices can be reused across layers, but **dequantized KV cannot**: each layer has different KV contents. The dequantization pre-pass runs 78 times per step, not 21.

   Finally, with unchanged 4.3 µs block time, one split predicts:

   | Case | Two splits, including combine | One split |
   |---|---:|---:|
   | Width 768 | 46.5 µs measured | 64.6 µs modeled |
   | Eight live blocks | 37.9 µs modeled | 47.4 µs modeled |

   With equal fixed costs, one-split block time must fall below approximately 2.79 µs at width 768 or 3.11 µs after early stop. Direct BF16 output reduces epilogue traffic, so this is worth a cheap measurement, but the present model favors two splits.

6. **All-masked blocks are already safe in this softmax, including in the middle.**

   `cur_max` becomes \(-\infty\), but `rM` starts at finite `MAX_INIT_VAL = -1e30`. Therefore, assuming the normal positive finite softmax scale:

   \[
   rM_{\rm new}=\max(-\infty,rM_{\rm old})=rM_{\rm old},
   \qquad
   \text{scale}_{\rm old}=\exp_2(0)=1.
   \]

   Every masked probability becomes zero, `rL` remains unchanged, and both output accumulators are multiplied by one. With finite V, PV adds zero. This works before the first valid block, between valid blocks, and at the tail.

   An entirely empty split produces zero output and split LSE \(-\infty\), as shown. The combine source is absent, so its empty-split handling still needs verification. The no-split path writes **positive infinity** for an empty row, apparently as an existing sentinel; preserve and verify that convention with the external DCP merge.

   There are two other correctness conditions worth elevating:

   **Deduplication must preserve per-token multiplicity.** If an original row contains duplicate valid indices, a Boolean membership mask changes their weight. Verify uniqueness rather than assuming it.

   **Reordering differences are not strictly limited to FP32 accumulation order.** This kernel rounds probabilities to BF16 before PV, relative to the running maximum at that block. Reordering or repartitioning can change those BF16 operands, as well as FP32 rescaling and accumulation order. Your accepted split changes already exercise this issue, but their measured tolerance is not a guarantee for union ordering.

   The packing transformation preserves the selected attention terms and operand precisions under the uniqueness assumption. It still needs numerical validation against your accepted tolerance. A scratch implementation must also reproduce the exact existing dequantization sequence: FP32 scale → BF16 scale, FP8 conversion to BF16, then BF16 multiplication.

7. **The largest integration obstacle is the scheduler’s meaning of a request, rather than either TMA descriptor.**

   The supplied scheduler metadata is indexed only by `partition_idx`; its request ranges refer to `b`. Here `b=1`, while all 64 query tokens occupy `s_q`. MODEL1’s length lookup is likewise `topk_length[batch_idx]`, not per `s_q` row.

   Therefore, “MODEL1 already supports lengths” does not directly provide per-token or per-sequence union planning for this call shape. V32 also explicitly rejects lengths. You need a new metadata dimension or a logical layout that makes sequence groups scheduler requests.

   For B, all four cluster members must receive identical union block ranges and execute compatible request loops. Independent per-token planning would break the collective pipeline.

   Cluster rank must also be separated from head-block index. Currently `blockIdx.x` controls Q head selection, output head offsets, attention-sink indexing, and peer rank. In B, cluster rank selects the token, while the head-block index remains zero. Simply increasing `NUM_M_BLOCKS` would address the wrong heads.

   Once those coordinates are separated, Q can still load a 64×576 tile from the correct token using TMA. The 5D O descriptor can likewise store to the correct token coordinate. If four tokens are contiguous, a logical `(b=sequences, s_q=4)` view may help without copying Q, but planner geometry and split-output indexing must be adapted consistently.

   PDL is not an available saving in this source: dependency synchronization and the launch attribute are disabled, even though the trigger call remains. Do not assume combine overlap in the performance model.

The strongest implementation case would be: **32 resident cluster-4s, union lengths usually at most ten blocks, and a constant-fill experiment showing a large reduction in block slope.** Without those results, balanced early stop is the better investment.

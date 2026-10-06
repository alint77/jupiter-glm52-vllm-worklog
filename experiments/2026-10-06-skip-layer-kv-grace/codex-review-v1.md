# Codex review of the v1 plan (gpt-6-astra, effort xhigh, 2026-10-06)

Inline-prompt review (Codex's sandbox cannot run on login nodes); prompt = plan v1 + code excerpts with file:line. Verbatim:

The single-buffer design is feasible, and the projected HBM saving is credible. The plan nevertheless has three correctness gaps that should block implementation as currently specified.

1. **[P1] `R = 16384` does not cover the reserved-row requirement.**  
   The selected union can contain 16,384 distinct **historical** rows, with none of this step’s tokens selected. Eight separately reserved, always-written rows then require 16,392 rows, or **16,448 after rounding to block 64**. Even reserving only the two locally owned current tokens exceeds 16,384 in that case.

   The plan’s “no overflow path” claim therefore needs a capacity invariant that includes reserved rows. Measuring overlap cannot establish that invariant; choosing R from Gate 0a’s percentiles would also conflict with having no overflow path. The cache uses direct row indices and 656-byte rows ([flash_mla_interface.py:90](/e/project1/profound/alint77/vllm/vllm/third_party/flashmla/flash_mla_interface.py:90), [flash_mla_interface.py:100](/e/project1/profound/alint77/vllm/vllm/third_party/flashmla/flash_mla_interface.py:100)).

2. **[P1] Waiting after the reserved-row write leaves its own dependencies unspecified.**  
   Step 2 writes reserved rows **before** waiting for `e_g`. That is safe only if their addressing is already available independently of the side-stream planner, and the planner/gather never writes those destinations. If reserved positions or their slot mapping depend on the side-stream result, the write can race with planning. Waiting immediately before attention does not repair an earlier incorrect write.

   Current-step tokens must be treated as selectable: indexer cache insertion precedes logits/top-k, and native speculative decoding supplies per-query context lengths ([sparse_attn_indexer.py:380](/e/project1/profound/alint77/vllm/vllm/model_executor/layers/sparse_attn_indexer.py:380), [sparse_attn_indexer.py:538](/e/project1/profound/alint77/vllm/vllm/model_executor/layers/sparse_attn_indexer.py:538)). Thus later verify queries can require earlier tokens from the same step. The plan correctly recognizes that these skip-layer KVs cannot be prefetched. Its gather must exclude their physical slots even when those slots contain plausible bytes from a previous rejected verification. The exact self-token eligibility depends on context-length construction, which is not shown.

3. **[P1] The named KV-write integration point covers only the direct-call path.**  
   The proposed hook at `mla_attention.py:611–620` is inside `if self.use_direct_call`. The alternative path invokes `unified_mla_kv_cache_update`, whose implementation independently calls `do_kv_cache_update` ([mla_attention.py:598](/e/project1/profound/alint77/vllm/vllm/model_executor/layers/attention/mla_attention.py:598), [mla_attention.py:636](/e/project1/profound/alint77/vllm/vllm/model_executor/layers/attention/mla_attention.py:636), [mla_attention.py:1111](/e/project1/profound/alint77/vllm/vllm/model_executor/layers/attention/mla_attention.py:1111)).

   An extra write implemented only at the cited location would leave reserved rows unwritten on the custom-op path. That path also deliberately carries a dependency token to preserve ordering through compilation. The excerpts do not establish which path production selects; standalone kernel tests cannot close this integration gap.

4. **[P2] “Once per anchor” is conditional capture-time reuse, not a Python cache operation performed during replay.**  
   Anchors always execute conversion because only `_reuses_topk` layers can take the cache hit. During a full-forward capture, matching skip layers capture consumers of that anchor’s converted tensors. Replay reruns the recorded conversion kernels and consumers; it does **not** consult the dictionary. The dictionary ending with the last anchor’s entry therefore does **not inherently make full-graph replay stale** ([flashmla_sparse.py:939](/e/project1/profound/alint77/vllm/vllm/v1/attention/backends/mla/flashmla_sparse.py:939), [flashmla_sparse.py:966](/e/project1/profound/alint77/vllm/vllm/v1/attention/backends/mla/flashmla_sparse.py:966)).

   However, once-per-anchor reuse requires matching keys, including block-table and request-ID pointers. Pointer identity provides no freshness guarantee for a skip-only or separately captured invocation ([flashmla_sparse.py:925](/e/project1/profound/alint77/vllm/vllm/v1/attention/backends/mla/flashmla_sparse.py:925)). Python dictionary updates, Python ring-position increments, or host decisions based on `U_g` cannot supply changing per-replay behavior.

   Likewise, static buffers and events alone do not establish graph correctness. The captured dependency chain must include conversion → side-stream planning/gather → skip attention, with the side stream joined into the capture. Neither the graph wrapper nor the existing side-stream implementations is supplied, so their compatibility remains unverified.

5. **[P2] DCP and speculative/prefix lifecycle coverage is too weak for the proposed exactness claim.**  
   Converted decode indices identify **physical slots in this rank’s allocation**, not positions in a contiguous request-local sequence. Prefill conversion uses a different namespace: contiguous workspace offsets. The conversion also produces compacted valid prefixes and `-1` tails ([sparse_utils.py:297](/e/project1/profound/alint77/vllm/vllm/v1/attention/backends/mla/sparse_utils.py:297)). Consequently, the bitmap bound must cover the allocated physical slot space, including arbitrary block IDs; current-token matching must agree with the actual KV-write slot mapping and ownership.

   V1’s fresh gather each step is favorable here: it avoids persistent staging-cache invalidation. Rejected KV bytes also need not necessarily be erased, provided sequence bounds exclude them and replacement tokens overwrite their slots. But “same tensor shape” does not prove those conditions or prefix-cache publication and recycling behavior.

   The gates lack explicit coverage of repeated graph replay with changing metadata contents at unchanged addresses, every acceptance length, rejected-slot reuse, block-boundary crossings, prefix hits and recycled blocks, shortened/padded verifies, and ranks with no selected rows. The latter also requires correct LSE handling, not just matching local attention outputs ([flashmla_sparse.py:885](/e/project1/profound/alint77/vllm/vllm/v1/attention/backends/mla/flashmla_sparse.py:885)). Greedy fixed-prompt decoding and GSM8K do not establish these properties.

6. **[P2] The prefill description misses capture-dependent dispatch.**  
   `dcp_sparse_prefill` is selected only when the current stream is **not capturing**, along with the other conditions ([flashmla_sparse.py:849](/e/project1/profound/alint77/vllm/vllm/v1/attention/backends/mla/flashmla_sparse.py:849)). If this attention call executes inside a captured prefill piece, capture records the mixed FP8 path; replay does not reconsider the Python branch. Therefore eligible small prefill captures can read Grace through sparse FP8 decode rather than through whole-shard upconversion.

   The proposed hook also encounters `_dcp_converted_indices(..., prefill=True)`, whose results address the BF16 workspace, not Grace physical slots ([flashmla_sparse.py:897](/e/project1/profound/alint77/vllm/vllm/v1/attention/backends/mla/flashmla_sparse.py:897)). An eight-query staging path cannot be selected merely because execution is captured or conversion occurred.

   For eager eligible prefill, the stated whole-shard read is supported by the code ([flashmla_sparse.py:278](/e/project1/profound/alint77/vllm/vllm/v1/attention/backends/mla/flashmla_sparse.py:278)). The quoted 2–10 ms is a bandwidth estimate, however: the supplied measurements do not establish the performance of this particular upconversion kernel on Grace or the captured fallback path.

7. **[P2] “Same bytes and same kernel” does not establish end-to-end bit exactness.**  
   Sorting the storage union is compatible with exactness **if remapping preserves each query’s original entry order, multiplicity, and `-1` positions**. Sorting or deduplicating the attention lists themselves is a different operation.

   Moreover, existing DCP compaction explicitly has unspecified prefix order and can use atomic allocation ([sparse_utils.py:304](/e/project1/profound/alint77/vllm/vllm/v1/attention/backends/mla/sparse_utils.py:304)). Independent baseline and treatment runs therefore do not automatically receive identical ordered inputs; floating-point reduction order can matter. Gate 1’s comparison using the same converted indices is valuable, but narrower than the production bit-exactness claim.

   `extra_k_cache` is also **additive**, not a transparent address substitution ([flash_mla_interface.py:94](/e/project1/profound/alint77/vllm/vllm/third_party/flashmla/flash_mla_interface.py:94)). Adding all current tokens would introduce unselected or future tokens, while retaining their original entries could double-count them. Sparse attention supplies no causal mask to repair that selection. Scheduler reuse has explicit compatibility conditions, and the current backend wrapper does not pass the extra-cache arguments ([flash_mla_interface.py:85](/e/project1/profound/alint77/vllm/vllm/third_party/flashmla/flash_mla_interface.py:85), [flashmla_sparse.py:991](/e/project1/profound/alint77/vllm/vllm/v1/attention/backends/mla/flashmla_sparse.py:991)). SM90 support and hot-versus-cold expert numerical equivalence cannot be decided from these excerpts.

8. **[P2] Copy cost and available overlap are overstated in the favorable direction.**  
   Using the supplied 361–384 GB/s gather measurements:

   | Per-rank union U | Traffic across 57 layers | Link time |
   |---|---:|---:|
   | 1,000 | 37.392 MB | 0.097–0.104 ms |
   | 16,384 | 612.631 MB | 1.60–1.70 ms |

   Thus **0.1 ms is not the worst case**. At U=1,000, copying all three layers costs approximately **5.1–5.5 μs per group**, before planning and other overhead, rather than 2 μs. Even 420 GB/s gives about 1.46 ms for the stated worst-case step.

   Conversion occurs inside anchor attention, immediately before its FlashMLA call—not at the beginning of the anchor layer ([flashmla_sparse.py:859](/e/project1/profound/alint77/vllm/vllm/v1/attention/backends/mla/flashmla_sparse.py:859)). A full layer of lead time is therefore not established.

   Gate 4’s “no main-stream waits” criterion is also imprecise: dependency waits are required; the desired result is no exposed delay. Copies can slow cold-expert transfers or other work even when `e_g` is ready before attention. A/B with an enlarged hot set can mask such overhead. Gate 0b’s mean expert-count estimate does not establish the critical-rank latency saving, and a few hundred samples per group cannot characterize p99.99 reliably.

9. **[P2] Allocation accounting needs invariants beyond the new tier label.**  
   The proposed V2 allocation and per-layer classification locations are appropriate. However, V1’s implementation includes both packed-layout rejection and retained `GraceAllocation` owners ([gpu_model_runner.py:7224](/e/project1/profound/alint77/vllm/vllm/v1/worker/gpu_model_runner.py:7224), [gpu_model_runner.py:7262](/e/project1/profound/alint77/vllm/vllm/v1/worker/gpu_model_runner.py:7262)); V2 currently returns raw tensors ([attn_utils.py:183](/e/project1/profound/alint77/vllm/vllm/v1/worker/gpu/attn_utils.py:183)). Lifetime and layout compatibility are part of the port’s correctness.

   Also, the existing “available memory” calculation is **cross-tier capacity accounting** used to derive block count ([tiered_moe_kv.py:208](/e/project1/profound/alint77/vllm/vllm/model_executor/model_loader/tiered_moe_kv.py:208), [kv_cache_utils.py:1395](/e/project1/profound/alint77/vllm/vllm/v1/core/kv_cache_utils.py:1395)). Freed HBM cannot simply be subtracted from that quantity without affecting capacity. The allocation gate needs to establish preserved block capacity alongside actual HBM savings, including staging scratch and graph allocations.

The **21 anchors, 57 skip layers, and 19 groups are correct** under the supplied configuration ([deepseek_v2.py:1106](/e/project1/profound/alint77/vllm/vllm/model_executor/models/deepseek_v2.py:1106)). One shared set is sufficient under the stated sequential execution: a properly ordered fork after anchor conversion occurs after the preceding group’s readers, including across steps. Double buffering is not inherently required.

The memory arithmetic is also sound: approximately **3.486 GiB** moves to Grace; three 16,384-row buffers consume **30.75 MiB**, leaving about **3.456 GiB**, or **174 experts** before additional overhead. “~170” is reasonable. If cold experts fall from 2.00 to 1.88 across 75 MoE layers, the linear estimate is **0.459 ms**, about **1.9%** of 24.5 ms. The unproven part is that residency improvement and its realized latency benefit, not the basic arithmetic.

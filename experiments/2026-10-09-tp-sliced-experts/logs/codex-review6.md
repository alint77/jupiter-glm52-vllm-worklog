The TP decomposition is sound. I would make **a correct TP4 tiered Marlin baseline the first milestone**, then integrate td_v27 and shared fusion. Several assumptions in the plan need to become explicit contracts before served A/B testing.

I did not run shell commands. This review uses your excerpts plus upstream documentation; the omitted runner and td_v27 implementation still need verification against your exact revision.

1. **TP ownership and reduction semantics are the main correctness prerequisite.**

   For identical tokens and routes on all TP ranks, the intended result is:

   \[
   y=\sum_{r=0}^{3}\left[
   \mathrm{rsf}\sum_{e\in\mathrm{topk}}p_e
   W_{2,e,r}\,\mathrm{SiLU}(W_{g,e,r}x)\odot(W_{u,e,r}x)
   +S_r(x)\right].
   \]

   Here each down projection consumes its matching intermediate slice. There is **no division by TP size**, and every selected expert must contribute on every rank.

   Consequently, verify these independently:
   - Each rank receives the same token rows, in the same order, with the same global expert IDs and routing weights.
   - EP ownership masks, replica assignment, redundant physical expert IDs, and balancer remapping cannot discard or redirect TP contributions.
   - Both execution paths return **unreduced rank-local partials**. The runner, enclosing model, or sequence-parallel machinery performs the required reduction exactly once.
   - The slice index comes from the correct TP-group rank, not global rank or an EP/planner rank.

   Using `ep_size=1` for expert inventory is reasonable. Keep the actual TP rank, TP size, and slice coordinates in the load plan; “plan one rank” must not accidentally give all four workers slice zero.

   The claim that “the 75 all-reduces are the same” is conditional. EP can use different dispatch/combine arrangements, and upstream distinguishes already-reduced outputs from outputs requiring a later reduction. Audit `reduce_results`, `output_is_reduced()`, and the enclosing model’s reduction path in your revision. [Upstream runner](https://docs.vllm.ai/en/v0.24.0/api/vllm/model_executor/layers/fused_moe/runner/moe_runner/)

   DCP, sequence parallelism, and speculative paths belong in this audit: matching tensor shapes alone does not establish matching token rows.

2. **Shared fusion needs an execution contract with the runner, not just a skip flag.**

   Your inverse-scale algebra is correct **when the runner really multiplies the combined output by `rsf`**:

   | Scaling location | Routed contribution entering the combined output | Required shared coefficient |
   |---|---:|---:|
   | Runner scales output | \(R_r\) | \(1/\mathrm{rsf}\) |
   | Router already scales `topk_weights` | \(\mathrm{rsf}R_r\) | \(1\) |

   The excerpt passes `apply_routed_scale_to_output`, but does not establish its value. Upstream explicitly routes the scaling factor to either the router or runner. Derive the coefficient from the effective execution mode, not just the model configuration’s `routed_scaling_factor`. [Upstream factory](https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/fused_moe/layer/)

   Shared execution must be suppressed **before it is scheduled**. Your existing predicate excludes `MK_INTERNAL_OVERLAPPED`, but permits other orders. A shared expert may already have been queued on an auxiliary stream before `apply_tiered_moe` returns. Upstream has precisely such an asynchronous scheduling path. [Shared-expert scheduling](https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/fused_moe/runner/shared_experts/)

   Select an immutable per-call execution mode early enough to determine:
   - Which kernel runs and whether it computes shared output.
   - Whether the shared module is scheduled, waited on, and added.
   - The runner/custom-op return contract and reduction behavior.

   A runner initialized to expect a separate shared tensor may also require changes to its output schema or assertions. Avoid mutating `shared_experts=None` around individual calls.

   Three additional gates are missing:
   - `quant_config` is passed to the shared MLP. Its actual loaded weights are not proven to be dense BF16.
   - The proposed dimensions assume `n_shared_experts == 1` and ordinary TP slicing.
   - `shared_experts_input` must match the kernel’s shared input, or be passed separately.

   Explicitly handle the existing built-in shared-fusion setting: your initializer can set `self.shared_experts=None` because fusion is already enabled. Restrict the initial fused path to the verified BF16 configuration and a finite, nonzero output scale.

3. **The packed slices look correct, but the benchmark does not validate the production conversion.**

   Assuming the stated compressed-tensors layout—weights `[N, K/8]`, scales `[N, K/group_size]`—your offsets are correct for group 32. After slicing, transposition, and gate/up concatenation, staging should be:

   | Tensor | Expected shape |
   |---|---:|
   | `w13` | `[1, 768, 1024]` |
   | `w2` | `[1, 64, 6144]` |
   | `w13_scale` | `[1, 192, 1024]` |
   | `w2_scale` | `[1, 16, 6144]` |

   Important implementation details:
   - Slice gate and up independently, then concatenate. Slicing an already-fused gate/up tensor as one contiguous intermediate block selects the wrong channels.
   - Down scale offsets are `512*r/group_size`, with length `512/group_size`. The hard-coded `16` only covers group 32.
   - Slice the checkpoint representation **before Marlin repacking**. A slice of a repacked full expert is not established to be the repack of the slice.
   - Make sliced staging tensors contiguous and verify packed word/nibble ordering against the actual checkpoint.
   - Replace full-expert `weight_shape` metadata using its expected orientation. The current compressed-tensors branch copies the original checkpoint shape.
   - Update the hard-coded down `g_idx` length and AutoRound shapes. AutoRound’s `qweight`/`scales` use different axis conventions, so they need their own slicing implementation or an explicit rejection.

   Passing a layer with intermediate size 512 is necessary, but inspect every dimension used by repacking and scale permutation. Upstream’s conversion performs both operations and treats grouped act-order specially. [Marlin conversion](https://docs.vllm.ai/en/v0.24.0/api/vllm/model_executor/layers/fused_moe/oracle/int_wna16/)

   The existing `torch.empty` g-index tensors are safe only if the non-grouped converter discards their contents; do not accidentally introduce a path that reads them.

   Add a conversion test using **real packed checkpoint tensors and the production converter**, covering all four ranks. Compare dequantized slices and Marlin execution against the corresponding full-expert reference. The unpacked-code result of `2.2e-3` supports the kernel arithmetic, but does not prove serialized packing, metadata, or production scale permutation.

4. **Dispatch and Marlin fallback need explicit layout isolation.**

   Give `tp_sliced` its own capability predicate. The current decode predicate does not establish all td_v27 requirements: hidden size, intermediate size, expert count, top-k, tensor layouts, route capacity, shared-weight format, and workspace requirements.

   In particular:
   - Do not globally change the existing `MAX_TOKENS` from 8 to 32. The EP kernel retains its own limits.
   - If sliced decode is ineligible, fall directly to the verified sliced Marlin path. Prevent the generic EP decode or WGMMA path from claiming the call.
   - In the current expression, `decode_placement(...)` is evaluated **before** the `replica_assignment == "off"` alternative. Disabling replicas does not itself bypass that function.
   - Audit unconditional `prepare_replica_routing` and related cleanup calls for assumptions about EP state.

   For Marlin, distinguish **256 logical expert IDs** from the hot or cold tier’s physical slot count. Verify that EP-off execution still honors the supplied maps instead of indexing tier storage directly with global IDs.

   Each expert must appear in exactly one tier on each rank. Hot and cold outputs must combine without per-tier renormalization, dropped routes, duplicated contributions, or extra reductions. Test empty tiers, sparse maps, and IDs near 255.

5. **A ctypes launch can be captured; “host launch only” does not establish the complete integration.**

   CUDA stream capture records eligible CUDA work submitted through the launch API. ctypes itself is not an obstacle, and the Python/C launch wrapper does not execute again during replay. [CUDA stream capture](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html)

   Require the following:
   - Build, `dlopen`, symbol binding, device setup, kernel configuration, and warmup complete before capture.
   - Launch on the correct device’s current stream, with explicit pointer-width-safe ctypes argument types and checked return values.
   - Weight, map, workspace, output, and mapped-host allocations retain valid addresses and lifetimes.
   - Every necessary workspace reset is captured GPU work or happens inside the kernel. Host initialization performed once during capture will not reset counters on replay.
   - Route decisions depend on device tensor contents, not host-side state evaluated only during capture.
   - Concurrent executions cannot reuse the same mutable scratch unsafely; this includes DBO and multiple graph instances.

   Separately, keep the call inside an appropriate opaque PyTorch operator boundary, with accurate output and mutation semantics. If the existing MoE custom op already supplies this boundary, verify it rather than automatically adding another. Direct pointer-based external calls do not by themselves compose with `torch.compile`. [PyTorch custom operators](https://docs.pytorch.org/tutorials/advanced/custom_ops_landing_page.html)

   Start with `pdl_launch=False`. Validate PDL’s launch attributes and producer/consumer synchronization on the target CUDA/SM configuration in eager and captured execution before enabling it.

6. **Per-rank temporary files plus atomic rename prevent partial publication, not duplicate builds.**

   With the proposed scheme, all ranks can still compile simultaneously. That can be correct for identical artifacts, but it is not “build once.”

   Use a content-addressed cache key covering source and headers, compiler/toolkit, architecture, flags, and C ABI/layout version. Then either serialize builders with an interprocess lock and recheck the cache under the lock, or designate a builder for the relevant cache scope.

   Temporary names need uniqueness across jobs, processes, and restarts; rank alone is insufficient. Publish on the same filesystem, load only completed artifacts, and propagate build failure without leaving other ranks indefinitely waiting at a barrier.

   Distinguish **one cached build** from **per-process library loading and per-device initialization**. Do all of this during initialization, before profiling/capture. Any converted-weight cache also needs TP layout, TP size, and slice rank in its key.

7. **The planner needs clearer memory and profile semantics.**

   Quarter-sized expert slices do **not** quarter total routed-weight storage per rank:

   \[
   256(B/4)=64B.
   \]

   That equals the nominal EP4 storage for 64 full experts, before replicas and overhead. Savings come from eliminating replicas or changing residency—not simply from slicing.

   Budget per-expert metadata, routing buffers, Marlin scratch, shared weights, graph pools, conversion peaks, and KV cache explicitly. Some allocations depend on expert count rather than total weight bytes.

   A common hot set is sensible, but:
   - Its capacity must fit the least available rank if all ranks use the same plan.
   - An EP hot-set union is a useful seed, not sufficient information to choose promotions/demotions optimally. Use logical-expert demand counts with replicas deduplicated.
   - Identical residency is a policy choice; correctness only requires complete, valid local coverage.
   - UVA addressability does not prove “its own Grace.” Verify NUMA placement, allocation lifetime, and residency policy.

   Also separate checkpoint size, logical bytes read, and physical storage traffic. Whole-expert reads imply roughly four times the logical reads versus ownership-filtered EP, but caching and the loader determine actual I/O and startup time.

8. **The proposed validation jumps too quickly from a wrapper test to model quality.**

   “Coherent outputs” is a smoke test. Add deterministic layer-level and distributed checks before evaluating agentic quality:

   - Full-expert reference versus the sum of four production-converted slices.
   - TP Marlin versus td_v27 with identical inputs, routes, and weights.
   - Shared fusion on/off with non-unit `rsf`, including routed-only and shared-only contributions.
   - Hot-only, cold-only, mixed, and maximally concentrated routing.
   - Token counts around dispatch boundaries: 8/9 and 32/33, plus zero-token handling and graph padding.
   - Repeated graph replay with changing routes and poisoned workspace contents.
   - Alternation between fused decode and unfused fallback, checking for stale shared outputs or flags.

   Specify absolute and relative error metrics. TP changes floating-point accumulation order, so neither bitwise equality nor a single random-test maximum error is an adequate acceptance criterion.

I would reorder implementation as follows:

1. **Document and assert the supported configuration and reduction contract.** Pin the revision; identify scaling ownership, shared representation, token layout, and collective ownership.
2. **Implement packed slicing, metadata, and production repack tests.**
3. **Implement planner/loading and establish TP4 tiered Marlin correctness**, with shared execution left to the runner.
4. **Integrate td_v27 Phase A in eager mode**, including initialization-time build/load and explicit fallback.
5. **Validate compilation and CUDA-graph replay for Phase A**, initially without PDL.
6. **Implement shared fusion through the runner contract**, then repeat numerical, fallback-transition, and graph tests.
7. **Enable DCP/MTP/DFlash2 and overlap features individually**, with explicit unsupported combinations until validated.
8. **Run served comparisons among prod EP, TP Marlin, and TP td_v27.** Record actual token-count distributions, kernel/fallback usage, collective time, cold-memory traffic, TTFT, inter-token latency, and task quality. Concurrency `c` alone does not determine MoE step size, especially with speculation.

For the stated DCP4 decode path through `cp_all_gather_q`, commit 2 establishes the required collective ordering. The supplied waits also cover the proposed previous-anchor top-k WAR hazard.

I see two conditional branch hazards, a compiler-contract risk, and gaps in the mode/lifecycle guard. These are distinct from a demonstrated remaining failure in the stated configuration.

1. **P1 — The head-major query gather bypasses the new collective fence.**

   `vllm/model_executor/layers/attention/mla_attention.py:867` calls `get_dcp_group().all_gather(...)` without waiting. Only the alternative branch waits at `:870`.

   If a forked sparse layer can take the head-major branch and that gather uses the same one-shot communicator, this interleaving remains possible:

   - Indexer stream launches candidate gather I at `sparse_attn_indexer.py:121`.
   - Main launches head-major query gather Q at `mla_attention.py:867`.
   - Both kernels read the same local `_flag[b] = f`, calculate `f + 1`, and publish that generation.
   - On another rank, the corresponding arrivals can belong to different collectives. Equality of generations no longer identifies matching operations; early barriers or hangs follow.

   The relevant accesses are `csrc/custom_all_reduce.cuh:202–213`. Waiting later in `forward_mqa` cannot repair this.

   Move the gather wait before both query-gather alternatives. **Qualification:** the construction of `q_head_major` is absent, so these excerpts do not establish that your GLM run takes this branch.

2. **P1 — Tuple-query concatenation can overwrite the indexer’s shared scratch before joining.**

   `vllm/v1/attention/backends/mla/flashmla_sparse.py:1308–1312` obtains the workspace and writes concatenated Q **before** the indexer join at `:1318`. The indexer obtains its radix scratch at `sparse_attn_indexer.py:592–599`.

   Those requests alias: `vllm/v1/worker/workspace.py:108–115` starts each request at offset zero in the same workspace.

   A concrete failing execution, when tuple Q reaches this function without an intervening gather fence, is:

   - Indexer stream is still computing logits.
   - Main concatenates Q into workspace bytes W.
   - Indexer runs cooperative top-k using W as scratch.
   - Main waits for indexer completion.
   - Attention reads Q from W, now overwritten by radix scratch.

   Simultaneous writes are unnecessary; overlapping **lifetimes** suffice. The reverse timing can corrupt radix scratch instead.

   A DCP1 tuple-query path is an example: it skips `mla_attention.py:868–871`, and the feature’s eligibility check does not require DCP4.

   **For your stated DCP4 path, no issue between the shown radix scratch and a subsequent concat:** radix top-k precedes candidate gather, whose completion precedes the query gather through the new event. That fence already orders those uses.

   The robust fix is separate indexer scratch, or a join before every conflicting workspace use. Whether the overlapping projection kernels or StableTopK also use this manager cannot be determined from their omitted implementations.

3. **Potential P1 — The custom-op schemas do not fully describe the implementation’s effects.**

   There are two separate concerns.

   **Ordering:** declaring both operations as mutating `topk_indices_buffer` at `vllm/model_executor/layers/indexer_fork.py:142` and `:148` gives the compiler a mutation dependency between them. Having no tensor outputs does **not** by itself make these operations removable; ordinary side-effect-preserving compilation should retain their observable mutations.

   However, that declaration does not describe `_indexer_kw`, `_pending`, `_gathered`, or communicator state. Nor does it automatically order the operations against unrelated main-stream work.

   In particular, the wrapper’s attention call at `vllm/model_executor/layers/mla.py:250–255` does not visibly pass top-k. Its actual compiler boundary/schema is missing. If the compiler sees no dependency between `fork_q` and attention, a problematic generated order is:

   - Execute attention first.
   - Its `join_gather()` and `join()` see no pending event.
   - Attention reads the previous top-k.
   - Execute `fork_q` afterward.

   Moving only independent `q_b` work before the fork can merely reduce overlap. Moving the consumer before the fork is a correctness failure. Moving the query gather before `fork_q` does **not automatically recreate concurrent Signal access**: `fork_q`’s `side.wait_stream(main)` would then order the indexer after that gather. The consumer ordering is the critical distinction.

   **Mutated storage:** `_fork_q` accepts `topk_indices_buffer` at `indexer_fork.py:104`, but never passes it to the indexer at `:122`; the indexer writes its registered module buffer. If functionalization supplies a replacement buffer C, the declared operation should mutate C, but this implementation still writes the original buffer T. A subsequent materialization of C back into T could restore stale contents.

   These are concrete failure mechanisms, **not proof that this particular Inductor configuration performs either transformation**. Review generated code and make the mutated tensor an actual input to the implementation. Ensure the attention boundary exposes the required dependency and executes joins at runtime. If joins are traced while the fake implementations leave `_pending=None`, their conditional waits can disappear.

4. **P2 — The predicate identifies “capture with metadata,” not FULL decode capture.**

   `indexer_fork.py:52–55` checks stream capture, non-`None` metadata, and a truthy slot mapping. It does not check graph mode, decode participation, backend compatibility, or target versus drafter.

   A capture with those fields populated but `num_mqa_tokens == 0` can fork an indexer without reaching the decode consumer guarded by `mla_attention.py:776`. Then:

   - `_fork_q` sets `_pending`.
   - No `forward_mqa` clears it.
   - The next forked anchor fails `indexer_fork.py:114`, or the capture ends with an unjoined branch.

   Similarly, the predicate alone cannot prove that piecewise captures remain inline. That requires a runner invariant about their metadata.

   There is another eligibility invariant worth asserting: `_fork_k` executes before every wrapper branch at `mla.py:165`, but `_fork_indexer_q` appears only in the glue and Q-LoRA branches at `:185` and `:208`. The `q_lora_rank is None` branch at `:210` has no matching call, while the original indexer is suppressed at `:243`. This is not a demonstrated GLM configuration, but the selection guard itself does not exclude it.

The following areas are covered by the supplied ordering, subject to the compiler preserving those dependencies:

| Area | Assessment |
|---|---|
| **Named candidate/query collective pair** | **No issue on the `cp_all_gather_q` branch.** `gathered()` records on the indexer stream after candidate gather; main waits before launching query gather. StableTopK may overlap Q because it follows the recorded event. |
| **Current anchor’s top-k consumers** | **No issue.** Hot-cache conversion waits at `flashmla_sparse.py:1114`; main waits at `:1318`. Clearing the Python pointer after enqueueing the wait does not remove the GPU dependency. |
| **Previous shared layers reading top-k** | **No issue in the shown sequence.** The next anchor’s indexer stream waits for main at `indexer_fork.py:94`, and again at `:116`. All earlier main-stream reads precede those waits, including reads before an early top-k clear in the new indexer. |
| **Skip-KV side branch** | **No issue for the shown staging/consumption path.** Staging reads indices at `skip_kv_stage.py:253` and `:258`, then records readiness at `:286`. A shared layer waits for that readiness at `:299`; its completion precedes the next anchor’s fork wait. |
| **Allocator-managed tensor lifetimes** | **No missing `record_stream` is evident for the shown transfers.** Main-created inputs are recorded on the indexer stream at `indexer_fork.py:95` and `:117–118`. K/weights remain referenced between forks and are produced/consumed on the same stream. Persistent top-k storage does not need allocator lifetime protection merely because another stream accesses it. |
| **Events across serial replays** | **No issue.** Replay executes captured GPU dependencies, not the Python `_pending` state machine. Reusing per-layer events across sequential captures/replays is not inherently a stale-event bug. |
| **cuBLAS/custom GEMM scratch** | **Not established.** No handle or workspace implementation is supplied. A race cannot be inferred merely from concurrent GEMMs, but shared custom scratch cannot be cleared without inspecting those implementations. |

The skip-KV dependency chain is especially important:

`staging reads → ready → shared-layer wait → remaining shared layers → next anchor’s side.wait_stream(main) → new top-k writes`

Also, `flashmla_sparse.py:910` or `:916` supplies **converted** indices to the staging path, rather than directly passing the original shared top-k at `:992`. Its allocation/aliasing implementation is not shown, but the readiness chain already orders the shown staging reads.

`record_stream` protects allocator reuse; it does **not** protect explicitly shared workspace views or compiler-planned buffer reuse. Those need real dependencies or separate storage.

For rank ordering and registration, **no mismatch is apparent in the shown path**. Every rank submits candidate gather before query gather, and each rank locally enforces that order. Rank skew does not require StableTopK to finish everywhere before Q begins.

The host calls also remain sequential: candidate gather is invoked inside `_fork_q` before it returns; query gather is invoked afterward. GPU overlap does not reorder those host calls. The side stream has joined capture before the candidate call, so `one_shot/__init__.py:92–94` takes the captured-address path. However, the actual graph-buffer registration implementation is absent, so registration correctness cannot be fully certified from this excerpt alone.

The normal host lifecycle is sound:

- **Eager steps:** no issue. `_fork_k` sets the inline mode; `_fork_q` consumes and clears its saved tuple. `gathered()` does nothing because `_recording` is unset.
- **Warmup inside a capture context, without active stream capture:** no issue in this state machine. It remains inline. However, `one_shot/__init__.py:95–96` deliberately skips communication, so those outputs are not correctness references or meaningful production timings.
- **Dummy/profile runs without attention metadata:** no issue. Forking is disabled, and the indexer uses its profiling branch at `sparse_attn_indexer.py:325–359`.
- **The stated FULL decode capture:** no issue on successful completion. `_indexer_kw` is consumed; `_recording` is reset; the main join clears `_pending` and `_gathered`.
- **Piecewise capture:** conditional on metadata invariants not supplied.
- **MTP:** not cleared by these excerpts. The eligibility check includes its DSA indexer too. Sequential use is fine if its names are unique and its attention reaches the joins. `indexer_fork.py:35` silently overwrites duplicate layer names: registering target and draft under the same name would make target calls resolve the draft wrapper, irrespective of the top-k argument passed. Neither namespace uniqueness nor the drafter’s consumer path is shown.

The assertions check host sequencing, not replay-time synchronization. They also do not provide recovery after a partially failed capture.

**The test is a useful smoke test, but it does not guard most of the risks above.**

It exercises capture, replay with changed inputs, and two producer/consumer waits. Its limitations are concrete:

- `tests/model_executor/test_indexer_fork.py:58` checks a **sum of top-k indices**, so different index sets—and every permutation of a set—can escape detection.
- The long delay at `:30` is **before** `gathered()` at `:38`. Thus the gather wait already covers that delay. Removing the final join may still pass because only the short top-k operation remains.
- Main’s GEMM at `:56` follows the gather wait at `:54`, which already follows the indexer GEMMs. The test does not exercise concurrent cuBLAS GEMMs.
- Its “collective” is a local copy, so it cannot reproduce shared Signal races, rank-order mismatches, or registration problems.
- It has no shared workspace manager, anchor/shared-layer sequence, skip-KV branch, production mode predicate, drafter, or compiled execution.
- Its reference uses the same refactored implementation with forking disabled at `:68`; it does not establish equivalence to the original compiled indexer.

The most valuable additions are full-buffer comparisons, a deliberately delayed write **after** the gather marker, an anchor/shared-layer/staging sequence, and a real four-rank compiled FULL-graph test. Separate tests should exercise the head-major branch and aliasing workspace path.

Finally, **an env-off versus env-on speedup does not isolate stream overlap**. Enabling the feature also moves the indexer behind opaque eager custom ops, including during nonforking execution. That can change fusion, kernel selection, memory allocation, and numerical rounding.

Use three otherwise identical captured variants:

1. Original serial implementation.
2. New opaque implementation forced to run serially.
3. New opaque implementation with forks.

Comparing 2 versus 3 isolates overlap. Comparing 1 versus 3 measures the total implementation change.

Bitwise intent also needs verification against variant 1: identical mathematical operations do not establish identical results across eager/Inductor execution, and small score differences can change selected top-k entries. Compare full indices and downstream outputs, not their sums.

Time completed replays after registration, with equivalent KV/hot-cache state and workload across ranks. **Main-stream timing is valid when all measured side work rejoins it**, as the shown indexer path does. Warmups with skipped collectives, unequal cache warming, omitted side work, or changed MTP acceptance behavior would make the reported speedup misleading.

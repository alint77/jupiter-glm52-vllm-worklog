The `bfi` arithmetic is correct, and MUT2’s early release is harmless to buffer reuse in the short launch. The main problems are the strength of the hardware-level claims, unsupported conclusions about sanitizer coverage, and scripts that can report stale results.

I reviewed only the supplied material and ran no commands.

1. **High — The proof claim exceeds the verifier’s stated model.**

   **Claim:** README §Result: “proves skinny_gemm’s in-CTA TMA/mbarrier pipeline race-free and well-synchronized.”

   **Why weak:** The reported verdicts establish properties of the **prepared PTX and supplied configurations under SentryWard’s semantics**. Its immediate-visibility assumption and absence of fence modeling leave a separate obligation to connect those semantics to Hopper.

   In particular, audit the reuse chain:

   `ldmatrix` reads at **169–172** → `__syncwarp()` and `empty` arrival at **176–177** → producer wait at **139** → next TMA writes at **144–145**.

   A happens-before path in SentryWard does not by itself establish that this chain supplies every ordering requirement between generic-proxy reads and async-proxy writes on Hopper. Whether the existing handoff suffices requires an architecture-level argument. The initialization fence at **126** addresses a different ordering obligation; compiler `"memory"` clobbers also do not establish hardware proxy ordering.

   Successful parsing of `sm_90a` PTX by a Blackwell verifier is insufficient evidence of architectural equivalence. Nor does verifying separately generated PTX validate its correspondence to the production cubin/SASS.

   **Fix:** Qualify the proof explicitly by model, rewritten artifact, configuration, and assumptions. Document the Hopper ordering argument separately, including initialization publication, buffer reuse, and TMA-completion visibility. Record compiler/tool versions and artifact hashes linking the checked PTX to the tested build.

2. **High — The reproduction scripts can produce apparently successful results after failures.**

   **Claim:** README §SentryWard: “`run.sh` reproduces everything below.”

   **Why weak:** `run.sh` uses only `set -u`. It suppresses compiler diagnostics, continues after compilation/preparation failures, ignores the verifier’s exit status, and reads reports from a reusable directory. A failed compilation can leave an old PTX artifact available; a failed or timed-out verifier invocation can leave an old JSON report that the following Python command prints successfully.

   Likewise, `sweep.sh` records exit codes but does not enforce success, finishes with `echo DONE`, and extracts only the last matching summary. Its exit status does not establish that the sweep passed.

   **Fix:** Use fresh run directories, preserve diagnostics, check each producer’s status, and require newly generated artifacts. Handle the verifier’s documented verdict-related exit codes explicitly. Assert expected verdicts and completed checks for each case. Configure sanitizer errors to produce an actionable exit status and validate application/test completion separately.

   This does not prove the supplied results are stale; it means the scripts do not establish their provenance reliably.

3. **High — “Racecheck tracks only generic-proxy accesses” is not demonstrated.**

   **Claim:** README §Positive controls: “TMA writes into shared memory are invisible to it”; §Result and §Survey generalize this to blindness to TMA hazards.

   **Why weak:** Failure to report these mutants establishes a detection gap for the tested version, configuration, and execution. It does not identify the implementation mechanism or establish that all TMA writes are untracked.

   Also, the table does not establish whether the wrong output occurred **during the same instrumented invocation** that reported zero hazards. Instrumentation can change scheduling. MUT3 shows that some ordinary shared-memory checking works, but does not establish TMA coverage.

   Synccheck’s failure to report a memory-ordering bug is not itself evidence that it violates its checking contract.

   **Fix:** Report the exact sanitizer version/options and state: “Racecheck and synccheck reported no diagnostics for MUT2 in these runs.” Preserve correctness results from those same runs. Add minimal TMA RAW, WAR, and WAW cases alongside ordinary shared-memory controls before asserting which accesses the tool models.

4. **Medium — The three rewrites have different soundness conditions; the actual `bfi` lowering is correct.**

   **Claim:** README §Workarounds calls them equivalent or exact.

   **`prefetch.tensormap`:** Removing a cache hint is defensible for this ring-synchronization property when descriptors are valid, immutable kernel parameters and the instruction contributes no synchronization semantics. The supplied `__grid_constant__` descriptor usage supports that argument. It does not validate descriptor-address legality or hypothetical concurrent descriptor updates. The claimed register-clobber implementation bug still needs a minimal reproducer.

   **`bfi`:** At the supplied site, the operation is:

   ```text
   result = ((a << 4) & 0xfffffff0) | (b & 0x0000000f)
   ```

   The mask calculation is correct:

   ```text
   ((1 << 28) - 1) << 4 = 0xfffffff0 = 4294967280
   ```

   Its 32-bit complement is `15`. The lowering inserts the low 28 bits of `%r569` into bits 4–31 and preserves `%r297`’s low four bits. There is no arithmetic error here.

   However, the script accepts unrestricted decimal position/length operands. Either restrict it to supported in-range cases or implement the full PTX immediate semantics.

   **Alignment:** At **111–113**, `raw` explicitly has 1024-byte alignment, so the replacement is algebraically correct **if that guarantee holds for the actual emitted shared symbol**. Merely aligning SentryWard’s allocation would not establish the real kernel’s alignment.

   The rewrite implementation is weaker than that argument: `pend` never invalidates register definitions, checks neither intervening control flow nor base-register reassignment, and applies to every matching register pair without proving the base is aligned.

   **Fix:** Make preparation fail unless the expected sites and counts match. Validate the emitted alignment and exact def-use chain; retain an auditable original/prepared diff. Treat this as a narrowly validated transformation of these artifacts.

5. **Medium — The ring-iteration counts are wrong, although the short MUT2 interpretation is substantially right.**

   **Claim:** `run.sh` comments say “8 K-steps” at K=512 and “64 K-steps” per tile at K=4096; README says the ring wraps eight times per tile.

   **Why wrong:** A stage iteration is a **128-element chunk**, not one 64-element TMA box. The two boxes are issued within one iteration at **137–146**.

   With `TILE_N=64`, `CHUNK_K=128`, `N=128`, and eight stages:

   | Launch | Chunks per tile | Iterations per CTA | Uses of each stage |
   |---|---:|---:|---|
   | K=512, grid 2 | 4 | 4 | Four stages used once; four unused |
   | K=4096, grid 1 | 32 | 64 across two tiles | Eight uses total |

   The long launch makes four complete ring traversals per tile. Across the CTA, each stage has one initial use and seven reuses.

   **Interpretation:** MUT2’s early release cannot enable a subsequent overwrite in the short launch because no stage is reused. That conclusion is correct. Replace “race_free: correct” with “no buffer-reuse hazard is introduced by this mutation at this launch”; the verdict is not a functional-correctness proof.

   MUT1’s two verdicts are plausible: without reuse, skipping the initial wait leaves an unordered read; with reuse, the absence of a successful wait for that phase can additionally violate the barrier handoff rule.

   For long MUT2, memory checking was **not run**. The evidence is that the synchronization checker rejects the mutation, not that the memory checker directly detects its WAR race.

6. **Medium — The verified configuration covers a substantially narrower pipeline family than the README suggests.**

   **Claim:** README §Result refers broadly to “skinny_gemm’s … pipeline.”

   **Why weak:** `mk_skinny.py` checks only configuration 1, with M=8, N=128, contiguous descriptors, and the specified grid/K pairs. Both completed baseline analyses avoid split-K.

   From **38–46**, configurations 0, 1, 2, and 3 have **6, 8, 3, and 6 stages**, respectively. The unverified variants change consumer counts, transfer counts, stage sizes, and phase/index arithmetic. They are not simply larger instances of the checked eight-stage case.

   Ragged dimensions, other strides, partial tiles, other CTA partitions, and workspace lifecycle are outside these two results. Race freedom also does not establish correct swizzling, GEMM arithmetic, or numerical accuracy.

   The split-K branch pin introduces another limitation: a single line-based decision may apply to multiple CTAs and dynamic visits. “This CTA flushes” is not sufficient justification for the resulting path. Its reachability and scope need validation before any future conditional proof. The present `unsupported` result provides no such proof.

   **Fix:** Enumerate the exact verified configurations and exclusions. Extend checks to each distinct stage/consumer configuration and to tile-boundary phase continuation.

7. **Medium — The initcheck attribution is plausible but not established by the reported evidence.**

   **Claims:** README §compute-sanitizer sweep says filtering makes “every torch-produced buffer” uninitialized and attributes all activation errors to the Marlin reference’s `torch.empty` cache.

   **Why weak:** The filtered/unfiltered difference is consistent with omitted producer-write tracking. It does not justify the universal “every buffer” claim or a blanket prohibition on filtering. A filter that includes the relevant producers and consumers may be adequate.

   The 18 reports also match the three scalar conversions per skinny case, but matching counts alone does not prove their origin.

   For the decode errors, an `act_and_mul_kernel` report locates the invalid read. `torch.empty` only explains why storage starts uninitialized; scratch storage is normally written before use. The root-cause claim requires showing exactly which rows are read, which Marlin writes omit them, and whether they are live or padding.

   Until that is established, the affected reference cannot automatically be treated as a reliable correctness oracle.

   **Fix:** Reproduce the reference alone in a fresh process and identify the allocation and read indices. Compare explicit initialization and distinct poison patterns, checking effects on valid outputs. Run the custom path separately. Describe the filter effect as observed behavior of this configuration, supported by the relevant diagnostics.

8. **Medium — The controls do not yet substantiate transaction accounting, phase identity, or initialization ordering.**

   **Claim:** README §SentryWard advertises phase agreement and “no early completion”; §Survey includes “TMA bytes.”

   **Why weak:** The two ordering mutants exercise only part of those obligations.

   The supplied byte arithmetic is internally consistent: each stage expects **18,432 bytes**. Each of two box iterations transfers **8,192 W bytes + 1,024 X bytes**, totaling the same amount. What remains unestablished is how faithfully the verifier checks transaction completion and accounting.

   Parity similarly needs a generation argument: a successful parity test must refer to the intended phase, despite parity repeating every two phases. The long launch exercises repeated parity values, but an accepting result alone does not show how the verifier rules out stale-phase interpretation or lapping.

   **Fix:** Add controls that distinguish these mechanisms:

   | Control | What it tests |
   |---|---|
   | Expected bytes ±1,024; omit one X transfer while preserving the expected count | Early completion, incomplete completion, transaction accounting |
   | Wrong full/empty parity; hold parity constant | Phase identity and stale successful waits |
   | Run `S−1`, `S`, `S+1`, `2S+1`, and `3S+1` iterations | First reuse and repeated parity |
   | Wrong consumer arrival count; skip or duplicate one arrival | Barrier participation and premature release |
   | Remove `__syncwarp()` at 176 while retaining lane-zero release | Ordering of every lane’s reads before stage release |
   | Producer-only stage-index or stage-stride error | Address overlap and mismatched ring ownership |
   | Remove `fence.mbarrier_init` at 126 | Explicitly characterize the fence-modeling boundary |
   | Known-required proxy-ordering litmus, with and without its fence | Characterize what immediate visibility conceals |

   Some malformed variants can hang, so future hardware controls need bounded execution. A passing missing-fence mutant under a model that excludes fences documents a limitation; it does not validate the fence’s dispensability.

9. **Medium — The sweep’s coverage claims exceed what its summaries establish.**

   **Claim:** README §Result says the correctness suites are clean; §Sweep says the namespace filters cover all 16 kernels.

   **Why weak:** Matching a namespace establishes filter eligibility, not that every kernel executed. The pytest commands select subsets and use `-x`; summaries must distinguish executed, skipped, and unselected cases.

   In `skinny_check.py`, only the second invocation’s result and final workspace state are checked. That is weaker than checking the stated zero-on-exit contract after every invocation. The global maximum-normalized error can also hide relatively large errors in small-magnitude outputs.

   **Fix:** Report selected/executed/skipped counts and observed kernel launches. Check correctness and workspace state after each invocation, with suitable elementwise tolerances and diagnostic inputs. Phrase cleanliness as “no diagnostics in the selected, completed runs.”

10. **Low — The survey and proposed litmus work need narrower wording.**

   **Claims:** README §Result: “Nothing else released verifies Hopper async kernels”; §Next: Dartagnan litmus tests “cover acquire/release and scopes.”

   **Why weak:** A survey does not establish universal absence. Its table also mixes partial support with coverage of the entire TMA/mbarrier/WGMMA/proxy-fence combination.

   Litmus results apply to their abstraction, bounds, and memory model. Transferring them to the actual cross-CTA implementation requires an argument about participant counts, fences, atomics, polling, and repeated invocations.

   **Fix:** Say “Among the surveyed versions, we did not identify a released tool covering the full combination.” Describe litmus work as checking extracted protocol obligations, with an explicit mapping back to the implementation.

A defensible replacement for the central README conclusion would be:

> SentryWard 60a1ebc reports `race_free`, with synchronization and memory checks passing, for prepared PTX of `skinny_kernel<64,128>` at the two listed configurations under its documented modeling assumptions. It rejects MUT1 and MUT2 on the stage-reuse launch during synchronization checking; MUT1 also fails memory checking on the short launch. These results provide bounded regression evidence. They do not establish Hopper proxy/fence correctness, PTX-to-SASS correspondence, functional correctness, split-K correctness, or correctness of other configurations.

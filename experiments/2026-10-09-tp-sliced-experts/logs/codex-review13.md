**Conditional ship.** I find no parity error or scheduling deadlock in NOAHEAD. Before shipping, I would reconcile the unused fence knob, add the map bound, and close the targeted test gaps below. The performance evidence supports a **projected** improvement over EP; it does not yet establish the smaller NOAHEAD improvement statistically.

This review uses only the supplied source and results; I ran no commands.

1. **Kernel correctness**

   The new parity is correct. For publication number `n`, slot `k = n % TD_SQ` is using generation `g = n / TD_SQ`.

   - The reuse wait checks completion of generation `g - 1`.
   - The new post-publication wait checks completion of generation `g`.
   - Thus `((n / TD_SQ) - 1) & 1` before reuse and `(n / TD_SQ) & 1` after publication are the right pair.

   For example, slot 0’s first publication waits on parity 0 afterward. Its second publication first checks completion of parity 0, then waits on parity 1 afterward. An arrival occurring before the wait is fine. Rechecking an already completed generation during reuse is also fine.

   **Two details differ from your description:** `sq_empty` arrives at line 1319, after the producer has issued **all chunks**, not when it first accepts the record. Also, `heavy` includes `K_S1`, shared w2, because it excludes only `K_R1`. Consequently, the new wait can indirectly include a ready spin for the published shared-w2 group itself.

   Neither detail introduces a cycle. The progress argument is:

   - Each routed-w2 dependency was claimed earlier in that tier’s queue; likewise, all shared-w13 dependencies precede shared w2.
   - Each CTA processes its groups in claim order, including across its one tier switch.
   - If a required w13 group is queued behind a ready-blocked group on another CTA, that blocking group was claimed still earlier.
   - Following such blocking dependencies therefore moves strictly backward through claim order. It cannot return to its starting point.

   NOAHEAD delays subsequent claims; it does not invert this ordering. Ring and handoff backpressure preserve FIFO order, and the finisher has no ready spin that introduces a reverse dependency.

   Stealing and termination also look correct. Stealing leaves `n` and `k` unchanged because nothing was published. The extra claim made on an already exhausted queue is necessarily out of range. `K_END` is never heavy, so the scheduler does not wait for an acknowledgment the producer will never send. The producer checks `K_END` before reading the remaining record fields, and consumers send their finisher sentinel after preceding handoffs.

   The workspace cleanup is consistent with sequential reuse: used `y13` rows are cleared by their finisher, shared rows by activation, and output rows by finalize. Publishing ready before clearing `y13` is compatible with w2 reading `x2`; the next call must remain ordered after predecessor completion through PDL.

   **One concrete source/configuration mismatch needs reconciliation:** `TD_FIN_FENCE` is defined at lines 794–796 but never invoked. Neither is `fence_acq_rel_gpu()`. Therefore that advertised override currently does nothing.

   This absence alone does **not** demonstrate a race: the shown ordering uses the handoff barriers, the lane-0 `atom.acq_rel.gpu`, and warp synchronization. But the source does not contain the separately invoked fence described in the review-12 defaults. Establish whether that fence was deliberately superseded, then remove the dead knob or restore its intended use and revalidate. Reported SASS identity establishes equivalence to the compiled baseline, not agreement with the prose description.

   I found no other concrete cleanup defect in the supplied default path. Update the scheduler comments to say “after all chunks of a non-R1 group are issued.”

2. **Guard `num_experts`**

   I recommend adding the bound at the placement-less lookup:

   ```cpp
   if (e >= 0 && e < num_experts) {
     const int h = maps[e], c = maps[MAX_EXPERTS + e];
     // ...
   }
   ```

   The existing line-591 guard is inactive here because `pl.num_experts == 0`. IDs between `num_experts` and 511 read uninitialized shared-memory map entries; larger IDs can read outside the array.

   This is not a defect for the stated valid-router contract, but the extra comparison is worthwhile at this raw-pointer boundary. It requires no additional memory access or synchronization. Keep map **values** being valid tier slots as a separate caller contract.

   The shown ctypes argument insertion matches the C signature, and the equal-map-length check is appropriate.

3. **Adversarial coverage**

   The existing suite is substantial, especially its workspace reuse, odd token counts, and mixed-call graph. These are the additions I would prioritize:

   | Case | Why it matters |
   |---|---|
   | All routes masked, **shared disabled**, after a busy call | Exercises both queues empty, immediate termination, and exact-zero output. |
   | Independent hot/cold slot permutations, with tensors reordered consistently | Tests logical expert IDs versus physical tier slots. |
   | Routed IDs absent from both maps, mixed with present IDs | Tests dropped local contributions and `live`/list construction. |
   | T=32, all 256 routes on one cold expert | Exercises 32 entries for one slot and repeated token destinations. Useful stress even though ordinary top-k does not duplicate IDs within a token. |
   | Single-route and one-hot-weight cases, **shared disabled** | Makes a missing routed contribution observable without shared output masking it. |
   | Actual empty hot/cold tensor tiers, and the real `padding` argument | Existing routing-only cases do not exercise these API paths. |

   **The first case is currently absent.** Moreover, this line would skip its validation if you simply added it:

   ```python
   if kind == "shared_only" and not shared:
       continue
   ```

   Remove that skip. The existing zero-activation case checks an entirely zero output at M=1, but does not exercise empty work queues. Also assert exact zero **per zero-reference row**; the current exact-zero branch only applies when the whole reference tensor is zero.

   For PDL, the existing graph already provides useful adjacent-call coverage. Strengthen it by changing input-buffer contents between replays, with matching references, and including shared-on/off and populated/empty transitions. Repeated fixed inputs can conceal some stale-data errors.

   Ordinary adjacent calls should **not** have equal ready epochs. If you mean rollover, test that explicitly: `int epoch += 1` has signed-overflow concerns, and a sufficiently old inactive entry can retain a colliding tag after a full tag cycle. A rollover-safe design needs defined arithmetic plus invalidation on wrap, or a wider generation scheme; unsigned arithmetic alone does not prevent tag aliasing.

   **Keep `5e-3` as a broad numerical acceptance threshold, but do not treat it as proof that every route contributed.** A route has no guaranteed one-eighth contribution: router weights, expert magnitudes, cancellation, shared output, and another token’s larger maximum can all hide it. Add per-token error checks and isolated-route tests rather than simply tightening the global threshold.

   Also, unless `kdev.reference` reproduces them, the kernel’s f16 routed activation and bf16 shared activation rounding contribute to error. The scalar worst error alone cannot attribute everything to final bf16 rounding and fp32 summation order.

4. **Performance interpretation**

   Using the slowest EP GPU per layer is the appropriate modeled baseline for synchronized execution. Your replay predicts:

   | M | Ship versus EP slowest-GPU time | NOAHEAD versus v53 |
   |---|---:|---:|
   | 8 | 19.9% lower | 0.71% lower; **0.67 μs/layer** |
   | 32 | 15.8% lower | 1.92% lower; **4.93 μs/layer** |

   That supports saying: **“The fitted replay predicts approximately 20% and 16% lower MoE time than production EP.”** It is not yet a measured production or end-to-end speedup.

   The largest modeling concern is that unique hot/cold expert counts omit route multiplicity and entry splitting. Those directly affect this scheduler—and motivate NOAHEAD. Holding out production routing traces does not validate timing predictions unless representative held-out routings are also timed. Check that those traces lie within the fitted range and that their concentration patterns are represented.

   Also ensure both sides use equivalent timing boundaries, including shared computation and any communication included in the claim, and account for the slowest sliced rank where relevant.

   **Maximum fit residual is not an error bar on the mean difference.** Large residuals can cancel in paired comparisons; systematic residuals can persist across all 75 layers. Therefore the reported residuals neither disprove nor establish the NOAHEAD gain.

   To establish that smaller gain, use paired v53/NOAHEAD timings on the same production routings, repeated with interleaved variant order. Report the distribution and uncertainty of the paired difference, accounting for repeated layers/steps. The raw table compares against v48, so it does not independently establish the v53-to-NOAHEAD effect.

5. **Ship decision**

   I would approve the default kernel after:

   - Reconciling the unused `TD_FIN_FENCE` configuration claim.
   - Adding the bounded map lookup.
   - Passing the empty-work, map-permutation, missing-tier, and concentrated-cold tests, with route-isolation checks.
   - Receiving clean results from the pending exact-source seeds and vLLM tests.

   **No NOAHEAD redesign is indicated by this review.** Proving its small performance increment need not block shipping a validated kernel, but present that increment—and the EP replay comparison—as modeled results until paired measurements establish them.

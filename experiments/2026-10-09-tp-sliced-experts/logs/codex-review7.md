The GLM/bf16 scaling checks out, but I would request changes to the execution-order handling. The hook establishes capability; it does not guarantee that the runner assigns ownership to the method.

1. **[P1] Fused decode conflicts with forced `NO_OVERLAP`.**  
   `_determine_shared_experts_order()` checks `_disable_shared_experts_overlap` **before** checking MK capability. With EPLB on a non-default backend, or `use_fi_nvl_two_sided_kernels`, the sequence is:

   - The runner executes the shared MLP in its `NO_OVERLAP` slot.
   - `apply_tiered_moe()` still includes shared computation in `_sliced_decode()`.
   - `_set_shared_output()` finds an occupied slot and asserts.

   Fusion must depend on the **actual selected execution order**, or these configurations must be explicitly excluded. Merely skipping the private write when the slot is occupied would double-count shared output.

   The non-fusable and Marlin paths handle this override better: their `MK_INTERNAL_OVERLAPPED` calls become no-ops, preserving the output already produced by the runner.

2. **[P2] The fusibility predicate does not establish shared-MLP equivalence.**  
   Matching weight shapes, contiguity, and bf16 dtype are insufficient. For example, a matching `down_proj` with a bias passes this check, but the kernel receives no bias. Activation differences or shared gating can also change the result.

   There is also an input contract: fusion computes from `x`, whereas the ordinary shared MLP receives `shared_experts_input`. A routed input transform can make these different. A routed output transform would subsequently transform the shared contribution embedded in `fused_output`, although the separate shared contribution normally bypasses that transform.

   Restrict fusion to a known-compatible shared module and transformation configuration. If other guards already enforce these restrictions, that needs verification beyond this diff.

For normal MK ownership, the three paths have the right structure. Let `E` be the local routed result, including any scale already incorporated by routing; `S` the local shared result; and `R` the runner’s scale:

| Step | Shared output slot | Returned fused output |
|---|---|---|
| Fusable sliced decode | `0` | `E + S/R` |
| Sliced Marlin | `S` | `E` |
| Non-fusable sliced decode | `S` | `E` |

The bf16 runner therefore produces `S + R*E` in all three cases. **Marlin’s exactly-once execution additionally requires that the omitted continuation never calls the shared hook again.** Calling it before entering a downstream MK that also owns shared execution would assert on the second call.

Reduction needs a separate invariant. `_fused_output_is_reduced` still consults `method.moe_kernel`, even when sliced decode bypasses that kernel. For local TP slices, that property must report **false**. If it reports true, the runner all-reduces the dummy zeros and skips reducing the actual fused result, leaving both routed and embedded shared contributions local. With false reduction metadata and ordinary TP reduction settings, all three paths correctly combine first and all-reduce once. The ownership override alone does not establish this.

The scale derivation is correct under the supplied GLM wiring:

- With `apply_routed_scale_to_output=True`, the routed layer/router factor is `1`, the runner factor is the model factor `M`, and `shared_scale=1/M`.
- With output scaling disabled, the routed layer/router factor is `M`, the runner factor is `1`, and `shared_scale=1`.

For bf16, `_maybe_apply_routed_scale_to_output()` multiplies `fused_output`; the FP16 overflow branch does not apply. Thus `M*(E + S/M)` gives the intended shared contribution. This is mathematical equivalence, **not bitwise equivalence**, because fusion changes rounding.

The derivation assumes the HF config factor exactly matches the factor passed into this layer’s construction. Passing the actual runner scale through the execution contract would avoid that assumption. A zero runner scale requires separate execution because reciprocal cancellation is impossible; `or 1.0` does not solve that case.

**DBO and CUDA graphs:** I see no intrinsic problem with using the current `_output_idx`, provided the method remains in the same ubatch context through production and consumption. `torch.zeros_like(out)` is capture-compatible: its allocation uses graph-managed storage and its zeroing is recorded. The Python list assignment need not replay when both assignment and `.output` consumption occur inside the captured forward; consumption clears the slot during capture. The private write’s demonstrated problem is bypassing execution-order arbitration, not graph capture itself.

Two additional configuration checks remain:

- Takeover must be restricted to modular execution; the shown monolithic branch never passes the shared wrapper to the method.
- With fusion disabled, native `moe_kernel.can_overlap_shared_experts=True` could still select MK ownership while sliced decode bypasses that MK and produces no shared output.

Review based on the supplied excerpts only; no shell commands or tests run.

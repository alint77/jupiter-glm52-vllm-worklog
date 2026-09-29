#!/usr/bin/env python3
"""Move the replica-padding mask into the one-kernel MoE's route_prep.

mask_replica_padding (torch.where(is_padding[:, None], -1, topk_ids)) costs two
kernels per MoE layer; route_prep already reads topk_ids, so it takes the
per-token padding flags and treats a padded token's routes as -1 itself.
Applied to the vLLM tree in place (run from the repo root).
"""

from pathlib import Path

cu = Path("vllm/model_executor/layers/fused_moe/tiered_decode/tiered_decode.cu")
s = cu.read_text()
old = """template <typename IdT>
__global__ void route_prep_kernel(Workspace* ws, const __nv_bfloat16* x,
                                  const IdT* topk_ids,
                                  const float* topk_weights, const int* hot_map,
                                  const int* cold_map, Placement pl,
                                  int num_tokens, int hot_size, int cold_size) {"""
new = """template <typename IdT>
__global__ void route_prep_kernel(Workspace* ws, const __nv_bfloat16* x,
                                  const IdT* topk_ids, const bool* padding,
                                  const float* topk_weights, const int* hot_map,
                                  const int* cold_map, Placement pl,
                                  int num_tokens, int hot_size, int cold_size) {"""
assert old in s
s = s.replace(old, new)
old = """  if (r < routes) {
    e = static_cast<int>(topk_ids[r]);
    wt = topk_weights[r];"""
new = """  if (r < routes) {
    e = static_cast<int>(topk_ids[r]);
    // a padded token's routes are dropped, as if its ids were -1
    if (padding != nullptr && padding[r / TOPK]) e = -1;
    wt = topk_weights[r];"""
assert old in s
s = s.replace(old, new)
old = """             torch::Tensor cold_w2, torch::Tensor cold_s2,
             torch::Tensor workspace, bool pdl_launch) {"""
new = """             torch::Tensor cold_w2, torch::Tensor cold_s2,
             torch::Tensor workspace, bool pdl_launch, torch::Tensor padding) {"""
assert old in s
s = s.replace(old, new)
old = """  const auto* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());"""
new = """  const bool* pad = nullptr;
  if (padding.defined() && padding.numel() > 0) {
    TORCH_CHECK(padding.scalar_type() == at::kBool && padding.is_contiguous() &&
                    padding.numel() >= T,
                "padding must be >= T contiguous bools");
    pad = padding.data_ptr<bool>();
  }
  const auto* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());"""
assert old in s
s = s.replace(old, new)
s = s.replace(
    "&c, route_prep_kernel<int>, p.ws, xp, topk_ids.data_ptr<int>(),",
    "&c, route_prep_kernel<int>, p.ws, xp, topk_ids.data_ptr<int>(), pad,",
)
s = s.replace(
    "&c, route_prep_kernel<int64_t>, p.ws, xp, topk_ids.data_ptr<int64_t>(),",
    "&c, route_prep_kernel<int64_t>, p.ws, xp, topk_ids.data_ptr<int64_t>(), pad,",
)
assert s.count(", pad,") == 2
cu.write_text(s)

py = Path("vllm/model_executor/layers/fused_moe/tiered_decode/__init__.py")
s = py.read_text()
old = """    placement: Placement | None = None,
) -> torch.Tensor:
    \"\"\"Routed MoE output for up to 8 tokens, summed over this rank's experts."""
new = """    placement: Placement | None = None,
    padding: torch.Tensor | None = None,
) -> torch.Tensor:
    \"\"\"Routed MoE output for up to 8 tokens, summed over this rank's experts."""
assert old in s
s = s.replace(old, new)
old = """        placement: Assign replicas in the kernel (4 GPUs), or None.
"""
new = """        placement: Assign replicas in the kernel (4 GPUs), or None.
        padding: [>= T] bool, True for padding tokens, whose routes are
            dropped as if their ids were -1; or None.
"""
assert old in s
s = s.replace(old, new)
old = """        # chain the layer's five kernels with programmatic dependent launch
        os.environ.get("VLLM_TIERED_DECODE_PDL", "1") != "0",
    )"""
new = """        # chain the layer's five kernels with programmatic dependent launch
        os.environ.get("VLLM_TIERED_DECODE_PDL", "1") != "0",
        padding if padding is not None else empty,
    )"""
assert old in s
s = s.replace(old, new)
py.write_text(s)

ex = Path("vllm/model_executor/model_loader/tiered_moe_execution.py")
s = ex.read_text()
old = """    \"\"\"Execute the hot and cold expert tiers through their shared runtime.\"\"\"
    if (
        getattr(method, "tiered_replica_assignment", "off") != "off"
        and topk_ids.shape[0] <= method.tiered_replica_max_tokens
    ):
        from vllm.model_executor.model_loader.tiered_moe_scheduler import (
            mask_replica_padding,
        )

        topk_ids = mask_replica_padding(topk_ids)
        if envs.VLLM_TIERED_MOE_ROUTE_TRACE:"""
new = """    \"\"\"Execute the hot and cold expert tiers through their shared runtime.\"\"\"
    one_kernel = _decode_kernel_applies(method, layer, x, shared_experts)
    padding = None
    if (
        getattr(method, "tiered_replica_assignment", "off") != "off"
        and topk_ids.shape[0] <= method.tiered_replica_max_tokens
    ):
        from vllm.model_executor.model_loader.tiered_moe_scheduler import (
            mask_replica_padding,
            replica_padding,
        )

        # The one-kernel path drops padded routes in its route kernel; the
        # route check / trace and the Marlin path need the masked ids.
        if (
            one_kernel
            and not envs.VLLM_TIERED_MOE_ROUTE_TRACE
            and not envs.VLLM_TIERED_MOE_ROUTE_CHECK
        ):
            padding = replica_padding(topk_ids)
        else:
            topk_ids = mask_replica_padding(topk_ids)
        if envs.VLLM_TIERED_MOE_ROUTE_TRACE:"""
assert old in s
s = s.replace(old, new)
old = """    if _decode_kernel_applies(method, layer, x, shared_experts) and (
        (placement := decode_placement(method, layer, topk_ids)) is not None"""
new = """    if one_kernel and (
        (placement := decode_placement(method, layer, topk_ids)) is not None"""
assert old in s
s = s.replace(old, new)
old = """            method.tiered_moe_kernels[1][1],
            placement,
        )"""
new = """            method.tiered_moe_kernels[1][1],
            placement,
            padding,
        )"""
assert old in s
s = s.replace(old, new)
ex.write_text(s)

sc = Path("vllm/model_executor/model_loader/tiered_moe_scheduler.py")
s = sc.read_text()
old = """def mask_replica_padding(topk_ids: torch.Tensor) -> torch.Tensor:"""
new = """def replica_padding(topk_ids: torch.Tensor) -> torch.Tensor | None:
    \"\"\"The step's per-token padding flags, [>= T] bool, or None.\"\"\"
    from vllm.forward_context import (
        get_forward_context,
        is_forward_context_available,
    )

    if is_forward_context_available():
        padding = get_forward_context().is_padding
        if padding is not None:
            return padding[: topk_ids.shape[0]]
    return None


def mask_replica_padding(topk_ids: torch.Tensor) -> torch.Tensor:"""
assert old in s
s = s.replace(old, new)
sc.write_text(s)
print("applied")

"""FlashMLA sparse attention at our long-context prefill shape, one DCP rank:
2464 query tokens, 64 heads (DCP-gathered), top-2048 indices of which ~1/4
point into this rank's KV shard (25K tokens of a 100K context), fp8_ds_mla
cache. Kernel time from the torch profiler (sum of kernel durations / iters).

  a  as served: fp8 decode kernel, one batch row of T queries, 2048 index
     slots per query, owned ones compacted to the front, tail -1 (masked)
  b  same, index tensor trimmed to the longest valid prefix (rounded to 64)
  c  one batch row per query (s_q = 1) with topk_length = its valid count
  d  all 2048 slots valid (scaling check: a's work if nothing were masked)
  e  FlashMLA sparse *prefill* kernel on a bf16 copy of the shard, with
     topk_length (excludes the fp8 -> bf16 upconvert)
"""
import sys

import torch
from torch.profiler import ProfilerActivity, profile

from vllm.third_party.flashmla.flash_mla_interface import (
    flash_mla_sparse_fwd,
    flash_mla_with_kvcache,
    get_mla_metadata,
)

torch.manual_seed(0)
dev = torch.device("cuda", 0)
from vllm import _custom_ops as ops

T = int(sys.argv[1]) if len(sys.argv) > 1 else 2464
CTX_ALL = int(sys.argv[2]) if len(sys.argv) > 2 else 100_000
H, DQK, DV, TOPK, DCP = 64, 576, 512, 2048, 4
CTX = CTX_ALL // DCP
print(f"== T={T} queries, context {CTX_ALL} ({CTX} on this rank)")
BS = 64
nblk = (CTX + BS - 1) // BS
scale = DQK**-0.5

nope = (torch.randn(nblk, BS, 512, device=dev) * 0.5).to(torch.float8_e4m3fn).view(torch.uint8)
sc = torch.full((nblk, BS, 4), 1.0, device=dev).view(torch.uint8)
rope = (torch.randn(nblk, BS, 64, device=dev) * 0.5).to(torch.bfloat16).view(torch.uint8)
kcache = torch.cat([nope, sc, rope], -1).contiguous()  # (nblk, BS, 656)
assert kcache.shape[-1] == 656
kv_bf16 = torch.cat([nope.view(torch.float8_e4m3fn).to(torch.bfloat16),
                     rope.view(torch.bfloat16)], -1).reshape(nblk * BS, 1, DQK)

q = (torch.randn(T, H, DQK, device=dev) * 0.5).to(torch.bfloat16)
valid = torch.randint(TOPK // DCP - 48, TOPK // DCP + 48, (T,), device=dev, dtype=torch.int32)
idx = torch.full((T, TOPK), -1, dtype=torch.int32, device=dev)
ranks = torch.rand(T, CTX, device=dev).argsort(dim=1)[:, : TOPK].int()
cols = torch.arange(TOPK, device=dev)
idx = torch.where(cols[None, :] < valid[:, None], ranks, idx)
idx_all = torch.randint(0, CTX, (T, TOPK), device=dev, dtype=torch.int32)
kt = (int(valid.max()) + 63) // 64 * 64


def dec(qq, ii, topk_length=None):
    meta, _ = get_mla_metadata()
    return lambda: flash_mla_with_kvcache(
        q=qq, k_cache=kcache.unsqueeze(-2), block_table=None, head_dim_v=DV,
        cache_seqlens=None, tile_scheduler_metadata=meta, is_fp8_kvcache=True,
        indices=ii, softmax_scale=scale, topk_length=topk_length)


variants = {
    "a served (b=1, 2048 slots, -1 tail)": dec(q.unsqueeze(0), idx.unsqueeze(0)),
    f"b trimmed to {kt} slots": dec(q.unsqueeze(0), idx[:, :kt].contiguous().unsqueeze(0)),
    "d all 2048 valid": dec(q.unsqueeze(0), idx_all.unsqueeze(0)),
    "e sparse prefill kernel bf16, topk_length": lambda: flash_mla_sparse_fwd(
        q, kv_bf16, idx.unsqueeze(1), scale, DV, topk_length=valid),
}
ws = torch.empty(nblk * BS, DQK, dtype=torch.bfloat16, device=dev)
bt = torch.arange(nblk, dtype=torch.int32, device=dev)[None]
sl = torch.tensor([CTX], dtype=torch.int32, device=dev)
st = torch.zeros(1, dtype=torch.int32, device=dev)


def upconvert_then_prefill():
    ops.cp_gather_and_upconvert_fp8_kv_cache(kcache, ws, bt, sl, st, 1)
    return flash_mla_sparse_fwd(q, ws[:, None], idx.unsqueeze(1), scale, DV,
                                topk_length=valid)


variants["f = e + upconvert this rank's shard to bf16"] = upconvert_then_prefill
outs = {}
for name, fn in variants.items():
    for _ in range(3):
        r = fn()
    torch.accelerator.synchronize()
    outs[name] = r[0].reshape(T, H, DV).float()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(10):
            fn()
        torch.accelerator.synchronize()
    ks = [e for e in prof.events() if e.device_type.name == "CUDA"]
    t = sum(e.device_time for e in ks) / 10
    top = max(ks, key=lambda e: e.device_time).name[:70]
    print(f"{name:<45} {t:8.1f} us   ({top})", flush=True)
ref = outs["a served (b=1, 2048 slots, -1 tail)"]
for name, o in outs.items():
    if name.startswith(("b", "e", "f")):
        print(f"  {name[:1]} vs a: max |diff| {(o - ref).abs().max().item():.3g}, "
              f"ref max {ref.abs().max().item():.3g}")

"""Module test of the shipped anchor hot cache: build from a forward context,
KV update + reserved write + map vs Grace, prefill invalidation, block
invalidation (CoW), and the expect_map guards."""
import os
os.environ["VLLM_ANCHOR_HOT_CACHE"] = "1"
from types import SimpleNamespace
import torch
from vllm import _custom_ops as ops
from vllm.v1.attention.backends.mla import skip_kv_stage
skip_kv_stage.limits = lambda: (16, 768)
from vllm.v1.attention.backends.mla import anchor_hot_cache as ahc
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

dev = torch.device("cuda")
S = 64 * 1024
ahc._ANCHORS = frozenset({0, 2})
graces, cpus = [], []
fc = {}
for i in range(3):
    cpu = torch.zeros(S * 656, dtype=torch.uint8).pin_memory()
    cpus.append(cpu)
    g = get_accelerator_view_from_cpu_tensor(cpu).view(S // 64, 64, 656)
    graces.append(g)
    fc[f"model.layers.{i}.self_attn.attn"] = SimpleNamespace(
        impl=SimpleNamespace(_reuses_topk=(i == 1)), kv_cache=g)
fc["model.layers.0.self_attn.indexer.k_cache"] = SimpleNamespace(kv_cache=None)
ahc.build(fc)
C = ahc._CACHES
assert len(C.caches) == 2 and ahc.get(graces[1]) is None
print("rows", ahc.rows(), "entries", ahc.max_entries())
md = SimpleNamespace(attn_metadata=None)
ahc.get_forward_context = lambda: md
gen = torch.Generator().manual_seed(0)
scale = torch.tensor([0.05], device=dev)

def write(layer, slots, md_):
    md.attn_metadata = md_
    t = slots.numel()
    kv = torch.randn(t, 512, generator=gen).to(dev, torch.bfloat16)
    pe = torch.randn(t, 1, 64, generator=gen).to(dev, torch.bfloat16)
    ops.concat_and_cache_mla(kv, pe.squeeze(1), graces[layer], slots,
                             kv_cache_dtype="fp8_ds_mla", scale=scale)
    ahc.kv_update(graces[layer], kv, pe, slots, scale)
    return kv, pe

def check(layer, idx, rows):
    torch.cuda.synchronize()
    i, r = idx.cpu(), rows.cpu()
    buf = ahc.get(graces[layer]).buffer.view(-1, 656).cpu()
    src = cpus[layer].view(S, 656)
    m = i >= 0
    bad = int((~torch.all(buf[r[m].long()] == src[i[m].long()], dim=1)).sum())
    return bad + int((r[~m] != -1).sum())

bad = {"decode": 0, "cow": 0, "cow_unhooked": 0}
for step in range(300):
    T = 16  # this step's tokens: the same slots for every anchor
    new = torch.randint(0, 30000, (T,), generator=gen).to(dev)
    new[3] = -1
    for layer in (0, 2):
        hot = ahc.get(graces[layer])
        if step % 11 == 5:  # prefill: invalidates
            slots = torch.randint(0, 30000, (2000,), generator=gen).to(dev)
            write(layer, slots, SimpleNamespace(num_prefills=1, num_actual_tokens=2000))
            assert not hot.expect_map
        if step % 13 == 7:  # copy-on-write of 4 blocks
            src = torch.randint(0, 400, (4,), generator=gen)
            dst = torch.randint(0, 400, (4,), generator=gen) + 50
            g = graces[layer]
            g[dst.to(dev)] = g[src.to(dev)]
            hooked = layer == 0 or step % 2 == 0
            if hooked:
                ahc.invalidate_blocks(dst.to(dev))
        kv, pe = write(layer, new, SimpleNamespace(num_prefills=0, num_actual_tokens=T))
        assert hot.expect_map
        idx = torch.randint(0, 30000, (T, 768), generator=gen, dtype=torch.int32)
        idx[torch.rand(T, 768, generator=gen) < 0.3] = -1
        idx[:, :4] = new.cpu()[:, None].int()  # this step's own tokens
        idx[:, 4] = -1
        rows = hot.map(idx.to(dev), new)
        b = check(layer, idx, rows)
        if step % 13 == 7:
            bad["cow" if hooked else "cow_unhooked"] += b
            if not hooked:  # the control's stale rows must not leak on
                ahc.invalidate_blocks(dst.to(dev))
        else:
            bad["decode"] += b
print(bad)
assert bad["decode"] == 0 and bad["cow"] == 0
# guards
hot = ahc.get(graces[0])
write(0, torch.arange(16, device=dev), SimpleNamespace(num_prefills=0, num_actual_tokens=16))
try:
    write(0, torch.arange(16, device=dev), SimpleNamespace(num_prefills=0, num_actual_tokens=16))
    raise SystemExit("double kv_update not caught")
except RuntimeError as e:
    print("caught:", e)
hot.expect_map = False
try:
    hot.map(torch.zeros(16, 768, dtype=torch.int32, device=dev), torch.arange(16, device=dev))
    raise SystemExit("unpaired map not caught")
except AssertionError as e:
    print("caught:", e)
print("overflow", [int(h.ctl[6]) for h in C.caches.values()], "ok")

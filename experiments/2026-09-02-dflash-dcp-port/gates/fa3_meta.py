"""What does FA3's scheduler_metadata size actually depend on?"""
import torch
from vllm.vllm_flash_attn.flash_attn_interface import get_scheduler_metadata

dev="cuda"
def size(batch=4, q=8, k=4096, hq=8, hkv=1, d=128, splits=0, causal=False):
    cs = torch.full((batch,), k, dtype=torch.int32, device=dev)
    cu = torch.arange(batch+1, dtype=torch.int32, device=dev) * q
    m = get_scheduler_metadata(batch_size=batch, max_seqlen_q=q, max_seqlen_k=k,
        num_heads_q=hq, num_heads_kv=hkv, headdim=d, cache_seqlens=cs,
        qkv_dtype=torch.bfloat16, cu_seqlens_q=cu, page_size=64,
        causal=causal, num_splits=splits)
    return tuple(m.shape)

base = size()
print(f"base(batch=4,q=8,k=4096,hq=8,splits=0) = {base}")
print("\n-- vary batch:")
for b in (1,2,3,4,5,8,16): print(f"   batch={b:3d} -> {size(batch=b)}")
print("\n-- vary the others (batch fixed at 4):")
for name,kw in [("q=1",dict(q=1)),("q=16",dict(q=16)),("k=400000",dict(k=400000)),
                ("hq=32",dict(hq=32)),("hq=4",dict(hq=4)),("hkv=8",dict(hkv=8)),
                ("d=64",dict(d=64)),("splits=8",dict(splits=8)),
                ("splits=1",dict(splits=1)),("causal=True",dict(causal=True))]:
    print(f"   {name:12s} -> {size(**kw)}   {'SAME' if size(**kw)==base else 'DIFFERS'}")

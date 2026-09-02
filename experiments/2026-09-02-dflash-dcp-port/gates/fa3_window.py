import torch
from vllm.vllm_flash_attn.flash_attn_interface import get_scheduler_metadata
dev="cuda"
def size(win, k=1, mk=514, batch=4, q=8, hq=64, hkv=2, splits=0):
    cs=torch.full((batch,),k,dtype=torch.int32,device=dev)
    cu=torch.arange(batch+1,dtype=torch.int32,device=dev)*q
    m=get_scheduler_metadata(batch_size=batch,max_seqlen_q=q,max_seqlen_k=mk,
        num_heads_q=hq,num_heads_kv=hkv,headdim=128,cache_seqlens=cs,
        qkv_dtype=torch.bfloat16,cu_seqlens_q=cu,page_size=64,causal=False,
        window_size=win,num_splits=splits)
    return tuple(m.shape)
print("cache_seqlens=1, max_seqlen_k=514, varying window_size:")
for w in [(-1,-1),(2047,0),(2047,2047),(2048,2048),(2047,-1)]:
    print(f"   window={str(w):14s} -> {size(w)}")
print("\ncache_seqlens=0, max_seqlen_k=514:")
for w in [(-1,-1),(2047,0),(2047,2047)]:
    print(f"   window={str(w):14s} -> {size(w, k=0)}")

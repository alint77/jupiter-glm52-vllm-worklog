import torch
from vllm.vllm_flash_attn.flash_attn_interface import get_scheduler_metadata
dev="cuda"
def size(batch=4,q=8,k=4096,splits=0,causal=False,vals=None):
    cs = torch.full((batch,),k,dtype=torch.int32,device=dev) if vals is None \
         else torch.tensor(vals,dtype=torch.int32,device=dev)
    cu = torch.arange(batch+1,dtype=torch.int32,device=dev)*q
    m = get_scheduler_metadata(batch_size=batch,max_seqlen_q=q,max_seqlen_k=max(k,1),
        num_heads_q=64,num_heads_kv=8,headdim=128,cache_seqlens=cs,
        qkv_dtype=torch.bfloat16,cu_seqlens_q=cu,page_size=64,
        causal=causal,num_splits=splits)
    return tuple(m.shape)
print("cache_seqlens ALL ZERO      ->", size(k=0))
print("cache_seqlens all 1         ->", size(k=1))
print("cache_seqlens all 64        ->", size(k=64))
print("cache_seqlens all 4096      ->", size(k=4096))
print("cache_seqlens mixed [0,0,0,4096] ->", size(vals=[0,0,0,4096],k=4096))
print("cache_seqlens mixed [4096]*4     ->", size(vals=[4096]*4,k=4096))

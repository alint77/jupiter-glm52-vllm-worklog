"""Which parameter set reproduces the observed build shape of (5,)?
Observed at runtime: batch=4, max_q=8, kvlens=[1,1,1,1], maxdcp=514,
causal=False, num_splits=0, draft window (2047,0) -> symmetrized (2047,2047)."""
import torch, itertools
from vllm.vllm_flash_attn.flash_attn_interface import get_scheduler_metadata
dev="cuda"
def size(hkv, dt, win, d=128, page=64, mk=514, kv=1, batch=4, q=8, hq=64, splits=0):
    cs=torch.full((batch,),kv,dtype=torch.int32,device=dev)
    cu=torch.arange(batch+1,dtype=torch.int32,device=dev)*q
    try:
        m=get_scheduler_metadata(batch_size=batch,max_seqlen_q=q,max_seqlen_k=mk,
            num_heads_q=hq,num_heads_kv=hkv,headdim=d,cache_seqlens=cs,
            qkv_dtype=dt,cu_seqlens_q=cu,page_size=page,causal=False,
            window_size=win,num_splits=splits)
        return m.shape[0]
    except Exception as e:
        return f"ERR:{str(e)[:30]}"
fp8 = torch.float8_e4m3fn
print(f"{'hkv':>4} {'dtype':>10} {'window':>14} -> size   (looking for 5)")
for hkv, dt, win in itertools.product((2,8),(torch.bfloat16,fp8),((2047,2047),(2047,0),(-1,-1))):
    n=size(hkv,dt,win)
    mark = "   <== MATCHES OBSERVED 5" if n==5 else ""
    print(f"{hkv:>4} {str(dt).replace('torch.',''):>10} {str(win):>14} -> {n}{mark}")

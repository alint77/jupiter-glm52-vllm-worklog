"""Confirm which build/call parameter disagreement produces
'scheduler_metadata must have shape (metadata_size)'."""
import torch
from vllm.vllm_flash_attn.flash_attn_interface import (
    get_scheduler_metadata, flash_attn_varlen_func)

dev="cuda"; D=128; HQ=8; HKV=1; PAGE=64
def build(batch, q, k, causal, splits, hq=HQ):
    cs = torch.full((batch,), k, dtype=torch.int32, device=dev)
    cu = torch.arange(batch+1, dtype=torch.int32, device=dev)*q
    return get_scheduler_metadata(batch_size=batch, max_seqlen_q=q, max_seqlen_k=k,
        num_heads_q=hq, num_heads_kv=HKV, headdim=D, cache_seqlens=cs,
        qkv_dtype=torch.bfloat16, cu_seqlens_q=cu, page_size=PAGE,
        causal=causal, num_splits=splits)

def call(batch, q, k, causal, splits, sm, hq=HQ):
    nblocks = (k + PAGE - 1)//PAGE + 1
    kc = torch.zeros(nblocks, PAGE, HKV, D, dtype=torch.bfloat16, device=dev)
    vc = torch.zeros_like(kc)
    bt = torch.arange(batch*nblocks, dtype=torch.int32, device=dev).reshape(batch,-1) % nblocks
    qt = torch.randn(batch*q, hq, D, dtype=torch.bfloat16, device=dev)
    cu = torch.arange(batch+1, dtype=torch.int32, device=dev)*q
    sk = torch.full((batch,), k, dtype=torch.int32, device=dev)
    return flash_attn_varlen_func(q=qt, k=kc, v=vc, cu_seqlens_q=cu, max_seqlen_q=q,
        seqused_k=sk, max_seqlen_k=k, softmax_scale=D**-0.5, causal=causal,
        block_table=bt, scheduler_metadata=sm, fa_version=3, num_splits=splits)

B, Q, K = 4, 8, 4096
cases = [
    ("baseline: everything agrees",      dict(), dict()),
    ("causal differs (build True/call False)", dict(causal=True), dict()),
    ("causal differs (build False/call True)", dict(), dict(causal=True)),
    ("batch differs (build 8 / call 4)",  dict(batch=8), dict()),
    ("splits differs (build 1 / call 0)", dict(splits=1), dict()),
    ("splits differs (build 0 / call 8)", dict(), dict(splits=8)),
    ("heads differ (build 32 / call 8)",  dict(hq=32), dict()),
]
for name, bo, co in cases:
    ba = dict(batch=B, q=Q, k=K, causal=False, splits=0); ba.update(bo)
    ca = dict(batch=B, q=Q, k=K, causal=False, splits=0); ca.update(co)
    try:
        sm = build(**ba)
        call(**{k2: v for k2, v in ca.items()}, sm=sm)
        print(f"  OK        {name}   (built shape {tuple(sm.shape)})")
    except RuntimeError as e:
        msg = str(e).split("\n")[0][:70]
        print(f"  RAISES    {name}   (built {tuple(sm.shape)}) -> {msg}")

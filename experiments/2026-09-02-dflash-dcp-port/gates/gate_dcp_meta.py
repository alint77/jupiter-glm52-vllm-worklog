"""The DCP4 blocker: both defects, reproduced and fixed in isolation.

Runtime values observed from the instrumented arm (job 1623176):
  batch=4, max_q=8, cache_seqlens=[1,1,1,1], max_dcp_context_kv_len=514,
  causal=False, num_splits=0, draft layer window (2047, 0),
  draft num_kv_heads/rank=2, draft KV dtype bf16, target cache fp8_ds_mla.
"""
import sys, torch
from vllm.vllm_flash_attn.flash_attn_interface import (
    get_scheduler_metadata, flash_attn_varlen_func)

dev="cuda"; B,Q,D,HQ,HKV,PAGE = 4,8,128,64,2,64
KV_LEN, MAXK, RAW_WINDOW, CAUSAL = 1, 514, (2047,0), False
FP8, BF16 = torch.float8_e4m3fn, torch.bfloat16

def symmetrize(window, causal):  # verbatim from flash_attn.py:312
    non_causal = isinstance(causal, torch.Tensor) or causal is False
    if window is not None and window[0] >= 0 and window[1] == 0 and non_causal:
        return (window[0], window[0])
    return window

def build(qkv_dtype, window):
    cs = torch.full((B,), KV_LEN, dtype=torch.int32, device=dev)
    cu = torch.arange(B+1, dtype=torch.int32, device=dev) * Q
    return get_scheduler_metadata(batch_size=B, max_seqlen_q=Q, max_seqlen_k=MAXK,
        num_heads_q=HQ, num_heads_kv=HKV, headdim=D, cache_seqlens=cs,
        qkv_dtype=qkv_dtype, cu_seqlens_q=cu, page_size=PAGE, causal=CAUSAL,
        window_size=window, num_splits=0)

nb = (MAXK + PAGE - 1)//PAGE + 1
kc = torch.zeros(nb, PAGE, HKV, D, dtype=BF16, device=dev)   # draft cache is bf16
vc = torch.zeros_like(kc)
bt = (torch.arange(B*nb, dtype=torch.int32, device=dev).reshape(B,-1)) % nb
qt = torch.randn(B*Q, HQ, D, dtype=BF16, device=dev)
cu = torch.arange(B+1, dtype=torch.int32, device=dev) * Q
sk = torch.full((B,), KV_LEN, dtype=torch.int32, device=dev)

def call(sm, window):
    try:
        flash_attn_varlen_func(q=qt, k=kc, v=vc, cu_seqlens_q=cu, max_seqlen_q=Q,
            seqused_k=sk, max_seqlen_k=MAXK, softmax_scale=D**-0.5, causal=CAUSAL,
            window_size=list(window), block_table=bt, scheduler_metadata=sm,
            fa_version=3, num_splits=0, return_softmax_lse=True)
        return None
    except RuntimeError as e:
        return str(e).splitlines()[0][:52]

planned = symmetrize(RAW_WINDOW, CAUSAL)
# The builder ALWAYS schedules with the planned (symmetrized) window
# (flash_attn.py:529) -- neither fix changes that. What the two fixes change is
# the dtype the build uses, and the window the CALL passes.
rows = [
    ("BEFORE both fixes  (build fp8,  call raw window)",     FP8,  RAW_WINDOW),
    ("window fix only    (build fp8,  call planned window)", FP8,  planned),
    ("dtype fix only     (build bf16, call raw window)",     BF16, RAW_WINDOW),
    ("AFTER both fixes   (build bf16, call planned window)", BF16, planned),
]
print(f"draft window {RAW_WINDOW}, causal={CAUSAL} -> planned {planned}\n")
results = []
for label, dt, call_win in rows:
    sm = build(dt, planned)          # build window is fixed at planned
    err = call(sm, call_win)
    results.append(err is None)
    print(f"  {'OK    ' if err is None else 'RAISES'}  {label}")
    print(f"          built shape ({sm.shape[0]},){'' if err is None else '  -> ' + err}")

ok = results == [False, False, False, True]
print("\nGATE:", "PASS - only both fixes together resolve it" if ok
      else f"UNEXPECTED pattern {results}")
sys.exit(0 if ok else 1)

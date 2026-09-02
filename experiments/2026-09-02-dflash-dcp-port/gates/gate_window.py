"""The DCP4 blocker, reproduced and fixed in isolation.

Builds scheduler_metadata exactly as FlashAttentionMetadataBuilder does for a
non-causal sliding-window layer (window symmetrized), then calls FA3 the way
_forward_with_dcp did before the fix (raw window) and after (planned window).
"""
import sys, torch
from vllm.vllm_flash_attn.flash_attn_interface import (
    get_scheduler_metadata, flash_attn_varlen_func)

def _maybe_symmetrize_window(window, causal):
    # Verbatim from vllm/v1/attention/backends/flash_attn.py:312 (imported
    # directly it drags in the whole vllm config stack, which is slow here).
    non_causal = isinstance(causal, torch.Tensor) or causal is False
    if window is not None and window[0] >= 0 and window[1] == 0 and non_causal:
        return (window[0], window[0])
    return window


dev="cuda"; B,Q,D,HQ,HKV,PAGE = 4,8,128,64,2,64
KV_LEN, MAXK = 1, 514
RAW_WINDOW = (2047, 0)          # the draft layer's own causal window
CAUSAL = False                  # DFlash2's draft attention is non-causal

planned = _maybe_symmetrize_window(RAW_WINDOW, CAUSAL)
print(f"layer window {RAW_WINDOW}, causal={CAUSAL} -> builder plans with {planned}")

cs = torch.full((B,), KV_LEN, dtype=torch.int32, device=dev)
cu = torch.arange(B+1, dtype=torch.int32, device=dev) * Q
sm = get_scheduler_metadata(batch_size=B, max_seqlen_q=Q, max_seqlen_k=MAXK,
    num_heads_q=HQ, num_heads_kv=HKV, headdim=D, cache_seqlens=cs,
    qkv_dtype=torch.bfloat16, cu_seqlens_q=cu, page_size=PAGE,
    causal=CAUSAL, window_size=planned, num_splits=0)
print(f"scheduler_metadata built with planned window: shape {tuple(sm.shape)}")

nblocks = (MAXK + PAGE - 1)//PAGE + 1
kc = torch.zeros(nblocks, PAGE, HKV, D, dtype=torch.bfloat16, device=dev)
vc = torch.zeros_like(kc)
bt = (torch.arange(B*nblocks, dtype=torch.int32, device=dev).reshape(B,-1)) % nblocks
qt = torch.randn(B*Q, HQ, D, dtype=torch.bfloat16, device=dev)
sk = torch.full((B,), KV_LEN, dtype=torch.int32, device=dev)

def attempt(win, label):
    try:
        flash_attn_varlen_func(q=qt, k=kc, v=vc, cu_seqlens_q=cu, max_seqlen_q=Q,
            seqused_k=sk, max_seqlen_k=MAXK, softmax_scale=D**-0.5, causal=CAUSAL,
            window_size=list(win), block_table=bt, scheduler_metadata=sm,
            fa_version=3, num_splits=0, return_softmax_lse=True)
        print(f"  OK      {label}: window={win}")
        return True
    except RuntimeError as e:
        print(f"  RAISES  {label}: window={win} -> {str(e).splitlines()[0][:60]}")
        return False

print("\ncalling FA3 with that metadata:")
before = attempt(RAW_WINDOW, "BEFORE fix (raw self.sliding_window)")
after  = attempt(planned,    "AFTER  fix (attn_metadata.sliding_window)")
ok = (not before) and after
print("\nGATE:", "PASS - reproduces the failure and the fix resolves it" if ok else "INCONCLUSIVE")
sys.exit(0 if ok else 1)

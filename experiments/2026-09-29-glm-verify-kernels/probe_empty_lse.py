#!/usr/bin/env python3
"""What does FlashMLA sparse fp8 decode return for an all-invalid index row?

mask_empty_dcp_lse rewrites such rows' LSE to -inf before the DCP combine. The
combine already maps +inf/NaN LSEs to -inf and zeroes that rank's share, so
the mask is redundant iff the kernel returns +inf or NaN there (not a finite
value). Also records whether the output row is finite. 1 GPU.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tests/v1/attention"))
from test_flashmla_sparse_dcp import (  # noqa: E402
    HEAD_DIM,
    NUM_HEADS,
    TOPK,
    _pack_fp8_ds_mla_cache,
    _run_sparse_decode,
)

dev = torch.device("cuda")
torch.manual_seed(0)
n_ctx = 4096
kv_c = torch.randn(n_ctx, 512, dtype=torch.bfloat16, device=dev)
k_pe = torch.randn(n_ctx, 64, dtype=torch.bfloat16, device=dev)
cache = _pack_fp8_ds_mla_cache(kv_c, k_pe)
for T in (1, 8):
    q = torch.randn(T, NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    idx = torch.randint(0, n_ctx, (T, TOPK), dtype=torch.int32, device=dev)
    idx[0] = -1  # empty row
    if T > 1:
        idx[1, 1:] = -1  # one valid slot
        idx[2, 512:] = -1  # compacted prefix, -1 tail (the DCP layout)
    out, lse = _run_sparse_decode(q, cache, idx)
    e = lse[0]
    kind = ("+inf" if torch.isposinf(e).all() else "NaN" if e.isnan().all()
            else "-inf" if torch.isneginf(e).all() else f"finite/mixed {e[:4].tolist()}")
    print(f"T={T}: empty-row lse {kind}; empty-row out finite: "
          f"{bool(out[0].isfinite().all())}; other rows lse finite: "
          f"{bool(lse[1:].isfinite().all()) if T > 1 else '-'}")

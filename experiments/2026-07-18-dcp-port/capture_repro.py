"""Locally reproduce the DCP FULL-graph capture crash, one op at a time."""

import sys
import torch

torch.cuda.init()
device = torch.device("cuda")

WHICH = sys.argv[1] if len(sys.argv) > 1 else "all"


def try_capture(name, fn, warmups=2):
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g):
            fn()
        g.replay()
        torch.cuda.synchronize()
        print(f"[OK]    {name}: captured and replayed")
    except Exception as e:
        print(f"[FAIL]  {name}: {type(e).__name__}: {e}")


if WHICH in ("topk", "all"):
    from vllm.model_executor.kernels.attention.dsa.dcp_indexer_cutedsl import (
        stable_topk_from_gathered_candidates_cutedsl,
    )

    gathered = torch.randn(4, 8192, 2, dtype=torch.float32, device=device)
    gathered[..., 1] = torch.randint(0, 400000, (4, 8192), device=device).float()
    out = torch.empty(4, 2048, dtype=torch.int32, device=device)
    try_capture(
        "cutedsl stable_topk",
        lambda: stable_topk_from_gathered_candidates_cutedsl(gathered, 2048, out=out),
    )

if WHICH in ("pack", "all"):
    from vllm.model_executor.kernels.attention.dsa.dcp_indexer_cutedsl import (
        pack_dcp_topk_candidates_cutedsl,
    )

    logits = torch.randn(4, 100000, dtype=torch.float32, device=device)
    topk_indices = torch.randint(0, 100000, (4, 2048), device=device).to(torch.int32)
    packed = torch.empty(4, 2048, 2, dtype=torch.float32, device=device)
    try_capture(
        "triton pack",
        lambda: pack_dcp_topk_candidates_cutedsl(
            logits, topk_indices, packed, 0, 4, 1, None
        ),
    )

if WHICH in ("filter", "all"):
    from vllm.v1.attention.backends.mla.sparse_utils import (
        triton_filter_and_convert_dcp_index,
    )

    req_id = torch.zeros(4, dtype=torch.int32, device=device)
    block_table = (
        torch.arange(16, dtype=torch.int32, device=device).unsqueeze(0).contiguous()
    )
    tokens = torch.randint(0, 4096, (4, 2048), device=device).to(torch.int32)
    try_capture(
        "dcp index filter",
        lambda: triton_filter_and_convert_dcp_index(
            req_id, block_table, tokens, dcp_size=4, dcp_rank=0,
            cp_kv_cache_interleave_size=1, BLOCK_SIZE=64, NUM_TOPK_TOKENS=2048,
            return_valid_counts=True,
        ),
    )

if WHICH in ("flashmla", "all"):
    sys.path.insert(0, "tests/v1/attention")
    from test_flashmla_sparse_dcp import (
        _pack_fp8_ds_mla_cache,
        _run_sparse_decode,
        NUM_HEADS,
        HEAD_DIM,
        TOPK,
    )

    kv_c = torch.randn(4096, 512, dtype=torch.bfloat16, device=device)
    k_pe = 0.5 * torch.randn(4096, 64, dtype=torch.bfloat16, device=device)
    q = torch.randn(4, NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    cache = _pack_fp8_ds_mla_cache(kv_c, k_pe)
    idx = torch.arange(0, TOPK, dtype=torch.int32, device=device).unsqueeze(0)
    idx = idx.repeat(4, 1).contiguous()
    try_capture("flashmla sparse fp8", lambda: _run_sparse_decode(q, cache, idx))

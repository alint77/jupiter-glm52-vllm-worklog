#!/usr/bin/env python3
"""Single-tile index remap (upstream #50365 + in-kernel -1 tail) vs the
frozen multi-tile atomic version (sparse_utils_ref.py), 1 GPU.

DCP filter + compaction: per row the valid count and the *set* of the valid
prefix must match (the old prefix order depended on atomic arrival), and the
tail must be all -1. Non-DCP conversion: exact. Then CUDA-graph timings of
78 calls each at the decode shape (8 tokens x 2048).
"""

import importlib.util
from pathlib import Path

import torch

from vllm.v1.attention.backends.mla import sparse_utils as new

spec = importlib.util.spec_from_file_location(
    "sparse_utils_ref", Path(__file__).with_name("sparse_utils_ref.py"))
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)

dev = torch.device("cuda")
T, K, BS, NBLK = 8, 2048, 64, 1600  # 400K / 4 ranks / 64-token blocks
gen = torch.Generator(device=dev).manual_seed(0)


def case(ctx: int, frac_invalid: float):
    req = torch.zeros(T, dtype=torch.int32, device=dev)
    bt = torch.randperm(NBLK * 4, generator=gen, device=dev)[:NBLK].to(torch.int32)
    bt = bt.view(1, NBLK)
    idx = torch.randint(0, ctx, (T, K), generator=gen, device=dev, dtype=torch.int32)
    idx[torch.rand((T, K), generator=gen, device=dev) < frac_invalid] = -1
    if ctx < K:  # short context: tail of -1, as the indexer emits
        idx[:, ctx:] = -1
    return req, bt, idx


def check_dcp(req, bt, idx, rank):
    kw = dict(dcp_size=4, dcp_rank=rank, cp_kv_cache_interleave_size=1,
              BLOCK_SIZE=BS, NUM_TOPK_TOKENS=K, return_valid_counts=True)
    o_new, c_new = new.triton_filter_and_convert_dcp_index(req, bt, idx, **kw)
    o_ref, c_ref = ref.triton_filter_and_convert_dcp_index(req, bt, idx, **kw)
    assert torch.equal(c_new, c_ref), "valid counts differ"
    for t in range(T):
        n = int(c_ref[t])
        a, b = o_new[t, :n].sort().values, o_ref[t, :n].sort().values
        assert torch.equal(a, b), f"row {t}: valid set differs"
        assert (o_new[t, :n] >= 0).all() and (o_new[t, n:] == -1).all(), "tail"
    return o_new, o_ref


def check_plain(req, bt, idx):
    for counts in (False, True):
        kw = dict(BLOCK_SIZE=BS, NUM_TOPK_TOKENS=K, return_valid_counts=counts)
        a = new.triton_convert_req_index_to_global_index(req, bt, idx, **kw)
        b = ref.triton_convert_req_index_to_global_index(req, bt, idx, **kw)
        for x, y in zip(a if counts else (a,), b if counts else (b,)):
            assert torch.equal(x, y), f"plain convert differs (counts={counts})"


for ctx in (100, 3000, 60000, 400000):
    for frac in (0.0, 0.3):
        req, bt, idx = case(ctx, frac)
        check_plain(req, bt, idx)
        for rank in range(4):
            check_dcp(req, bt, idx, rank)
print("index remap: equal (DCP sets/counts/tail, plain exact)")


def timed(fn):
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(78):
            fn()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(50):
        g.replay()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / 50 * 1000 / 78


req, bt, idx = case(60000, 0.0)
kw = dict(dcp_size=4, dcp_rank=1, cp_kv_cache_interleave_size=1, BLOCK_SIZE=BS,
          NUM_TOPK_TOKENS=K, return_valid_counts=True)
for label, mod in (("ref (zeros + full_like + tiled atomics)", ref), ("new", new)):
    us = timed(lambda: mod.triton_filter_and_convert_dcp_index(req, bt, idx, **kw))
    print(f"{label:42s} {us:6.2f} us per layer")

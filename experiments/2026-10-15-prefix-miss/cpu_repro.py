"""CPU repro of append-only prefix misses with the real KVCacheManager:
cold A (chunked prefill + a few decode steps), A+tail (B), B+tail (C), at a
GLM-like layout. Prints each request's prefix hit.

    cpu_repro.py [--layout unitary|mla_idx|mla_idx_drafter] [--dcp 4] [--eagle]
"""
import argparse

import torch

from tests.v1.core.utils import create_requests
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    MLAAttentionSpec,
)


def spec(head, shardable=True):
    return MLAAttentionSpec(block_size=64, num_kv_heads=1, head_size=head,
                            dtype=torch.uint8, dcp_shardable=shardable)


def layout(name, eagle):
    groups = [KVCacheGroupSpec(["mla"], spec(656))]
    if name in ("mla_idx", "mla_idx_drafter", "mla_idx_drafter_repl"):
        groups.append(KVCacheGroupSpec(["idx"], spec(132)))
    if name == "mla_idx_drafter_repl":
        groups += [KVCacheGroupSpec(["d_mla"], spec(656, False), is_eagle_group=eagle),
                   KVCacheGroupSpec(["d_idx"], spec(132, False), is_eagle_group=eagle)]
    if name == "mla_idx_drafter":
        groups += [KVCacheGroupSpec(["d_mla"], spec(656), is_eagle_group=eagle),
                   KVCacheGroupSpec(["d_idx"], spec(132), is_eagle_group=eagle)]
    return KVCacheConfig(num_blocks=20000, kv_cache_tensors=[], kv_cache_groups=groups)


def run(mgr, req, chunk=4096, decode=16, spec_k=3):
    blocks, hit, *_ = mgr.get_computed_blocks(req)
    n = req.num_tokens
    first = min(chunk, n - hit)
    assert mgr.allocate_slots(req, first, hit, blocks) is not None
    req.num_computed_tokens = hit + first
    while req.num_computed_tokens < n:
        step = min(chunk, n - req.num_computed_tokens)
        assert mgr.allocate_slots(req, step) is not None
        req.num_computed_tokens += step
    for _ in range(decode):
        req.append_output_token_ids(1)
        assert mgr.allocate_slots(req, 1, num_lookahead_tokens=spec_k) is not None
        req.num_computed_tokens += 1
    mgr.free(req)
    return hit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", default="mla_idx_drafter")
    ap.add_argument("--dcp", type=int, default=4)
    ap.add_argument("--eagle", action="store_true")
    ap.add_argument("--n", type=int, default=60000)
    ap.add_argument("--hash-bs", type=int, default=0)
    a = ap.parse_args()
    hb = a.hash_bs or 64 * a.dcp
    mgr = KVCacheManager(layout(a.layout, a.eagle), max_model_len=400000,
                         scheduler_block_size=64 * a.dcp, hash_block_size=hb,
                         use_eagle=a.eagle, dcp_world_size=a.dcp)
    base = create_requests(1, num_tokens=a.n + 2000, block_size=hb)[0]
    toks = list(base.prompt_token_ids)
    for i, (name, n) in enumerate((("A", a.n), ("B", a.n + 254), ("C", a.n + 1047))):
        (r,) = create_requests(1, num_tokens=n, block_size=hb, req_ids=[name])
        r.prompt_token_ids[:] = toks[:n]
        r._all_token_ids[:] = toks[:n]
        r.block_hashes.clear()
        r.update_block_hashes()
        hit = run(mgr, r)
        print(f"{a.layout} dcp={a.dcp} eagle={a.eagle} {name}: {n} tokens, hit {hit}")


if __name__ == "__main__":
    main()

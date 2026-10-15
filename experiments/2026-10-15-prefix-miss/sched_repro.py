"""CPU repro through the real Scheduler (not just KVCacheManager): GLM-like KV
groups, DCP4 block math, chunked prefill 4096, interleave scheduler, async
scheduling, spec-decode lookahead, V2-runner flag. Replays cold A, A+tail,
A+tail+tail; prints each request's prefix hit (num_computed_tokens at admit).

    sched_repro.py [--async] [--spec 3] [--dcp 4] [--layout ...] [--n 60000]
"""
import argparse
import os

import torch

import vllm.envs as envs
from tests.v1.core.utils import create_requests
from vllm.config import (CacheConfig, ModelConfig, ParallelConfig, SchedulerConfig,
                         SpeculativeConfig, VllmConfig)
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, MLAAttentionSpec
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
from vllm.v1.structured_output import StructuredOutputManager


def spec(head):
    return MLAAttentionSpec(block_size=64, num_kv_heads=1, head_size=head,
                            dtype=torch.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--async", dest="async_", action="store_true")
    ap.add_argument("--spec", type=int, default=0)
    ap.add_argument("--dcp", type=int, default=4)
    ap.add_argument("--interleave", type=int, default=1)
    ap.add_argument("--groups", default="mla,idx,dmla,didx")
    ap.add_argument("--n", type=int, default=60000)
    ap.add_argument("--max-tokens", type=int, default=16)
    a = ap.parse_args()
    envs.VLLM_SCHED_INTERLEAVE_PREFILL = a.interleave
    mc = ModelConfig(model="facebook/opt-125m", trust_remote_code=True,
                     dtype="float16", seed=42, max_model_len=400000)
    sc = SchedulerConfig(max_num_seqs=4, max_num_batched_tokens=4096,
                         max_model_len=400000, enable_chunked_prefill=True,
                         async_scheduling=a.async_, is_encoder_decoder=False,
                         watermark=0.0)
    cc = CacheConfig(block_size=64, gpu_memory_utilization=0.9, cache_dtype="auto",
                     enable_prefix_caching=True)
    pc = ParallelConfig()
    spc = (SpeculativeConfig(model="ngram", num_speculative_tokens=a.spec,
                             method="ngram", prompt_lookup_max=a.spec,
                             prompt_lookup_min=1) if a.spec else None)
    vc = VllmConfig(scheduler_config=sc, model_config=mc, cache_config=cc,
                    parallel_config=pc, speculative_config=spc)
    vc.parallel_config.decode_context_parallel_size = a.dcp
    heads = {"mla": 656, "idx": 132, "dmla": 656, "didx": 132}
    groups = [KVCacheGroupSpec([g], spec(heads[g]), is_eagle_group=g.startswith("d"))
              for g in a.groups.split(",")]
    kvc = KVCacheConfig(num_blocks=20000, kv_cache_tensors=[], kv_cache_groups=groups)
    cc.num_gpu_blocks = 20000
    cls = AsyncScheduler if a.async_ else Scheduler
    s = cls(vllm_config=vc, kv_cache_config=kvc, block_size=64 * a.dcp, log_stats=True,
            structured_output_manager=StructuredOutputManager(vc))
    s.use_v2_model_runner = True
    hb = 64 * a.dcp
    base = create_requests(1, num_tokens=a.n + 3000, block_size=hb)[0]
    toks = list(base.prompt_token_ids)
    pending = None
    for name, n in (("A", a.n), ("B", a.n + 254), ("C", a.n + 1047), ("D", a.n + 2200)):
        (r,) = create_requests(1, num_tokens=n, block_size=hb, req_ids=[name],
                               max_tokens=a.max_tokens)
        r.prompt_token_ids[:] = toks[:n]
        r._all_token_ids[:] = toks[:n]
        r.block_hashes.clear()
        r.update_block_hashes()
        s.add_request(r)
        hit = None
        steps = 0
        while name in s.requests and not s.requests[name].is_finished():
            out = s.schedule()
            steps += 1
            if hit is None and name in out.num_scheduled_tokens:
                hit = s.requests[name].num_computed_tokens - out.num_scheduled_tokens[name]
            ids = list(out.num_scheduled_tokens)
            if any(v <= 0 for v in out.num_scheduled_tokens.values()):
                print("ZERO", name, steps, dict(out.num_scheduled_tokens), {k: (s.requests[k].num_computed_tokens, s.requests[k].num_tokens) for k in ids})
            sampled = []
            for rid in ids:
                rq = s.requests[rid]
                done_prompt = rq.num_computed_tokens >= rq.num_prompt_tokens
                sampled.append([7] if done_prompt else [])
            mro = ModelRunnerOutput(req_ids=ids,
                                    req_id_to_index={x: i for i, x in enumerate(ids)},
                                    sampled_token_ids=sampled, logprobs=None,
                                    prompt_logprobs_dict={}, pooler_output=[])
            if a.async_:
                if pending is not None:
                    s.update_from_output(*pending)
                pending = (out, mro)
            else:
                s.update_from_output(out, mro)
            if a.spec and ids:
                s.update_draft_token_ids(DraftTokenIds(ids, [[7] * a.spec for _ in ids]))
            if steps > 5000:
                raise RuntimeError("no progress")
        if pending is not None:
            s.update_from_output(*pending)
            pending = None
        st = s.kv_cache_manager.make_prefix_cache_stats()
        print(f"{name}: {n} tokens, hit at admit {hit}, steps {steps}, "
              f"free blocks {s.kv_cache_manager.block_pool.get_num_free_blocks()}")


if __name__ == "__main__":
    main()

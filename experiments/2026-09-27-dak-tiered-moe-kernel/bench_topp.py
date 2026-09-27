#!/usr/bin/env python3
"""Top-p masking of the verify logits: the torch sort path vs the Triton one.

    bench_topp.py --vocab 152576

Rows 1..16 (DFlash k=7 verifies 7 draft rows), fp32 logits, p = 0.95; each
case is a CUDA graph of 20 calls, graph-replay wall clock per call.
"""

import argparse

import torch

from vllm.v1.sample.ops.topk_topp_sampler import (
    apply_top_k_top_p_pytorch,
    apply_top_k_top_p_triton,
)


def wall_us(fn) -> float:
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(20):
            fn()
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    best = 1e9
    for _ in range(10):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        g.replay()
        e.record()
        torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e) * 1000 / 20)
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, required=True)
    args = ap.parse_args()
    torch.manual_seed(0)
    for rows in (1, 4, 7, 8, 16):
        logits = torch.randn(rows, args.vocab, device="cuda") * 4
        p = torch.full((rows,), 0.95, device="cuda")
        work = logits.clone()

        def torch_path():
            work.copy_(logits)
            apply_top_k_top_p_pytorch(work, None, p)

        def triton_path():
            work.copy_(logits)
            apply_top_k_top_p_triton(work, None, p)

        a = apply_top_k_top_p_pytorch(logits.clone(), None, p)
        b = apply_top_k_top_p_triton(logits.clone(), None, p)
        kept_a, kept_b = torch.isfinite(a).sum(1), torch.isfinite(b).sum(1)
        print(f"rows {rows:2d}: torch {wall_us(torch_path):7.1f} us, triton {wall_us(triton_path):7.1f} us, "
              f"kept tokens torch {kept_a.tolist()[:3]} triton {kept_b.tolist()[:3]}", flush=True)


if __name__ == "__main__" and not __import__("os").environ.get("RENORM"):
    main()


def renorm_main(vocab: int) -> None:
    """softmax + FlashInfer's sort-free top_p_renorm_probs vs the torch path
    followed by softmax (what the rejection sampler computes)."""
    from flashinfer.sampling import top_p_renorm_probs
    torch.manual_seed(0)
    for rows in (7, 8):
        logits = torch.randn(rows, vocab, device="cuda") * 4
        p = torch.full((rows,), 0.95, device="cuda")
        work = logits.clone()

        def torch_path():
            work.copy_(logits)
            apply_top_k_top_p_pytorch(work, None, p).softmax(-1, dtype=torch.float32)

        def renorm():
            top_p_renorm_probs(logits.softmax(-1, dtype=torch.float32), p)

        a = apply_top_k_top_p_pytorch(logits.clone(), None, p).softmax(-1, dtype=torch.float32)
        b = top_p_renorm_probs(logits.softmax(-1, dtype=torch.float32), p)
        diff = (a - b).abs().max().item()
        support = ((a > 0) != (b > 0)).sum(1).tolist()
        print(f"rows {rows}: torch+softmax {wall_us(torch_path):7.1f} us, softmax+renorm "
              f"{wall_us(renorm):7.1f} us, max |dprob| {diff:.2e}, support mismatch {support}", flush=True)


if __name__ == "__main__" and __import__("os").environ.get("RENORM"):
    renorm_main(152576)

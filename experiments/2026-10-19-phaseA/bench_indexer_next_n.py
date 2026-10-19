"""Paged MQA logits at MTP3 (next_n=4) on SM90: native vs flattened (#52164).

Flattened = what our indexer does today: each of the B x 4 query rows is its
own single-token request (next_n=1) with its own context length and a copy of
the block-table row. Native = B requests x next_n=4 with schedule metadata
sized for the 2-CTA multicast cluster (num_sms // 2). GLM-5.3 indexer shapes:
32 heads x 128, fp8 cache with a 4-byte scale per token, block 64.

Checks the logits agree on every valid position (and the top-2048 sets match),
then times both under CUDA graphs with the torch profiler (kernel time only).

  bench_indexer_next_n.py [--ctx 12500,50000,100000] [--batch 1,4]
"""

import argparse

import torch
from torch.profiler import ProfilerActivity, profile

from vllm.utils import deep_gemm as dg

H, D, BLOCK, NEXT_N = 32, 128, 64, 4


def make(b, ctx, device):
    nblk = (ctx + BLOCK - 1) // BLOCK
    num_blocks = b * nblk + 8
    kv = torch.randint(0, 255, (num_blocks, BLOCK, 1, D + 4), dtype=torch.uint8,
                       device=device)
    # sane fp8 values and positive scales
    kv[..., :D] = (torch.randn(num_blocks, BLOCK, 1, D, device=device) * 0.5).to(
        torch.float8_e4m3fn).view(torch.uint8)
    kv[..., D:] = torch.rand(num_blocks, BLOCK, 1, 1, device=device).add(0.5).view(
        torch.uint8).view(num_blocks, BLOCK, 1, 4)
    perm = torch.randperm(num_blocks - 8, device=device)[: b * nblk].int()
    bt = perm.view(b, nblk)
    q = (torch.randn(b, NEXT_N, H, D, device=device) * 0.5).to(torch.float8_e4m3fn)
    w = torch.randn(b * NEXT_N, H, device=device, dtype=torch.float32)
    # row j of request i sees ctx - NEXT_N + j + 1 tokens (causal within the step)
    lens = (ctx - NEXT_N + 1 + torch.arange(NEXT_N, device=device)).int()
    lens = lens.unsqueeze(0).repeat(b, 1).contiguous()
    return kv, bt, q, w, lens


def native(kv, bt, q, w, lens, max_len, sms):
    meta = dg._get_paged_mqa_logits_metadata_impl(lens, BLOCK, sms // 2)
    return lambda: dg.fp8_fp4_paged_mqa_logits(
        (q, None), kv, w, lens, bt, meta, max_model_len=max_len, clean_logits=False)


def flat(kv, bt, q, w, lens, max_len, sms):
    b = q.shape[0]
    qf = q.reshape(b * NEXT_N, 1, H, D).contiguous()
    lf = lens.reshape(-1, 1).contiguous()
    btf = bt.repeat_interleave(NEXT_N, dim=0).contiguous()
    meta = dg._get_paged_mqa_logits_metadata_impl(lf, BLOCK, sms)
    return lambda: dg.fp8_fp4_paged_mqa_logits(
        (qf, None), kv, w, lf, btf, meta, max_model_len=max_len, clean_logits=False)


def kernel_us(fn, reps=50):
    out = fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    g.replay()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        g.replay()
        torch.cuda.synchronize()
    ev = [e for e in p.events() if e.device_type.name == "CUDA"
          and "mqa_logits" in e.name and "metadata" not in e.name]
    ts = sorted(e.device_time for e in ev)
    return ts[len(ts) // 2], out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", default="12500,50000,100000")
    ap.add_argument("--batch", default="1,4")
    a = ap.parse_args()
    dg._lazy_init()
    dev = torch.device("cuda")
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    print(f"{torch.cuda.get_device_name()} sms={sms}")
    for b in map(int, a.batch.split(",")):
        for ctx in map(int, a.ctx.split(",")):
            torch.manual_seed(0)
            kv, bt, q, w, lens = make(b, ctx, dev)
            max_len = ((ctx + BLOCK - 1) // BLOCK) * BLOCK
            tn, ln = kernel_us(native(kv, bt, q, w, lens, max_len, sms))
            tf, lf = kernel_us(flat(kv, bt, q, w, lens, max_len, sms))
            valid = torch.arange(max_len, device=dev)[None, :] < lens.reshape(-1, 1)
            d = (ln - lf).abs()[valid]
            same = bool(torch.equal(ln[valid], lf[valid]))
            neg = torch.finfo(torch.float32).min
            k = min(2048, ctx - NEXT_N + 1)
            tk_n = ln.masked_fill(~valid, neg).topk(k, dim=1).indices.sort(1).values
            tk_f = lf.masked_fill(~valid, neg).topk(k, dim=1).indices.sort(1).values
            print(f"M={b * NEXT_N:3d} ctx/rank={ctx:7d}: native {tn:7.2f} us  "
                  f"flat {tf:7.2f} us  x{tf / tn:4.2f}  logits bitwise={same} "
                  f"max|d|={d.max().item():.3g}  top-k sets equal="
                  f"{bool(torch.equal(tk_n, tk_f))}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Hash FlashMLA sparse decode outputs of one _flashmla_C build, for a bitwise
comparison between builds (one library per process):

    python regs_hashes.py --lib A.so --out a.json
    python regs_hashes.py --lib B.so --out b.json
    python regs_hashes.py --compare a.json b.json

Every T in 1..64 plus 100 and 140, widths 64/576/768/2048, adversarial live
count patterns (serving-like, block boundaries, single key, all empty,
overflow, full), under FlashMLA's planner and under the balanced early-stop
schedule vLLM builds (same rule as dcp_decode_sched_meta). The cache has
zero groups, wide scale ranges and large q, and the index rows have repeats.
"""

import argparse
import hashlib
import json

import torch

H, D_QK, NUM_CTX = 64, 576, 3000


def pack(kv_c, k_pe):
    n = kv_c.shape[0]
    pad = (n + 63) // 64 * 64
    cache = torch.zeros(pad, 656, dtype=torch.uint8, device=kv_c.device)
    g = kv_c.float().view(n, 4, 128)
    s = g.abs().amax(-1) / 448.0
    s = torch.where(s > 0, s, torch.ones_like(s))
    cache[:n, :512] = (g / s.unsqueeze(-1)).to(torch.float8_e4m3fn).view(n, 512).view(
        torch.uint8)
    cache[:n, 512:528] = s.contiguous().view(torch.uint8)
    cache[:n, 528:] = k_pe.contiguous().view(torch.uint8)
    return cache.view(-1, 64, 1, 656)


def counts_for(pattern, t, width, g):
    if pattern == "live":
        hi = min(width, 540)
        c = torch.randint(hi * 5 // 6, hi + 1, (t,), generator=g).tolist()
        return [0 if i % 5 == 3 else x for i, x in enumerate(c)]
    if pattern == "boundary":
        e = [1, 63, 64, 65, 127, 128, 129, width - 1, width, 0]
        return [e[(i * 3) % len(e)] for i in range(t)]
    if pattern == "single":
        return [1 if i == t - 1 else 0 for i in range(t)]
    if pattern == "empty":
        return [0] * t
    if pattern == "overflow":
        return [width + 1 + i if i % 2 == 0 else width // 3 for i in range(t)]
    if pattern == "full":
        return [width] * t
    raise ValueError(pattern)


def balanced(counts, width, parts, dev):
    live = min(max(counts), width)
    nb = max((live + 63) // 64, 1)
    longest = -(-nb // parts)
    splits = -(-nb // longest)
    meta = torch.zeros((parts, 8), dtype=torch.int32)
    meta[:, 0] = 1
    for i in range(splits):
        meta[i] = torch.tensor([0, 0, nb * i // splits, nb * (i + 1) // splits, i,
                                int(splits > 1), int(splits > 1), 0])
    return meta.to(dev), torch.tensor([0, splits], dtype=torch.int32, device=dev)


def run(lib, out_path):
    torch.ops.load_library(lib)
    fmla = torch.ops._flashmla_C
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    torch.manual_seed(0)
    kv_c = 3.0 * torch.randn(NUM_CTX, 512, device=dev)
    kv_c[::7] = 0
    kv_c[1::11, :128] *= 200
    cache = pack(kv_c.bfloat16(), (0.5 * torch.randn(NUM_CTX, 64, device=dev)).bfloat16())
    hashes = {}
    for t in list(range(1, 65)) + [100, 140]:
        q = (2.0 * torch.randn(1, t, H, D_QK, device=dev)).bfloat16()
        parts = max(sms // t, 1)
        for width in (64, 576, 768, 2048):
            g = torch.Generator().manual_seed(t * 7919 + width)
            for pattern in ("live", "boundary", "single", "empty", "overflow", "full"):
                counts = counts_for(pattern, t, width, g)
                idx = torch.full((t, width), -1, dtype=torch.int32)
                for i, c in enumerate(counts):
                    n = min(c, width)
                    if n:
                        idx[i, :n] = torch.randint(0, NUM_CTX, (n,), generator=g)
                        idx[i, n - 1] = NUM_CTX - 1
                idx = idx.to(dev).unsqueeze(0)
                for sched_name in ("planner", "balanced"):
                    sched = ((None, None) if sched_name == "planner"
                             else balanced(counts, width, parts, dev))
                    out, lse, _, _ = fmla.sparse_decode_fwd(
                        q, cache, idx, None, None, *sched, None, None, None, 512,
                        D_QK ** -0.5, None)
                    h = hashlib.sha256(out.cpu().view(torch.int16).numpy().tobytes())
                    h.update(lse.cpu().numpy().tobytes())
                    hashes[f"T{t}:W{width}:{pattern}:{sched_name}"] = h.hexdigest()
        print(f"T={t} done", flush=True)
    json.dump(hashes, open(out_path, "w"), indent=0)
    print(f"HASHES DONE {len(hashes)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib")
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2)
    args = ap.parse_args()
    if args.compare:
        a, b = (json.load(open(p)) for p in args.compare)
        assert a.keys() == b.keys(), "different config sets"
        bad = [k for k in a if a[k] != b[k]]
        print(f"{len(a)} configs, {len(bad)} differ", *bad[:20], sep="\n")
        raise SystemExit(1 if bad else 0)
    run(args.lib, args.out)


if __name__ == "__main__":
    main()

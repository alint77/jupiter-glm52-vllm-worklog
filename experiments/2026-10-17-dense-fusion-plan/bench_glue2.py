"""qkv_a -> glue -> q_b plus the layer's two fp8_ds_mla KV writes: inductor
glue + 2x concat_and_cache_mla (as served) vs glue.cu's glue2 (one kernel).

Correctness: glue2's normalized q_c and both cache entries against the
inductor reference + ops.concat_and_cache_mla (mismatched elements / bytes),
with a padded slot (-1). Timing: as bench_qchain.py, one CUDA graph of REPS
chains over NW rotating weight sets. Variants:
  served   qkv_a -> inductor RoPE/norms -> q_b -> concat_and_cache x2
  glue     qkv_a -> glue (PDL) -> q_b -> concat_and_cache x2
  glue2    qkv_a -> glue2 (norms, RoPE, both KV writes; PDL) -> q_b
  floor    qkv_a -> q_b on a stale input
    bench_glue2.py [--m 4 8 16 32] [--reps 64] [--nw 8]
"""

import argparse
from pathlib import Path

import torch
import torch._inductor.config as ic
from torch.utils.cpp_extension import load

from vllm import _custom_ops as ops
from vllm.model_executor.layers import decode_gemm as dg

ap = argparse.ArgumentParser()
ap.add_argument("--m", type=int, nargs="+", default=[4, 8, 16, 32])
ap.add_argument("--reps", type=int, default=64)
ap.add_argument("--nw", type=int, default=8)
ap.add_argument("--iters", type=int, default=20)
a = ap.parse_args()

here = Path(__file__).resolve().parent
build = Path("/e/fscratch/profound") / Path.home().name / "caches/torch_extensions/qkv_glue"
build.mkdir(parents=True, exist_ok=True)
ext = load(name="qkv_glue", sources=[str(here / "glue.cu")],
           extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
           build_directory=str(build))
_gemm = dg._extension().gemm


def gemm(x, w, *cfg):
    """decode_gemm with the served config, or cuBLAS where it falls back."""
    return _gemm(x, w, *cfg) if cfg[0] is not None else torch.nn.functional.linear(x, w)
ic.combo_kernels = True
dev = torch.device("cuda:0")
torch.manual_seed(0)
EPS = 1e-5
QL, KVL, PE, H = 2048, 512, 64, 6144
NQKV, NQB, ROW = QL + KVL + PE, 4096, 656
bf = dict(device=dev, dtype=torch.bfloat16)


def rms(x, w):
    x = x.float()
    r = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)
    return ((x * r) * w.float()).to(torch.bfloat16)


def ref(qkv, pos, cos_sin, wq, wkv):
    q_c, kv = qkv.split([QL, KVL + PE], -1)
    kv_c, k_pe = kv.split([KVL, PE], -1)
    cos, sin = cos_sin[pos].float().chunk(2, -1)
    x = k_pe.float().unflatten(-1, (PE // 2, 2))
    x0, x1 = x[..., 0], x[..., 1]
    kpe = torch.stack((x0 * cos - x1 * sin, x1 * cos + x0 * sin), -1).flatten(-2)
    return rms(q_c, wq), rms(kv_c, wkv), kpe.to(torch.bfloat16)


ref_c = torch.compile(ref, dynamic=False)
k_scale = torch.ones(1, device=dev, dtype=torch.float32)


def served_kv(kvn, kpe, c1, s1, c2, s2):
    ops.concat_and_cache_mla(kvn, kpe, c1, s1, "fp8_ds_mla", k_scale)
    ops.concat_and_cache_mla(kvn, kpe, c2, s2, "fp8_ds_mla", k_scale)


def glue(qkv, pos, cos_sin, wq, wkv):
    m = qkv.shape[0]
    qn = torch.empty(m, QL, **bf)
    kvn = torch.empty(m, KVL, **bf)
    kpe = torch.empty(m, PE, **bf)
    ext.glue(qkv, pos, cos_sin, wq, wkv, qn, kvn, kpe, EPS, True)
    return qn, kvn, kpe


def glue2(qkv, pos, cos_sin, wq, wkv, c1, s1, c2, s2):
    qn = torch.empty(qkv.shape[0], QL, **bf)
    ext.glue2(qkv, pos, cos_sin, wq, wkv, qn, c1, s1, c2, s2, EPS, True)
    return qn


wqkv = [torch.randn(NQKV, H, **bf) * 0.02 for _ in range(a.nw)]
wqb = [torch.randn(NQB, QL, **bf) * 0.02 for _ in range(a.nw)]
wq = torch.rand(QL, **bf) + 0.5
wkv = torch.rand(KVL, **bf) + 0.5
P = 1 << 20
inv = 1.0 / (10000 ** (torch.arange(0, PE, 2, device=dev).float() / PE))
fr = torch.arange(P, device=dev).float()[:, None] * inv[None]
cos_sin = torch.cat((fr.cos(), fr.sin()), -1).to(torch.bfloat16)


def caches(m):
    c1 = torch.zeros(4096, 64, ROW, device=dev, dtype=torch.uint8)
    nb2 = max(16, (m + 63) // 64)
    c2 = torch.zeros(nb2, 64, ROW, device=dev, dtype=torch.uint8)
    s1 = torch.randperm(4096 * 64, device=dev)[:m]
    s1[m // 2] = -1  # a padded token
    s2 = torch.randperm(nb2 * 64, device=dev)[:m]
    return c1, s1, c2, s2


def check(rows, scale):
    qkv = torch.randn(rows, NQKV, **bf) * torch.rand(rows, 1, **bf) * scale
    pos = torch.randint(0, P, (rows,), device=dev)
    c1, s1, c2, s2 = caches(rows)
    r1, r2 = c1.clone(), c2.clone()
    qr, kvr, kper = ref_c(qkv, pos, cos_sin, wq, wkv)
    served_kv(kvr, kper, r1, s1, r2, s2)
    qn = glue2(qkv, pos, cos_sin, wq, wkv, c1, s1, c2, s2)
    print(f"  rows {rows:>4}: q_c mismatches {int((qn != qr).sum())}/{qn.numel()}, "
          f"cache bytes {int((c1 != r1).sum())} + {int((c2 != r2).sum())} "
          f"of {2 * (rows - 1) * ROW} written")


print("correctness vs inductor + concat_and_cache_mla:")
for rows in (4, 8, 16, 32):
    check(rows, 4)
check(4096, 4)

for m in a.m:
    x = torch.randn(m, H, **bf)
    pos = torch.randint(0, 400_000, (m,), device=dev)
    cq, cb = dg.config(NQKV, H, m), dg.config(NQB, QL, m)
    print(f"M={m}: decode_gemm cfg qkv_a {cq} q_b {cb}")
    c1, s1, c2, s2 = caches(m)
    stale = ref_c(gemm(x, wqkv[0], *(cq or (None, None)), True), pos, cos_sin, wq, wkv)[0]

    def chain(kind, i):
        qkv = gemm(x, wqkv[i % a.nw], *(cq or (None, None)), True)
        if kind == "glue2":
            qn = glue2(qkv, pos, cos_sin, wq, wkv, c1, s1, c2, s2)
            return gemm(qn, wqb[i % a.nw], *(cb or (None, None)), True)
        if kind == "floor":
            return gemm(stale, wqb[i % a.nw], *(cb or (None, None)), True)
        qn, kvn, kpe = (ref_c if kind == "served" else glue)(qkv, pos, cos_sin, wq, wkv)
        y = gemm(qn, wqb[i % a.nw], *(cb or (None, None)), True)
        served_kv(kvn, kpe, c1, s1, c2, s2)
        return y

    res = {}
    for kind in ("served", "glue", "glue2", "floor"):
        for i in range(3):
            chain(kind, i)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for i in range(a.reps):
                chain(kind, i)
        g.replay()
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        ts = []
        for _ in range(5):
            e0.record()
            for _ in range(a.iters):
                g.replay()
            e1.record()
            torch.cuda.synchronize()
            ts.append(e0.elapsed_time(e1) * 1e3 / (a.iters * a.reps))
        res[kind] = sorted(ts)[2]
        print(f"  {kind:<7} {res[kind]:6.2f} us/chain (median of 5: "
              f"{' '.join(f'{t:.2f}' for t in sorted(ts))})")
    print(f"  glue2 saves {res['served'] - res['glue2']:.2f} us/layer vs served "
          f"({res['glue'] - res['glue2']:.2f} vs glue); "
          f"{res['glue2'] - res['floor']:.2f} over the floor")

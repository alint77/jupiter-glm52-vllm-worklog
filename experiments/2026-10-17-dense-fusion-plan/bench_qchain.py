"""qkv_a -> glue -> q_b at decode sizes: inductor glue vs glue.cu.

Checks glue.cu against the eager and the inductor-compiled reference (bitwise
fraction, max diff), then times the chain in one CUDA graph of REPS chains over
NW rotating weight sets (cold L2, as in the model where other layers' weights
evict these). Variants:
  inductor   qkv_a -> torch.compile'd split/RMSNorm x2/k_pe RoPE -> q_b
  glue       qkv_a -> glue.cu, PDL (releases q_b at entry) -> q_b
  glue-nopdl qkv_a -> glue.cu without PDL -> q_b
  floor      qkv_a -> q_b on a stale input (no glue; not a valid chain)
    bench_qchain.py [--m 8 16] [--reps 64] [--nw 8]
"""

import argparse
from pathlib import Path

import torch
import torch._inductor.config as ic
from torch.utils.cpp_extension import load

from vllm.model_executor.layers import decode_gemm as dg

ap = argparse.ArgumentParser()
ap.add_argument("--m", type=int, nargs="+", default=[8, 16])
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
gemm = dg._extension().gemm
ic.combo_kernels = True
dev = torch.device("cuda:0")
torch.manual_seed(0)
EPS = 1e-5
QL, KVL, PE, H = 2048, 512, 64, 6144
NQKV, NQB = QL + KVL + PE, 4096


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


def glue(qkv, pos, cos_sin, wq, wkv, pdl=True):
    m = qkv.shape[0]
    qn = torch.empty(m, QL, device=dev, dtype=torch.bfloat16)
    kvn = torch.empty(m, KVL, device=dev, dtype=torch.bfloat16)
    kpe = torch.empty(m, PE, device=dev, dtype=torch.bfloat16)
    ext.glue(qkv, pos, cos_sin, wq, wkv, qn, kvn, kpe, EPS, pdl)
    return qn, kvn, kpe


def cmp(name, got, want):
    eq = [int((g != w).sum()) for g, w in zip(got, want)]
    md = [(g.float() - w.float()).abs().max().item() for g, w in zip(got, want)]
    print(f"  {name:<22} mismatches q {eq[0]} kv {eq[1]} kpe {eq[2]}"
          f" | max diff {md[0]:.3g} {md[1]:.3g} {md[2]:.3g}")


bf = dict(device=dev, dtype=torch.bfloat16)
wqkv = [torch.randn(NQKV, H, **bf) * 0.02 for _ in range(a.nw)]
wqb = [torch.randn(NQB, QL, **bf) * 0.02 for _ in range(a.nw)]
wq = torch.rand(QL, **bf) + 0.5
wkv = torch.rand(KVL, **bf) + 0.5
P = 1 << 20
inv = 1.0 / (10000 ** (torch.arange(0, PE, 2, device=dev).float() / PE))
fr = torch.arange(P, device=dev).float()[:, None] * inv[None]
cos_sin = torch.cat((fr.cos(), fr.sin()), -1).to(torch.bfloat16)

big = torch.randn(4096, NQKV, **bf) * torch.rand(4096, 1, **bf) * 4
bpos = torch.randint(0, P, (4096,), device=dev)
print("4096 rows, varied scales:")
cmp("glue.cu vs inductor", glue(big, bpos, cos_sin, wq, wkv),
    ref_c(big, bpos, cos_sin, wq, wkv))
cmp("glue.cu vs eager", glue(big, bpos, cos_sin, wq, wkv),
    ref(big, bpos, cos_sin, wq, wkv))

for m in a.m:
    x = torch.randn(m, H, **bf)
    pos = torch.randint(0, 400_000, (m,), device=dev)
    cq, cb = dg.config(NQKV, H, m), dg.config(NQB, QL, m)
    print(f"M={m}: decode_gemm cfg qkv_a {cq} q_b {cb}")
    if cq is None or cb is None:
        print("  a GEMM is on cuBLAS at this M; skipped")
        continue
    qkv = gemm(x, wqkv[0], *cq, True)
    want = ref(qkv, pos, cos_sin, wq, wkv)
    wantc = ref_c(qkv, pos, cos_sin, wq, wkv)
    got = glue(qkv, pos, cos_sin, wq, wkv)
    cmp("inductor vs eager", wantc, want)
    cmp("glue.cu vs inductor", got, wantc)
    cmp("glue.cu vs eager", got, want)
    stale = got[0]

    def chain(kind, i):
        qkv = gemm(x, wqkv[i % a.nw], *cq, True)
        if kind == "inductor":
            qn = ref_c(qkv, pos, cos_sin, wq, wkv)[0]
        elif kind == "floor":
            qn = stale
        else:
            qn = glue(qkv, pos, cos_sin, wq, wkv, kind == "glue")[0]
        return gemm(qn, wqb[i % a.nw], *cb, True)

    res = {}
    for kind in ("inductor", "glue", "glue-nopdl", "floor"):
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
        print(f"  {kind:<11} {res[kind]:6.2f} us/chain (median of 5: "
              f"{' '.join(f'{t:.2f}' for t in sorted(ts))})")
    print(f"  glue saves {res['inductor'] - res['glue']:.2f} us/chain; "
          f"inductor glue costs {res['inductor'] - res['floor']:.2f} over the floor")

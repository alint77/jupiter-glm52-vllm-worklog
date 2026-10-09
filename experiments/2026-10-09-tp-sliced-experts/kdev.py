#!/usr/bin/env python3
"""Kernel dev loop for the sliced tiered decode MoE (INTER = 512 per GPU).

Variants are "<source>:<defines>", source = kernels/<source>.cu. Every build
is its own extension (name from a hash of source + defines) with -lineinfo
and the ptxas report; `build` also writes the SASS.

  build  --v td_v0:"TD_INTER=512 ..."   -> kdev/<name>/{ptxas.txt,sass.txt}
  roof   [--numa-node 0]                 HBM / C2C / concurrent read GB/s
  bench  --v V --m 8 --cells 40,4 ...    us, HBM / C2C GB/s, roofline share
  once   --v V --m 8 --cell 40,4 [--n 3] [--trace out.pt]   eager calls (ncu)

Run NUMA-bound on the GPU's Grace node.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bench_slice import all_routing  # noqa: E402

HIDDEN, TOPK = 6144, 8
OUT = HERE / "kdev"
# measured on the node by `roof` (GB/s); bench prints shares against these
ROOF = {"hbm": float(os.environ.get("ROOF_HBM", 0) or 0),
        "c2c": float(os.environ.get("ROOF_C2C", 0) or 0)}


def version(v):
    src = v.partition(":")[0]
    return int(src.split("_v")[1]) if "_v" in src else 0


def parse(v):
    src, _, defines = v.partition(":")
    return src, defines.split()


def ext_name(v):
    src, defines = parse(v)
    h = hashlib.sha1((HERE / f"kernels/{src}.cu").read_bytes()
                     + " ".join(defines).encode()).hexdigest()[:10]
    return f"kdev_{src}_{h}"


class PlainMod:
    """A torch-free kernel library (v13+): nvcc -shared, called via ctypes with
    the torch extension's forward signature."""

    def __init__(self, so):
        import ctypes

        self.lib = ctypes.CDLL(str(so))
        P, I, F = ctypes.c_void_p, ctypes.c_int, ctypes.c_float
        self.lib.td_forward.argtypes = [P, P, P, P, P, P, I, P, P, P, P, I, P, P, P, P, I, P, I, P,
                                        P, P, F, P]
        self.lib.td_forward.restype = I
        self.lib.td_workspace_bytes.restype = ctypes.c_longlong
        self.trace = hasattr(self.lib, "td_dump")

    def workspace_bytes(self):
        return self.lib.td_workspace_bytes()

    def forward(self, out, x, ids, wt, hot_map, cold_map, _p, _s, _h, _rank, _sched,
                hw13, hs13, hw2, hs2, cw13, cs13, cw2, cs2, ws, pdl, padding,
                sw13=None, sw2=None, sscale=1.0):
        ptr = lambda t: t.data_ptr() if t is not None and t.numel() else None  # noqa: E731
        assert ids.dtype == torch.int32
        rc = self.lib.td_forward(
            ptr(out), ptr(x), ptr(ids), ptr(wt), ptr(hot_map), ptr(cold_map), x.shape[0],
            ptr(hw13), ptr(hs13), ptr(hw2), ptr(hs2), hw13.shape[0] if hw13.numel() else 0,
            ptr(cw13), ptr(cs13), ptr(cw2), ptr(cs2), cw13.shape[0] if cw13.numel() else 0,
            ptr(ws), int(bool(pdl)), ptr(padding), ptr(sw13), ptr(sw2), float(sscale),
            torch.cuda.current_stream().cuda_stream)
        assert rc == 0, f"td_forward returned {rc}"

    def td_dump(self):
        import ctypes

        buf = torch.empty((1 << 18, 4), dtype=torch.int64)
        n = self.lib.td_dump(ctypes.c_void_p(buf.data_ptr()), 1 << 18)
        return buf[:n].clone()


def plain_build(v, verbose=False):
    """nvcc the kernel alone into a shared library (seconds, no torch headers)."""
    src, defines = parse(v)
    name = ext_name(v)
    bdir = Path(os.environ.get("VLLM_CACHE_ROOT", "/tmp")) / "kdev" / name
    bdir.mkdir(parents=True, exist_ok=True)
    so = bdir / f"{name}.so"
    if not so.exists():
        cmd = ["nvcc", "-shared", "-Xcompiler", "-fPIC", "-O3",
               "-gencode=arch=compute_90a,code=sm_90a", "-std=c++17", "-lineinfo",
               "-Xptxas=-v", *(f"-D{d}" for d in defines), str(HERE / f"kernels/{src}.cu"),
               "-o", str(so) + ".tmp", "-lcuda"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        (bdir / "build.log").write_text(r.stdout + r.stderr)
        if r.returncode:
            raise RuntimeError(f"nvcc failed for {v}:\n{r.stderr[-3000:]}")
        os.replace(str(so) + ".tmp", so)
        if verbose:
            print(r.stderr)
    return PlainMod(so), bdir


def build(v, verbose=False):
    if version(v) >= 13:
        return plain_build(v, verbose)
    from torch.utils.cpp_extension import load

    src, defines = parse(v)
    name = ext_name(v)
    bdir = Path(os.environ.get("VLLM_CACHE_ROOT", "/tmp")) / "kdev" / name
    bdir.mkdir(parents=True, exist_ok=True)
    mod = load(name=name, sources=[str(HERE / f"kernels/{src}.cu")],
               extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a",
                                  "-std=c++17", "-lineinfo", "-Xptxas=-v",
                                  *(f"-D{d}" for d in defines)],
               extra_ldflags=["-lcuda"], build_directory=str(bdir), verbose=verbose)
    return mod, bdir


def build_many(variants):
    """Build several variants at once (one nvcc per variant)."""
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=len(variants)) as ex:
        list(ex.map(lambda v: build(v) if version(v) >= 13 else None, variants))


def cmd_build(a):
    mod, bdir = build(a.v, verbose=True)
    out = OUT / ext_name(a.v)
    out.mkdir(parents=True, exist_ok=True)
    so = next(bdir.glob("*.so"))
    if version(a.v) >= 13:
        print((bdir / "build.log").read_text())
    sass = subprocess.run(["cuobjdump", "-sass", str(so)], capture_output=True, text=True).stdout
    (out / "sass.txt").write_text(sass)
    (out / "variant.txt").write_text(a.v + "\n")
    print(f"built {ext_name(a.v)}; sass {len(sass.splitlines())} lines -> {out}")


SLICE_BYTES = None


def tier_shapes(inter, e):
    return {"w13_weight_packed": ((e, HIDDEN // 16, 2 * inter * 2), torch.int32),
            "w2_weight_packed": ((e, inter // 16, HIDDEN * 2), torch.int32),
            "w13_weight_scale": ((e, HIDDEN // 32, 2 * inter), torch.bfloat16),
            "w2_weight_scale": ((e, inter // 32, HIDDEN), torch.bfloat16)}


def expert_bytes(inter):
    return sum(torch.Size(s[1:]).numel() * torch.empty(0, dtype=d).element_size()
               for s, d in tier_shapes(inter, 1).values())


class Setup:
    def __init__(self, v, m, pool_hot, pool_cold, numa, inter=512, rand=False, shared=False):
        from vllm.model_executor.offloader.grace import GraceAllocation

        self.mod, _ = build(v)
        self.ver, self.shared = version(v), shared
        self.sw13 = (torch.randn((2 * inter, HIDDEN), device="cuda") * 0.02).to(torch.bfloat16)
        self.sw2 = (torch.randn((HIDDEN, inter), device="cuda") * 0.02).to(torch.bfloat16)
        self.ws = torch.zeros(self.mod.workspace_bytes(), dtype=torch.uint8, device="cuda")
        dev = torch.device("cuda:0")
        self.hot = {}
        for k, (s, d) in tier_shapes(inter, pool_hot).items():
            # random codes: zero weights could flatter a decode with value shortcuts
            t = torch.empty(s, dtype=d, device=dev)
            if d == torch.int32:
                t.random_()
            else:
                t.uniform_(1e-3, 1e-2)
            self.hot[k] = t
        self.keep, self.cold = [], {}
        for k, (s, d) in tier_shapes(inter, pool_cold).items():
            al = GraceAllocation.allocate_pinned(s, d, 0, numa)
            self.keep.append(al)
            self.cold[k] = al.cuda_alias
            self.cold[k].copy_(self.hot[k][: s[0]] if s[0] <= pool_hot else
                               torch.zeros(s, dtype=d, device=dev))
        self.m, self.pool_hot, self.pool_cold = m, pool_hot, pool_cold
        self.gen = torch.Generator().manual_seed(m)
        self.x = (torch.randn((m, HIDDEN), generator=self.gen) * 0.3).to(torch.bfloat16).cuda()

    def calls(self, h, c, n=20):
        out = []
        for _ in range(n):
            ids = all_routing(self.m, h, c, self.gen)
            hm = torch.full((512,), -1, dtype=torch.int32)
            cm = torch.full((512,), -1, dtype=torch.int32)
            hm[:h] = torch.randperm(self.pool_hot, generator=self.gen)[:h].to(torch.int32)
            cm[256:256 + c] = torch.randperm(self.pool_cold, generator=self.gen)[:c].to(torch.int32)
            out.append([t.cuda() for t in (ids, torch.rand((self.m, TOPK), generator=self.gen),
                                           hm, cm)])
        return out

    def parts(self, t):
        return (t["w13_weight_packed"], t["w13_weight_scale"], t["w2_weight_packed"],
                t["w2_weight_scale"])

    def side_shared(self):
        """Today's path: the shared expert as cuBLAS GEMMs on a side stream,
        joined before the output is used."""
        import torch.nn.functional as F

        if not hasattr(self, "side"):
            self.side = torch.cuda.Stream()
            self.xb = self.x.clone()
        main = torch.cuda.current_stream()
        self.side.wait_stream(main)
        with torch.cuda.stream(self.side):
            h = F.linear(self.xb, self.sw13)
            act = F.silu(h[:, :512]) * h[:, 512:]
            self.sh_out = F.linear(act, self.sw2)
        return self.side

    def run(self, cl, out=None):
        e = torch.empty(0, device="cuda")
        out = torch.empty_like(self.x) if out is None else out
        extra = ()
        if self.ver >= 2:
            extra = (self.sw13, self.sw2, 0.4) if self.shared else (e, e, 1.0)
        self.mod.forward(out, self.x, cl[0], cl[1], cl[2], cl[3], e, e, e, 0, False,
                         *self.parts(self.hot), *self.parts(self.cold), self.ws, True, e,
                         *extra)
        return out


def time_graph(fns, reps=10):
    for f in fns[:2]:
        f()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for f in fns:
            f()
    for _ in range(10):
        g.replay()
    torch.cuda.synchronize()
    best = 1e9
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(5):
            g.replay()
        e.record()
        torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e) * 1000 / (5 * len(fns)))
    return best


def cmd_bench(a):
    cells = [tuple(map(int, c.split(","))) for c in a.cells]
    st = Setup(a.v, a.m, max(h for h, _ in cells) + 16, max(c for _, c in cells) + 8,
               a.numa_node, shared=a.shared)
    eb = expert_bytes(512)
    sb = 3 * 512 * HIDDEN * 2 if (a.shared or a.side_shared) else 0  # shared slice bytes
    for h, c in cells:
        cls = st.calls(h, c)
        if a.side_shared:
            def call(cl):
                side = st.side_shared()
                o = st.run(cl)
                torch.cuda.current_stream().wait_stream(side)
                return o + st.sh_out
            us = time_graph([lambda cl=cl: call(cl) for cl in cls])
        else:
            us = time_graph([lambda cl=cl: st.run(cl) for cl in cls])
        hbm, c2c = (h * eb + sb) / us / 1e3, c * eb / us / 1e3  # GB/s
        rec = {"v": a.v, "m": a.m, "hot": h, "cold": c,
               "shared": "side" if a.side_shared else bool(a.shared),
               "us": round(us, 2), "hbm_gbs": round(hbm), "c2c_gbs": round(c2c)}
        if ROOF["hbm"] and ROOF["c2c"]:
            floor = max((h * eb + sb) / ROOF["hbm"], c * eb / ROOF["c2c"]) / 1e3  # us
            rec["roof_us"] = round(floor, 1)
            rec["roof_share"] = round(floor / us, 3)
        print(json.dumps(rec), flush=True)


def cmd_once(a):
    h, c = map(int, a.cell.split(","))
    st = Setup(a.v, a.m, h + 16, c + 8, a.numa_node, shared=a.shared)
    cls = st.calls(h, c, n=max(a.n, 2))
    for cl in cls[:2]:
        st.run(cl)  # warm
    torch.cuda.synchronize()
    if a.trace and hasattr(st.mod, "td_dump"):
        st.mod.td_dump()
    torch.cuda.cudart().cudaProfilerStart()
    for cl in cls[: a.n]:
        st.run(cl)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    if a.trace and hasattr(st.mod, "td_dump"):
        torch.save(st.mod.td_dump(), a.trace)
    print("once done", flush=True)


_CHECK_CACHE = {}


def check_fixture(dev):
    """Checkpoint experts, their 4 slices as hot / cold tiers, fp32 dequants,
    and the shared expert (bf16) with its slices; cached per process."""
    import bench_slice as BS

    if "f" in _CHECK_CACHE:
        return _CHECK_CACHE["f"]
    T = BS.T
    gen = torch.Generator().manual_seed(0)
    n_hot, n_cold = 6, 2
    ck = T._int4_experts(n_hot + n_cold, gen)
    slices = [T._int4_marlin_tier(*BS.slice_ckpt(*ck, r), dev) for r in range(4)]

    def split(tier):
        hot = {k: v[:n_hot].contiguous() for k, v in tier.items()}
        cold = T._to_grace({k: v[n_hot:].contiguous() for k, v in tier.items()}, dev)
        return hot, cold

    sg = torch.Generator().manual_seed(7)
    sw13 = (torch.randn((4096, HIDDEN), generator=sg) * 0.02).to(torch.bfloat16)
    sw2 = (torch.randn((HIDDEN, 2048), generator=sg) * 0.02).to(torch.bfloat16)
    rows = lambda r: torch.cat([torch.arange(512 * r, 512 * r + 512),  # noqa: E731
                                2048 + torch.arange(512 * r, 512 * r + 512)])
    f = {"n_hot": n_hot, "n_cold": n_cold, "tiers": [split(sl) for sl in slices],
         "w13": T._int4_dequant(ck[0], ck[2]).to(dev), "w2": T._int4_dequant(ck[1], ck[3]).to(dev),
         "sw13": sw13.to(dev), "sw2": sw2.to(dev),
         "sslices": [(sw13[rows(r)].contiguous().to(dev),
                      sw2[:, 512 * r:512 * r + 512].contiguous().to(dev)) for r in range(4)]}
    _CHECK_CACHE["f"] = f
    return f


def reference(f, x, ids, wt, shared, sscale):
    """fp32 routed (+ shared) output on the GPU, per expert over its tokens."""
    xf = x.float()
    ref = torch.zeros_like(xf)
    for ex in range(f["w13"].shape[0]):
        sel = ids == ex  # [m, k]
        tok = sel.any(1).nonzero().flatten()
        if len(tok) == 0:
            continue
        h = xf[tok] @ f["w13"][ex].T
        act = torch.nn.functional.silu(h[:, :2048]) * h[:, 2048:]
        ref[tok] += (wt[tok] * sel[tok]).sum(1, keepdim=True) * (act @ f["w2"][ex].T)
    if shared:
        h = xf @ f["sw13"].float().T
        act = (torch.nn.functional.silu(h[:, :2048]) * h[:, 2048:]).to(torch.bfloat16).float()
        ref += sscale * (act @ f["sw2"].float().T)
    return ref


def cmd_check(a):
    """The four 512 slices of the same checkpoint experts through variant V,
    summed, against fp32; per case `--reps` calls (counters / epochs reset,
    races)."""
    dev = torch.device("cuda:0")
    f = check_fixture(dev)
    mod, _ = build(a.v)
    ver = version(a.v)
    sh, sscale = ver >= 2, 0.4
    ws = torch.zeros(mod.workspace_bytes(), dtype=torch.uint8, device=dev)
    e = torch.empty(0, device=dev)
    n_hot, n_cold = f["n_hot"], f["n_cold"]
    hm = torch.full((16,), -1, dtype=torch.int32, device=dev)
    cm = torch.full((16,), -1, dtype=torch.int32, device=dev)
    hm[:n_hot] = torch.arange(n_hot, dtype=torch.int32, device=dev)
    cm[n_hot:n_hot + n_cold] = torch.arange(n_cold, dtype=torch.int32, device=dev)

    def parts(t):
        return (t["w13_weight_packed"], t["w13_weight_scale"], t["w2_weight_packed"],
                t["w2_weight_scale"])

    worst, fails = 0.0, 0
    for m in (1, 8, 16, 32):
        for case in range(4 if sh else 3):
            g = torch.Generator().manual_seed(100 * m + case)
            x = (torch.randn((m, HIDDEN), generator=g) * 0.3).to(torch.bfloat16)
            ids = torch.stack([torch.randperm(n_hot + n_cold, generator=g)[:TOPK]
                               for _ in range(m)]).to(torch.int32)
            if case == 2:  # hot only
                ids = torch.cat([torch.stack([torch.randperm(n_hot, generator=g) for _ in range(m)]),
                                 torch.full((m, 2), -1, dtype=torch.int64)], 1).to(torch.int32)
            if case == 3:  # shared expert only
                ids = torch.full((m, TOPK), -1, dtype=torch.int32)
            wt = torch.rand((m, TOPK), generator=g)
            x, ids, wt = x.to(dev), ids.to(dev), wt.to(dev)
            ref = reference(f, x, ids, wt, sh, sscale)
            for rep in range(a.reps):
                got = torch.zeros((m, HIDDEN), device=dev)
                for r, (hot, cold) in enumerate(f["tiers"]):
                    out = torch.empty((m, HIDDEN), dtype=torch.bfloat16, device=dev)
                    extra = (*f["sslices"][r], sscale) if sh else ()
                    mod.forward(out, x, ids, wt, hm, cm, e, e, e, 0, False, *parts(hot),
                                *parts(cold), ws, True, e, *extra)
                    got += out.float()
                err = float((got - ref).abs().max() / ref.abs().max().clamp_min(1e-30))
                worst = max(worst, err)
                if err > 6e-3:
                    fails += 1
                    print(json.dumps({"m": m, "case": case, "rep": rep, "rel_err": round(err, 5)}),
                          flush=True)
    print(json.dumps({"worst_rel_err": round(worst, 5), "fails": fails, "pass": fails == 0}))


ROOF_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
__global__ void rd(const uint4* __restrict__ a, long na, int ba,
                   const uint4* __restrict__ b, long nb,
                   unsigned long long* t, uint4* sink) {
  const bool first = blockIdx.x < ba;
  const uint4* p = first ? a : b;
  const long n = first ? na : nb;
  const int nblk = first ? ba : gridDim.x - ba;
  const int bid = first ? blockIdx.x : blockIdx.x - ba;
  unsigned long long t0;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
  uint4 acc = make_uint4(0, 0, 0, 0);
  const long stride = (long)nblk * blockDim.x;
  long i = (long)bid * blockDim.x + threadIdx.x;
  #pragma unroll 8
  for (; i < n; i += stride) {
    uint4 v;
    asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p + i));
    acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
  }
  __syncthreads();
  unsigned long long t1;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t1));
  if (threadIdx.x == 0) { atomicMin(&t[first ? 0 : 2], t0); atomicMax(&t[first ? 1 : 3], t1); }
  if (acc.x == 0x9e3779b9u && acc.y == 1u) sink[0] = acc;
}
torch::Tensor read_bw(torch::Tensor a, torch::Tensor b, int64_t ba, int64_t bb) {
  auto t = torch::tensor({(int64_t)-1, (int64_t)0, (int64_t)-1, (int64_t)0},
                         torch::dtype(torch::kInt64).device(a.device()));
  auto sink = torch::empty({16}, a.options());
  rd<<<ba + bb, 512>>>((const uint4*)a.data_ptr(), a.numel() / 16, ba,
                       (const uint4*)b.data_ptr(), b.numel() / 16,
                       (unsigned long long*)t.data_ptr(), (uint4*)sink.data_ptr());
  return t;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("read_bw", &read_bw); }
"""


def cmd_roof(a):
    from torch.utils.cpp_extension import load_inline
    from vllm.model_executor.offloader.grace import GraceAllocation

    mod = load_inline("kdev_roof", cpp_sources="", cuda_sources=ROOF_SRC,
                      extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"])
    hbm = torch.empty(1 << 30, dtype=torch.uint8, device="cuda").random_()
    al = GraceAllocation.allocate_pinned((256 << 20,), torch.uint8, 0, a.numa_node)
    c2c = al.cuda_alias
    c2c.copy_(hbm[: 256 << 20])

    def run(nh, ha, nc, cb, reps=5):
        best = None
        for _ in range(reps):
            t = mod.read_bw(ha, cb, nh, nc).cpu().tolist()
            torch.cuda.synchronize()
            r = (ha.numel() / (t[1] - t[0]) if nh else 0, cb.numel() / (t[3] - t[2]) if nc else 0)
            best = r if best is None or sum(r) > sum(best) else best
        return best

    empty = hbm[:16]
    for nh in (132, 264):
        print(json.dumps({"case": "hbm only", "blocks": nh, "hbm_gbs": round(run(nh, hbm, 0, empty)[0])}))
    for nc in (8, 16, 24, 32, 48, 132):
        print(json.dumps({"case": "c2c only", "blocks": nc, "c2c_gbs": round(run(0, empty, nc, c2c)[1])}))
    for nh, nc in ((116, 16), (108, 24), (100, 32)):
        h, c = run(nh, hbm[: 512 << 20], nc, c2c)
        print(json.dumps({"case": "concurrent", "hbm_blocks": nh, "c2c_blocks": nc,
                          "hbm_gbs": round(h), "c2c_gbs": round(c)}))
    del al


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("build", "bench", "once", "roof", "check"):
        p = sub.add_parser(name)
        p.add_argument("--v")
        p.add_argument("--m", type=int, default=8)
        p.add_argument("--cells", nargs="+", default=[])
        p.add_argument("--cell")
        p.add_argument("--n", type=int, default=3)
        p.add_argument("--trace")
        p.add_argument("--numa-node", type=int, default=0)
        p.add_argument("--shared", type=int, default=0)
        p.add_argument("--side-shared", type=int, default=0)
        p.add_argument("--reps", type=int, default=1)
    a = ap.parse_args()
    {"build": cmd_build, "bench": cmd_bench, "once": cmd_once, "roof": cmd_roof,
     "check": cmd_check}[a.cmd](a)


if __name__ == "__main__":
    main()

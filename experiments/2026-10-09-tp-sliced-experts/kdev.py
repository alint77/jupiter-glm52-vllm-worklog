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


def build(v, verbose=False):
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


def cmd_build(a):
    mod, bdir = build(a.v, verbose=True)
    out = OUT / ext_name(a.v)
    out.mkdir(parents=True, exist_ok=True)
    so = next(bdir.glob("*.so"))
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


def cmd_check(a):
    """The four 512 slices of the same checkpoint experts through variant V,
    summed, against fp32; three calls per case (counters / epochs reset)."""
    import bench_slice as BS

    T = BS.T
    dev = torch.device("cuda:0")
    gen = torch.Generator().manual_seed(0)
    n_hot, n_cold = 6, 2
    ck = T._int4_experts(n_hot + n_cold, gen)
    slices = [T._int4_marlin_tier(*BS.slice_ckpt(*ck, r), dev) for r in range(4)]

    def split(tier):
        hot = {k: v[:n_hot].contiguous() for k, v in tier.items()}
        cold = T._to_grace({k: v[n_hot:].contiguous() for k, v in tier.items()}, dev)
        return hot, cold

    slice_t = [split(sl) for sl in slices]
    mod, _ = build(a.v)
    ver = version(a.v)
    ws = torch.zeros(mod.workspace_bytes(), dtype=torch.uint8, device=dev)
    sh = ver >= 2
    sscale = 0.4
    sg = torch.Generator().manual_seed(7)
    sw13 = (torch.randn((4096, HIDDEN), generator=sg) * 0.02).to(torch.bfloat16)  # gate | up
    sw2 = (torch.randn((HIDDEN, 2048), generator=sg) * 0.02).to(torch.bfloat16)

    def shared_slice(r):
        rows = torch.cat([torch.arange(512 * r, 512 * r + 512),
                          2048 + torch.arange(512 * r, 512 * r + 512)])
        return (sw13[rows].contiguous().to(dev),
                sw2[:, 512 * r:512 * r + 512].contiguous().to(dev))

    sslices = [shared_slice(r) for r in range(4)]
    w13 = T._int4_dequant(ck[0], ck[2])
    w2 = T._int4_dequant(ck[1], ck[3])
    e = torch.empty(0, device=dev)

    def parts(t):
        return (t["w13_weight_packed"], t["w13_weight_scale"], t["w2_weight_packed"],
                t["w2_weight_scale"])

    worst = 0.0
    for m in (1, 8, 16, 32):
        for case in range(4 if version(a.v) >= 2 else 3):
            g = torch.Generator().manual_seed(100 * m + case)
            x = (torch.randn((m, HIDDEN), generator=g) * 0.3).to(torch.bfloat16)
            k = min(TOPK, n_hot + n_cold)
            ids = torch.stack([torch.randperm(n_hot + n_cold, generator=g)[:k]
                               for _ in range(m)]).to(torch.int32)
            if case == 2:  # hot only
                ids = torch.stack([torch.randperm(n_hot, generator=g)
                                   for _ in range(m)]).to(torch.int32)
                ids = torch.cat([ids, torch.full((m, 2), -1, dtype=torch.int32)], 1)
            if case == 3:  # shared expert only
                ids = torch.full((m, TOPK), -1, dtype=torch.int32)
            wt = torch.rand((m, TOPK), generator=g)
            hm = torch.full((16,), -1, dtype=torch.int32)
            cm = torch.full((16,), -1, dtype=torch.int32)
            hm[:n_hot] = torch.arange(n_hot, dtype=torch.int32)
            cm[n_hot:n_hot + n_cold] = torch.arange(n_cold, dtype=torch.int32)
            ref = torch.zeros((m, HIDDEN))
            xf = x.float()
            for t in range(m):
                for kk in range(TOPK):
                    ex = int(ids[t, kk])
                    if ex < 0:
                        continue
                    h = w13[ex] @ xf[t]
                    act = torch.nn.functional.silu(h[:2048]) * h[2048:]
                    ref[t] += wt[t, kk] * (w2[ex] @ act)
            if sh:
                h = sw13.float() @ x.float().T  # [4096, m]
                act = torch.nn.functional.silu(h[:2048]) * h[2048:]
                act = act.to(torch.bfloat16).float()
                ref += sscale * (sw2.float() @ act).T
            args = [t.to(dev) for t in (x, ids, wt, hm, cm)]
            got = torch.zeros((m, HIDDEN), device=dev)
            for r, (hot, cold) in enumerate(slice_t):
                out = torch.empty((m, HIDDEN), dtype=torch.bfloat16, device=dev)
                extra = (*sslices[r], sscale) if sh else ()
                mod.forward(out, *args, e, e, e, 0, False, *parts(hot), *parts(cold), ws,
                            True, e, *extra)
                got += out.float()
            err = float((got.cpu() - ref).abs().max() / ref.abs().max().clamp_min(1e-30))
            worst = max(worst, err)
            print(json.dumps({"m": m, "case": case, "rel_err": round(err, 5)}), flush=True)
    print(json.dumps({"worst_rel_err": round(worst, 5), "pass": worst < 6e-3}))


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
    a = ap.parse_args()
    {"build": cmd_build, "bench": cmd_bench, "once": cmd_once, "roof": cmd_roof,
     "check": cmd_check}[a.cmd](a)


if __name__ == "__main__":
    main()

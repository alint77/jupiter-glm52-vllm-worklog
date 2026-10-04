"""Does the shared expert (aux stream) overlap the routed decode MoE kernels,
and does either block the other's CTAs? One GPU, one GLM-5.3 decode MoE layer
x 20 in a CUDA graph, three arms:
  moe     main: router GEMM, top-k, tiered_decode_moe (route_prep, w13, act,
          w2, finalize)
  both    the above, with the shared expert (gate_up GEMM, silu*up, down GEMM,
          TP4 shard: 1024 / 512 wide) forked on an aux stream before the
          router and joined after the MoE, as vLLM does for <= 256 tokens
  shared  the shared expert alone
tiered_decode is built with -DTD_CTA_TRACE: every w13 / w2 CTA logs its SM,
entry, ready (past its PDL wait) and exit (%globaltimer); stamp kernels mark
the aux stream's GEMM boundaries on the same clock.
Run NUMA-bound on the GPU's Grace node.
"""
import importlib.util
import json
import os
import statistics as st
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
spec = importlib.util.spec_from_file_location(
    "bench_fmt", HERE.parent / "2026-09-27-glm53-mtp7-profile/bench_fmt.py")
BF = importlib.util.module_from_spec(spec)
spec.loader.exec_module(BF)
B = BF.B
from vllm.model_executor.layers.fused_moe import tiered_decode as TD  # noqa: E402

N_HOT, N_COLD, LAYERS = int(os.environ.get("N_HOT", 9)), int(os.environ.get("N_COLD", 2)), 20
src = REPO / "vllm/model_executor/layers/fused_moe/tiered_decode/tiered_decode.cu"
build = Path(os.environ["VLLM_CACHE_ROOT"]) / "torch_extensions" / "td_trace"
build.mkdir(parents=True, exist_ok=True)
ext = load(name="vllm_tiered_decode_trace", sources=[str(src)],
           extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-std=c++17",
                              "-DTD_CTA_TRACE"],
           extra_ldflags=["-lcuda"], build_directory=str(build))
TD._extension = lambda: ext
ws = torch.zeros(ext.workspace_bytes(), dtype=torch.uint8, device="cuda")
TD._workspace = lambda device: ws

dev = torch.device("cuda:0")
gen = torch.Generator().manual_seed(0)
hot, _ = BF.tier("int4", 16, dev, False, 0, gen, 300)
cold, keep = BF.tier("int4", 8, dev, True, 0, gen, 100)
x = (torch.randn((B.TOKENS, B.HIDDEN), generator=gen) * 0.3).to(torch.bfloat16).to(dev)
calls = [B.routing(N_HOT, N_COLD, 300, 100, r, gen, dev) for r in range(LAYERS)]
for ids, w, hmap, cmap in calls:  # fresh experts every layer, as in serving
    for m, pool in ((hmap, 300), (cmap, 100)):
        on = m >= 0
        m[on] = torch.randint(0, pool, (int(on.sum()),), device=dev, dtype=m.dtype)
w_router = torch.randn((256, 6144), device=dev).to(torch.bfloat16) * 0.02
w_gu = torch.randn((1024, 6144), device=dev).to(torch.bfloat16) * 0.02
w_d = torch.randn((6144, 512), device=dev).to(torch.bfloat16) * 0.02
aux = torch.cuda.Stream()


def layer(i, with_moe, with_shared):
    main = torch.cuda.current_stream()
    ext.td_stamp(10)
    if with_shared:
        aux.wait_stream(main)
        with torch.cuda.stream(aux):
            ext.td_stamp(20)
            gu = F.linear(x, w_gu)
            ext.td_stamp(21)
            h = F.silu(gu[:, :512]) * gu[:, 512:]
            ext.td_stamp(22)
            d = F.linear(h, w_d)
            ext.td_stamp(23)
    y = None
    if with_moe:
        logits = F.linear(x, w_router)
        torch.topk(logits.float(), 8, dim=-1)
        c = calls[i]
        y = TD.tiered_decode_moe(x, c[0], c[1], c[2], c[3], hot, cold)
    ext.td_stamp(11)
    if with_shared:
        main.wait_stream(aux)
        y = d if y is None else y + d
    ext.td_stamp(12)
    return y


def run(arm):
    with_moe, with_shared = arm in ("moe", "both"), arm in ("shared", "both")
    for i in range(2):
        layer(i, with_moe, with_shared)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(LAYERS):
            layer(i, with_moe, with_shared)
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    ext.td_dump()
    out = []
    for _ in range(10):
        g.replay()
        torch.cuda.synchronize()
        out.append(ext.td_dump().tolist())
    return out


def parse(recs):
    """Split one replay's records into layers by the main stream's stamp 10."""
    rows = sorted(({"ph": (r[0] >> 48) & 0xFFFF, "blk": (r[0] >> 32) & 0xFFFF,
                    "sm": r[0] & 0xFFFFFFFF, "t0": r[1], "t1": r[2], "t2": r[3]}
                   for r in recs), key=lambda r: r["t2"])
    layers, cur = [], None
    for r in rows:
        if r["ph"] == 110:
            cur = {"stamps": {}, "cta": []}
            layers.append(cur)
        if cur is None:
            continue
        if r["ph"] >= 100:
            cur["stamps"][r["ph"] - 100] = r["t2"]
        else:
            cur["cta"].append(r)
    return layers


def summarize(arm, replays):
    res = {"arm": arm}
    acc = {k: [] for k in ("layer", "w13", "w2", "gu", "silu_to_down", "down",
                           "w13_late", "w13_late_after_gu", "down_free_sms")}
    for recs in replays:
        for L in parse(recs)[1:-1]:
            s = L["stamps"]
            acc["layer"].append((s[12] - s[10]) / 1e3)
            if 21 in s:
                acc["gu"].append((s[21] - s[20]) / 1e3)
                acc["down"].append((s[23] - s[22]) / 1e3)
            w13 = [c for c in L["cta"] if c["ph"] == 0]
            w2 = [c for c in L["cta"] if c["ph"] == 1]
            if w13:
                first = min(c["t0"] for c in w13)
                acc["w13"].append((max(c["t2"] for c in w13) - min(c["t1"] for c in w13)) / 1e3)
                late = [c for c in w13 if c["t0"] - first > 3000]
                acc["w13_late"].append(len(late))
                if 21 in s:
                    acc["w13_late_after_gu"].append(
                        sum(1 for c in late if c["t0"] >= s[21] - 1000))
            if w2:
                acc["w2"].append((max(c["t2"] for c in w2) - min(c["t1"] for c in w2)) / 1e3)
            if 22 in s and (w13 or w2):
                # SMs with no resident MoE CTA at the down GEMM's start (its
                # stamp 22 is when it became launchable)
                t = s[22]
                busy = {c["sm"] for c in w13 + w2 if c["t0"] <= t < c["t2"]}
                acc["down_free_sms"].append(132 - len(busy))
    for k, v in acc.items():
        if v:
            res[k] = round(st.median(v), 1)
    return res


for arm in ("moe", "both", "shared"):
    print(json.dumps(summarize(arm, run(arm))), flush=True)

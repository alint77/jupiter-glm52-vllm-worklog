"""Capture the two-tier fork/join under a CUDA graph with the tight policy.

The tiered MoE runs both tiers inside a full CUDA graph, and graph capture has
broken this project twice before. The tight launch changes the dynamic shared
memory request and the grid, both of which are baked into the captured node, so
this checks capture, replay, and replay-equals-eager.
"""

import importlib.util
import os
import sys

import torch

import vllm._custom_ops as ops

_REPO = "/e/project1/profound/alint77/vllm"
spec = importlib.util.spec_from_file_location(
    "hb", os.path.join(_REPO, "benchmarks", "kernels",
                       "benchmark_moe_wna16_marlin_decode.py")
)
hb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hb)

from vllm.model_executor.layers.quantization.utils.marlin_utils import (  # noqa: E402
    marlin_make_workspace_new,
)
from vllm.scalar_type import scalar_types  # noqa: E402

HIDDEN, INTER, TOP_K, BLOCK_M = 6144, 2048, 8, 16
QUANT = scalar_types.uint4b8
M, N_HOT, N_COLD = 4, 6, 1


def build(ids, k, n, dev, pinned, ne, topk):
    q, s = hb.make_tier(len(ids), k, n, dev, pinned, 0)
    emap = hb.tier_map(ids, ne, dev)
    tok, eids, npost, _ = hb.align(topk, BLOCK_M, ne, emap)
    return dict(
        q=q, s=s, ws=marlin_make_workspace_new(dev, 4), tok=tok, eids=eids,
        npost=npost, k=k, n=n,
        c=torch.zeros((M * TOP_K, n), dtype=torch.bfloat16, device=dev),
    )


def call(t, a, w, smem, grid):
    ops.moe_wna16_marlin_gemm(
        a, t["c"], t["q"], None, t["s"], None, None, None, None, None, t["ws"],
        t["tok"], t["eids"], t["npost"], w, moe_block_size=BLOCK_M, top_k=TOP_K,
        mul_topk_weights=False, b_q_type=QUANT, size_m=M, size_n=t["n"],
        size_k=t["k"], is_k_full=True, use_atomic_add=False,
        use_fp32_reduce=True, is_zp_float=False, smem_mode=smem,
        grid_blocks=grid,
    )


def main():
    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    ne = N_HOT + N_COLD
    topk = torch.tensor(
        [[(i * TOP_K + j) % ne for j in range(TOP_K)] for i in range(M)],
        dtype=torch.int32, device=dev,
    )
    hot = build(list(range(N_HOT)), HIDDEN, 2 * INTER, dev, False, ne, topk)
    cold = build(list(range(N_HOT, ne)), HIDDEN, 2 * INTER, dev, True, ne, topk)
    a = torch.randn((M, HIDDEN), dtype=torch.bfloat16, device=dev) / 8
    w = torch.ones((M, TOP_K), dtype=torch.float32, device=dev)
    aux = torch.cuda.Stream()
    gh, gc = sms * 2, sms

    def both():
        main_s = torch.cuda.current_stream()
        aux.wait_stream(main_s)
        with torch.cuda.stream(aux):
            call(cold, a, w, ops.MARLIN_SMEM_TIGHT, gc)
        call(hot, a, w, ops.MARLIN_SMEM_TIGHT, gh)
        main_s.wait_stream(aux)

    for _ in range(5):
        both()
    torch.cuda.synchronize()
    eager = (hot["c"].clone(), cold["c"].clone())

    # warm up on a side stream, as torch's graph capture requires
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            both()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    hot["c"].zero_()
    cold["c"].zero_()
    with torch.cuda.graph(graph):
        both()
    print("capture: OK")

    hot["c"].zero_()
    cold["c"].zero_()
    graph.replay()
    torch.cuda.synchronize()
    replayed = (hot["c"].clone(), cold["c"].clone())
    print(f"replay == eager: hot={torch.equal(replayed[0], eager[0])} "
          f"cold={torch.equal(replayed[1], eager[1])}")

    hot["c"].zero_()
    cold["c"].zero_()
    graph.replay()
    torch.cuda.synchronize()
    print(f"replay repeatable: hot={torch.equal(hot['c'], replayed[0])} "
          f"cold={torch.equal(cold['c'], replayed[1])}")

    ok = (torch.equal(replayed[0], eager[0]) and torch.equal(replayed[1], eager[1])
          and torch.equal(hot["c"], replayed[0]))
    print("RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

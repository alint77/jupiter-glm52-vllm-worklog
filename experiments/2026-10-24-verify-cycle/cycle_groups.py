"""Per-cycle (A + S1 + S2 + S3) wall time by op group from cycle_nsys.py --json.

    cycle_groups.py an/cycle-<hold>-w<n>.json ...
"""
import json
import re
import sys

G = [("MoE (layer_kernel + finalize)", r"layer_kernel|finalize_kernel"),
     ("MoE head: router GEMM + topk + route_prep", r"linearcute|grouped_topk|route_prep"),
     ("all-reduce + RMSNorm (x2)", r"allreduce_fusion"),
     ("qkv_a GEMM + glue + q_b GEMM", r"^decode_gemm_kernel(?: #2)?$|qkv_glue"),
     ("q RoPE (Inductor) + W_UK bmm", r"triton_poi_fused_add_copy|nvjet_sm90_tst_64x16_64x16_2x1"),
     ("DCP q gather (gather_cat)", r"gather_cat"),
     ("FlashMLA + combine", r"flash_fwd"),
     ("DCP LSE reduce-scatter", r"lse_reduce_scatter"),
     ("W_UV bmm + o_proj GEMM", r"nvjet_sm90_tst_64x8|^decode_gemm_kernel #(?:3|4)$"),
     ("indexer k branch (wk, LN, k quant/cache)", r"splitK_TNT|splitKreduce|layer_norm|indexer_k_quant|^vectorized_elementwise_kernel$"),
     ("indexer q branch (wq_b, q rope/quant)", r"_fused_indexer_q_rope|triton_poi_fused_2"),
     ("indexer logits + top-k + DCP merge", r"paged_mqa|cooperative_topk|_pack_dcp|^gather_kernel|StableTopK"),
     ("anchor main-KV writes (2x concat_and_cache)", r"concat_and_cache"),
     ("anchor index convert + hot-cache map (anchor_hot.cu)", r"_convert_req|_dcp_decode_sched|map_kernel|vectorized_elementwise_kernel #2")]
for f in sys.argv[1:]:
    d = json.load(open(f))
    tot, out = 0.0, {}
    for kind in ("A", "S1", "S2", "S3"):
        for r in d["layers"][kind]:
            n = r["kernel"] + (f" #{r['occ']}" if r["occ"] > 1 else "")
            # in anchors the o_proj is decode_gemm #4 and #3 is the indexer wq_b
            if kind == "A" and n == "decode_gemm_kernel #3":
                g = "indexer q branch (wq_b, q rope/quant)"
            else:
                g = next((g for g, p in G if re.search(p, n)), "other")
            out[g] = out.get(g, 0.0) + r["idle"] + r["adds"]
            tot += r["idle"] + r["adds"]
    print(f"\n{f}: cycle wall {d['cycle']:.0f} us, sum of medians {tot:.0f}, step {d['span'] / 1e3:.2f} ms")
    for g, v in sorted(out.items(), key=lambda x: -x[1]):
        print(f"  {v:7.1f} us  {100 * v / tot:5.1f}%  x18 = {18 * v / 1e3:5.2f} ms  {g}")
